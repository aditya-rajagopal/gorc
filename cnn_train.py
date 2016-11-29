import tensorflow as tf
import numpy as np
import parse_data as p
import os
import time
import random
import math

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary(name + '/sttdev', stddev)
            tf.scalar_summary(name + '/max', tf.reduce_max(var))
            tf.scalar_summary(name + '/min', tf.reduce_min(var))
            tf.histogram_summary(name, var)

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

class GOCNN(object):
    def __init__(self, board, target, weights = None, sess = None, iftrain = True):
        self.board = board
        self.target = target
        self.wd = 1e-4
        
        if iftrain:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.cnn_model(self.board, iftrain), self.target), name = 'xentropy_mean')
            self.loss += tf.add_n(tf.get_collection("losses"), name = "total_loss")
            global_step_ad = tf.Variable(0, name = "global_step_ad", trainable = False)
            global_step_gd = tf.Variable(0, name = "global_step_gd", trainable = False)
            self.learning_rate_ad = tf.train.exponential_decay(0.0001, global_step_ad, 4000, 0.95, staircase = True)
            self.learning_rate_gd = tf.train.exponential_decay(0.001, global_step_gd, 4000, 0.95, staircase = True)
            _variable_summaries(self.learning_rate_gd)
            self.train_step_gd = tf.train.GradientDescentOptimizer(self.learning_rate_gd).minimize(self.loss, global_step = global_step_gd)
            self.train_step_ad	= tf.train.AdamOptimizer(self.learning_rate_ad).minimize(self.loss, global_step = global_step_ad)
            self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.fc3),1), tf.argmax(self.target,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            _variable_summaries(self.accuracy)
            self.merged = tf.merge_all_summaries()
            self.writer = tf.train.SummaryWriter("./summaries/logs/train", sess.graph)
        else:
            self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.cnn_model(self.board, iftrain)),1), tf.argmax(self.target,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.initialize_all_variables().run(session = sess)
        
    def _add_wd_and_collection(self, var, wd, collection_name = "losses"):
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name = 'weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        return var
    
    def conv_2d(self, bottom, name, shape, strides, padding):
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", shape = shape, dtype = tf.float32, 
                                     initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, dtype=tf.float32))
            conv = tf.nn.conv2d(bottom, kernel, strides=strides, padding=padding)
            biases = tf.get_variable("biases", shape = [shape[3]], dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(out, name=name)
            self._add_wd_and_collection(kernel, self.wd)
            _activation_summary(conv)
            _variable_summaries(biases)
            return conv
    
    def fc_layer(self, bottom, name, shape, iftrain, dropout = 0.5):
        with tf.variable_scope(name) :
            weights = tf.get_variable("weights", shape = shape, dtype = tf.float32, 
                initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
            biases = tf.get_variable("biases", shape = [shape[1]], dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            fcl = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
            if iftrain:
                fcl = tf.nn.dropout(fcl,dropout)
            self._add_wd_and_collection(weights, self.wd)
            _activation_summary(fcl)
            _variable_summaries(biases)
            return fcl
    """
    def score_layer(self, bottom, name, num_classes, stddev = 0.001, ifdrop = True):
        with tf.variable_scope(name) as scope:
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            initializer = tf.truncated_normal_initializer(stddev = stddev)
            weights = tf.get_variable(name = "weights", shape = shape, initializer = initializer)
            self._add_wd_and_collection(weights, self.wd)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding = "SAME")
            init_bias = tf.constant_initializer(0.0)
            bias = tf.get_variable(name = "bias", shape = [num_classes], initializer = init_bias)
            out = tf.nn.bias_add(conv, bias)
            if ifdrop:
                out = tf.nn.dropout(out, 0.5)
            _activation_summary(out)
            return out
    
    def upscore_layer(self, bottom,name, shape, num_classes, ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.pack(new_shape)

            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            self._add_wd_and_collection(weights, self.wd, "fc_wlosses")
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                            strides=strides, padding='SAME')
            _activation_summary(deconv)
            return tf.nn.relu(deconv)
    """
    def cnn_model(self, board, iftrain):
        conv1_1 = self.conv_2d(board, "conv1_1", [7, 7, 3, 64], [1, 1, 1, 1], "SAME")
        conv1_2 = self.conv_2d(conv1_1, "conv1_2", [5, 5, 64, 64], [1, 1, 1, 1], "SAME")
        conv1_3 = self.conv_2d(conv1_2, "conv1_3", [5, 5, 64, 64], [1, 1, 1, 1], "SAME")
        conv1_4 = self.conv_2d(conv1_3, "conv1_4", [3, 3, 64, 64], [1, 1, 1, 1], "SAME")
        conv1_5 = self.conv_2d(conv1_4, "conv1_5", [3, 3, 64, 64], [1, 1, 1, 1], "SAME")
        #pool1 = tf.nn.max_pool(conv1_3, ksize =[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name = "pool1")

        conv1_6 = self.conv_2d(conv1_5, "conv1_6", [3, 3, 64, 128], [1, 1, 1, 1], "SAME")
        
        conv2_1 = self.conv_2d(conv1_6, "conv2_1", [3, 3, 128, 128], [1, 1, 1, 1], "SAME")
        conv2_2 = self.conv_2d(conv2_1, "conv2_2", [3, 3, 128, 192], [1, 1, 1, 1], "SAME")
        conv2_3 = self.conv_2d(conv2_2, "conv2_3", [3, 3, 192, 192], [1, 1, 1, 1], "SAME")
        #pool2 = tf.nn.max_pool(conv2_3, ksize =[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name = "pool2")
        
        """
        conv3_1 = self.conv_2d(pool2, "conv3_1", [3, 3, 256, 512], [1, 1, 1, 1], "SAME")
        conv3_2 = self.conv_2d(conv3_1, "conv3_2", [3, 3, 512, 512], [1, 1, 1, 1], "SAME")
        pool3 = tf.nn.max_pool(conv3_2, ksize =[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = "pool3")
        """

        #shape = pool3.get_shape()
        h_pool3_flat = tf.reshape(conv2_3, [-1, 19*19*192])
        fc1 = self.fc_layer(h_pool3_flat, "fc1", [19*19*192, 1024], iftrain)
        #fc2 = self.fc_layer(tf.nn.relu(fc1), "fc2", [1024, 1024], iftrain)
        self.fc3 = self.fc_layer(tf.nn.relu(fc1), "fc3", [1024, 361], iftrain = False)
        return self.fc3

if __name__== '__main__':
    sess = tf.Session()
    batch_size = 100
    num_epoch = 200
    train_data, test_data = p.load_data('output.adi')
    board = tf.placeholder(tf.float32, [batch_size, 19, 19, 3])
    target = tf.placeholder(tf.float32, [batch_size, 19*19])
    board_test = tf.placeholder(tf.float32, [1, 19, 19, 3])
    target_test = tf.placeholder(tf.float32, [1, 19*19])
    with tf.variable_scope("model"):
        gocnn = GOCNN(board, target, sess = sess)
    with tf.variable_scope("model", reuse = True):
        gocnn_test = GOCNN(board_test, target_test, sess = sess, iftrain = False)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    for i in range(num_epoch):
        print "EPOCH {} : ---------------------------------".format(i+1)
        random.shuffle(train_data)
        start = time.time()
        accuracy = 0
        for step, (x, y, z) in enumerate(p.batch_iter(train_data, batch_size)):
            if i > 30:
                _, loss, lr, acc, summary = sess.run([gocnn.train_step_ad, gocnn.loss, gocnn.learning_rate_ad, gocnn.accuracy,gocnn.merged ], 
                                                     feed_dict = {gocnn.board:x, gocnn.target:y})
            else:
                _, loss, lr, acc, summary = sess.run([gocnn.train_step_gd, gocnn.loss, gocnn.learning_rate_gd, gocnn.accuracy,gocnn.merged], 
                                                     feed_dict = {gocnn.board:x, gocnn.target:y})
            if step%100 ==0:
                gocnn.writer.add_summary(summary, i*10000/batch_size + step)
            accuracy+=acc
            if step%int(math.ceil(len(train_data)/batch_size/10.0))==0:
                print "step:{}\taccuracy:{:3.3f}\tloss:{:3.5f}\t".format(step,accuracy/(step+1), loss)

        accuracy = 0
        #if (i+1)%2==0 or i==0:
        print "Validation : --------------------------------------------"
        for step, (x, y, z) in enumerate(p.batch_iter(test_data, 1)):
            acc = sess.run(gocnn_test.accuracy, 
                feed_dict = {gocnn_test.board:x, gocnn_test.target:y})
            accuracy+=acc
        print "step:{}\taccuracy:{:3.3f}".format(step, accuracy/(step+1))
        f.write("V:{:0.5f}\n".format(accuracy/(step+1)))
        if (i+1)%10 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=i)
            print("Saved model checkpoint to {}\n".format(path))
