import tensorflow as tf
import numpy as np

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
            self.train_step_gd = tf.train.GradientDescentOptimizer(self.learning_rate_gd).minimize(self.loss, global_step = global_step_gd)
            self.train_step_ad	= tf.train.AdamOptimizer(self.learning_rate_ad).minimize(self.loss, global_step = global_step_ad)
            self.correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.fc3),1), tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
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
            conv = tf.nn.conv2d(blob, kernel, strides=strides, padding=padding)
            biases = tf.get_variable("biases", shape = [shape[3]], dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(out, name=name)
            self._add_wd_and_collection(kernel, self.wd)
            _activation_summary(conv)
            _variable_summaries(biases)
            return conv
    
    def fc_layer(self, bottom, name, shape, iftrain, dropout = 0.5):
        with tf.variable_scope(name) :
            weight = tf.get_variable("weights", shape = shape, dtype = tf.float32, 
                initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
            bias = tf.get_variable("biases", shape = [shape[2]], dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            fcl = tf.nn.bias_add(tf.matmul(blob, weight), bias)
            if iftrain:
                fc = tf.nn.dropout(self.fcl,dropout)
            self._add_wd_and_collection(weight, self.wd)
            _activation_summary(fc)
            _variable_summaries(biases)
            return fc
    
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
    
    def cnn_model(self, board, iftrain):
        conv1_1 = self.conv_2d(board, "conv1_1", [3, 3, 1, 64], [1, 1, 1, 1], "SAME")
        conv1_2 = self.conv_2d(conv1_1, "conv1_2", [3, 3, 64, 64], [1, 1, 1, 1], "SAME")
        conv1_3 = self.conv_2d(conv1_1, "conv1_2", [3, 3, 64, 128], [1, 1, 1, 1], "SAME")
        pool1 = tf.nn.max_pool(conv1_3, ksize =[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = "pool1")
        
        conv2_1 = self.conv_2d(pool1, "conv1_1", [3, 3, 128, 256], [1, 1, 1, 1], "SAME")
        conv2_2 = self.conv_2d(conv2_1, "conv1_2", [3, 3, 256, 256], [1, 1, 1, 1], "SAME")
        pool2 = tf.nn.max_pool(conv2_2, ksize =[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = "pool2")
        
        conv3_1 = self.conv_2d(pool2, "conv1_1", [3, 3, 256, 512], [1, 1, 1, 1], "SAME")
        conv3_2 = self.conv_2d(conv3_1, "conv1_2", [3, 3, 512, 512], [1, 1, 1, 1], "SAME")
        pool3 = tf.nn.max_pool(conv3_2, ksize =[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name = "pool3")
        
        shape = tf.get_shape(pool3)
        h_pool3_flat = tf.reshape(self.convlayers(imgs), [pool[0], pool[1]*pool[2]*pool[3]])
        fc1 = self.fc_layer(h_pool3_flat, "fc1", [pool[1]*pool[2]*pool[3], 1024], iftrain)
        fc2 = self.fc_layer(tf.nn.relu(fc1), "fc2", [1024, 1024], iftrain)
        fc3 = self.fc_layer(tf.nn.relu(fc2), "fc2", [1024, 361], iftrain = False)
        return fc3

if __name__== '__main__':
    sess = tf.Session()
    batch_size = 100
    num_epoch = 200
    train_data, train_labels, test_data,test_labels = p.load_data('train.csv', delimiter = ',')
    board = tf.placeholder(tf.float32, [batch_size, 19, 19, 1])
    target = tf.placeholder(tf.float32, [batch_size, 19*19])
    board_test = tf.placeholder(tf.float32, [1, 19, 19, 1])
    target_test = tf.placeholder(tf.float32, [1, 19*19])
    with tf.variable_scope("model"):
        gocnn = GOCNN(board, target, sess = sess)
    with tf.variable_scope("model", reuse = True):
        gocnn_test = GOCNN(board_test, target_test, sess = sess, iftrain = False)

    txt = []
    for i, name in enumerate(test_data):
        txt.append(name+";"+str(test_labels[i]) +"\n")
    with open('valid.csv', 'w') as f:
        f.writelines(txt)
    train_label_data = dict(zip(train_data,train_labels))
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
        label = []
        for n,x in enumerate(train_data):
            label.append(train_label_data[train_data[n]])
        start = time.time()
        accuracy = 0
        for step, (x, y) in enumerate(p.batch_iter(train_data, label, batch_size, True)):
            if i < 30:
                _, loss, lr, acc, summary = sess.run([gocnn.train_step_ad, gocnn.loss, gocnn.learning_rate, gocnn.accuracy,gocnn.merged ], 
                                                     feed_dict = {gocnn.board:x, gocnn.target:y})
            else:
                _, loss, lr, acc, summary = sess.run([gocnn.train_step_gd, gocnn.loss, gocnn.learning_rate, gocnn.accuracy,gocnn.merged], 
                                                     feed_dict = {gocnn.board:x, gocnn.target:y})
            if step%100 ==0:
                gocnn.writer.add_summary(summary, i*10000/batch_size + step)
            accuracy+=acc
            if step%int(math.ceil(len(train_data)/batch_size/100.0))==0:
                print "step:{}\taccuracy:{:3.3f}\tloss:{:3.5f}\t".format(step,accuracy/(step+1), loss)

        accuracy = 0
        print "Validation : --------------------------------------------"
        if (i+1)%10==0 or i==0:
            for step, (x, y) in enumerate(p.batch_iter(test_data, test_labels, 1, False)):
                acc = sess.run(gocnn_test.accuracy, 
                    feed_dict = {gocnn_test.imgs:x, gocnn_test.y:y})
                accuracy+=acc
                if step%int(math.ceil(len(train_data)/batch_size/100.0))==0:
                    print "step:{}\taccuracy:{:3.3f}".format(step, accuracy/(step+1))
            path = saver.save(sess, checkpoint_prefix, global_step=i)
            print("Saved model checkpoint to {}\n".format(path))
