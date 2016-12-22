import numpy as np
import math
import random

def read_file(filename):
    """
    Reads a datafile of .adi format and returns lists of black stone positions, white stone psitions, the player to play, 
    the move played and ko restrictions each for a particular state respectively

    Input : 
    filename : string location of the .adi file 

    Returns
    black : list of list of black positions for each example
    white : list of list of white positions for each example
    player : list of the player playing at the particular example
    move : list of moves played by the player at the given example
    ko : list of locations of ko restrictions for the give example
    """
    data = []
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    output = [[], [], [], [], [], []]
    for i, a in enumerate(data):
        output[i%6].append(a)
    output = output[1:]
    black = [[int(a) for a in x[2:-1].split(';') if a != ''] for x in output[0]]
    white = [[int(a) for a in x[2:-1].split(';') if a != ''] for x in output[1]]
    player = [2 if x[2:-1] == 'B' else 1 for x in output[2]]
    move = [int(x[2:-1]) for x in output[3]]
    ko = [[int(a) for a in x[2:-1].split(';') if a != ''] for x in output[4]]
    return (black, white, player, move, ko)

def load_data(filename, split = 0.9):
    """
    Reads the given .adi file and returns data split into train and test set where each element is a 5-tuple of the following format
    (black stones, white_stones, player, move, ko_restrictions)

    Input:
    filename : string the .adi dataset file
    split : float ratio of train data

    Returns:
    train : train split of the dataset
    test : Test split of the dataset
    """
    (black, white, player, move, ko) = read_file(filename)
    dataset = zip(black, white, player, move, ko)
    train_data = dataset[:int(math.ceil(split*len(dataset)))]
    test_data = dataset[int(math.ceil(split*len(dataset))):]
    return train_data, test_data

def get_board(pos, num):
    """
    returns a 19x19 board with pos positions filled with 'num'

    Input:
    pos : list of positions to be filled
    num : int value to be filled in said positions as int

    Returns:
    board : np.array() type int32 with pos filled with num
    """
    board = np.zeros((361), np.int32)
    if pos == -1:
        return board.reshape(19, 19, 1)
    for p in pos:
        board[p] = num
    return board.reshape(19, 19, 1)

def get_target(pl, pos):
    """
    Generate target matrix given player and position of move

    Input:
    pos : postion of move as int

    Returns:
    tar : (361) np.array() of type int32 with 1 at pos 
    """
    tar = np.zeros(361, dtype = np.int32)
    tar[pos] = 1
    return tar

def process_mini_batch(data):
    """
    Processes a minibatch of data to give workable data for the network

    Input:
    data : list of data points in the mini batch

    Returns:
    board : list of (19,19,3) np.array() representing the state of the GO board for every point in data
    target : the target move corresponding to the each board state
    ko : ko restrictions for each board state 
    """
    board = []
    target = []
    ko = []
    for p in data:
        black = p[0] if len(p[0]) != 0 else -1
        white = p[1] if len(p[1]) != 0 else -1
        mat1, mat2 = (black, white) if p[2] == 1 else (white, black)
        board.append(np.dstack((get_board(mat1, 1), get_board(mat2, 1), get_board(p[4], 1))))
        target.append(get_target(p[2], p[3]))
        ko.append(get_board(p[4], 1))
    return board, target, ko

def batch_iter(data, batch_size):
    """
    An iterator that generates mini batches of size batch_size from the given data set with shuffling

    Input :
    data : list of tauples containing the information of each exmaple extracted from the .adi file
    batch_size : int length of minibatch size

    Yeilds:
    board : list of (19,19) np.array() representing the state of the GO board for every point in data
    target : the target move corresponding to the each board state
    ko : ko restrictions for each board state
    """
    random.shuffle(data)
    num_data = len(data)
    num_steps = int(math.ceil(num_data/float(batch_size)))
    rem = num_data%batch_size
    for step in range(num_steps):
        if step == (num_steps-1) and rem !=0:
            data_batch = data[(step)*batch_size:]
            random.shuffle(data)
            data_batch.extend(data[:(batch_size - rem)])
            board, target, ko = process_mini_batch(data_batch)
            yield (board, target, ko)
        else:
            data_batch = data[step*batch_size:(step+1)*batch_size]
            board, target, ko = process_mini_batch(data_batch)
            yield (board, target, ko)