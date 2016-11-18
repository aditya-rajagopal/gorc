import math
import string
import numpy as np

def process_board(board):
	"""
	Processes the board to identify groups of stones that have lost all their liberties.
	Future work merge into a more elegent fucntion and return a bard with 1s where a stone may not be placed.

	Input:
	board : a 361 sized list containing the position of the stones with 1 representing white, 2 black and 0 no stone

	Output:
	safe_black : black stones that have atleast 1 liberty or are connected to a black stone with atleast 1 liberty
	dead_black : black stones with no liberties or connected to black stones with no liberty 
	safe_white : white stones that have atleast 1 liberty or are connected to a white stone with atleast 1 liberty
	dead_white : white stones with no liberties or connected to white stones with no liberty 
	"""
    def check_alive_black(pos):
        pos = np.array(pos)
        neigh = np.clip([pos+[0, 1], pos+[1, 0], pos + [-1, 0], pos + [0, -1]], 0, 18)
        neigh_v_v = [(tuple(x), a[tuple(x)]) for x in neigh if not (tuple(x) in group)]
        neigh_v = [x[1] for x in neigh_v_v]
        if 0 in neigh_v:
            return "safe"
        elif 2 in neigh_v:
            for x in neigh_v_v:
                if x[1] == 2 and not(x[0] in group):
                    if x[0] in safe_black:
                        return "safe"
                    group.append(x[0])
                    res = check_alive_black(x[0])
                    if res== "safe":
                        return "safe"
                    elif res=="dead":
                        continue
        else:
            return "dead"

    def check_alive_white(pos):
        pos = np.array(pos)
        neigh = np.clip([pos+[0, 1], pos+[1, 0], pos + [-1, 0], pos + [0, -1]], 0, 18)
        neigh_v_v = [(tuple(x), a[tuple(x)]) for x in neigh if not (tuple(x) in group)]
        neigh_v = [x[1] for x in neigh_v_v]
        if 0 in neigh_v:
            return "safe"
        elif 1 in neigh_v:
            for x in neigh_v_v:
                if x[1] == 1 and not(x[0] in group):
                    if x[0] in safe_white:
                        return "safe"
                    group.append(x[0])
                    res = check_alive_white(x[0])
                    if res== "safe":
                        return "safe"
                    elif res=="dead":
                        continue
        else:
            return "dead"
    
    a = np.array(board)
    a = a.reshape(19,19)
    black = np.where(a == 2)
    white = np.where(a == 1)
    white = list(zip(white[0], white[1]))
    black = list(zip(black[0], black[1]))
    black = np.array(black)
    white = np.array(white)
    
    safe_black = []
    safe_white = []
    dead_black = []
    dead_white = []

    for pi in black:
        if not(tuple(pi) in safe_black) and not(tuple(pi) in dead_black):
            group = [tuple(pi)]
            neighbours = np.clip([pi+[0, 1], pi+[1, 0], pi + [-1, 0], pi + [0, -1]], 0, 18)

            neighbours_v_v = [(tuple(x), a[tuple(x)]) for x in neighbours if tuple(x) != tuple(pi)]

            neighbours_v = [x[1] for x in neighbours_v_v]
            flag = 0
            if 0 in neighbours_v:
                safe_black.extend(group)
            elif 2 in neighbours_v:
                for x in neighbours_v_v:
                    if x[1] == 2 and not(x[0] in group) and flag == 0:
                        if x[0] in safe_black:
                            safe_black.extend(group)
                            flag = 1
                            break
                        group.append(x[0])
                        res = check_alive_black(x[0])
                        if res == "safe":
                            safe_black.extend(group)
                            group = []
                            flag = 1
                            break
                        elif res=="dead":
                            continue
                if flag != 1:
                    dead_black.extend(group)
                    group = []
            else:
                dead_black.extend(group)
                group = []
    for pi in white:
        if not(tuple(pi) in safe_white) and not(tuple(pi) in dead_white):
            group = [tuple(pi)]
            neighbours = np.clip([pi+[0, 1], pi+[1, 0], pi + [-1, 0], pi + [0, -1]], 0, 18)

            neighbours_v_v = [(tuple(x), a[tuple(x)]) for x in neighbours if tuple(x) != tuple(pi)]

            neighbours_v = [x[1] for x in neighbours_v_v]
            flag = 0
            if 0 in neighbours_v:
                safe_black.extend(group)
            elif 1 in neighbours_v:
                for x in neighbours_v_v:
                    if x[1] == 2 and not(x[0] in group) and flag == 0:
                        if x[0] in safe_white:
                            safe_white.extend(group)
                            flag = 1
                            break
                        group.append(x[0])
                        res = check_alive_white(x[0])
                        if res == "safe":
                            safe_white.extend(group)
                            group = []
                            flag = 1
                            break
                        elif res=="dead":
                            continue
                if flag != 1:
                    dead_white.extend(group)
                    group = []
            else:
                dead_white.extend(group)
                group = []
    return safe_black, dead_black, safe_white, dead_white


def process_file(filename):
	"""
	An Iterator that reads an sgf file for the moves and yeilds the the entire board
	move by move. Future work to also yeild positions where a player may not play and other features.
	temporary function.

	Parameters:
	filename : location to the sgf file as a string

	Yields:
	board_state : state of the board after applying piece elimination rules as a 19x19 np array with white pieces as 1 and black as 2
				  no pices are represented by 0
	"""
	alphabets = string.ascii_lowercase[0:19]
	alph_num_dict = dict(zip(list(alphabets), range(19))) #create a dictionary mapping the alpabet co-ordinate with numerical one
	txt = []
	with open(filename, 'r') as f:	#read sgf file
		txt = f.readlines()

	moves = []
	for x in txt:
		if x[0] == ';':
			moves.extend(x[1:-1].split(';'))

	def pos(move):
		return 1 if move[0]=="W" else 2, 19*dict1[move[2]] + dict1[move[3]]

	board = [0]*(19*19)
	for move in moves:
		x, y = pos(move)		# calculate the position on the board the next piece is placed
		board[y] = x
		safe_black, dead_black, safe_white, dead_white = process_board(board) # process board to see if any pieces have lost all liberties
		for x in dead_black:
			board[19*x[0] + x[1]] = 0
		for x in dead_white:
			board[19*x[0] + x[1]] = 0
		board_state = np.array(board).reshape(19, 19)
		yield board_state