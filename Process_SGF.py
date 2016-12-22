import math
import string
import numpy as np
import os
import random

def get_stones(board):
	"""
	Returns locations of black and white stones in the board

	Input :
	board : a list of length 361

	Returns:
	black : list of locations of black stones as an np array
	white : list of lcoations of white stones as an np array
	"""
	a = np.array(board)
	black = np.where(a == 2)[0]
	white = np.where(a == 1)[0]
	return black, white

def process_data_folder(path):
	"""
	Returns the locations of all the sgf files in a folder and all its subfolders

	Input:
	path : relative or absolute path to the folder which is to be searched

	Returns:
	sgf_files : list of paths to all sgf files
	"""
	sgf_files = []
	for dirpath, _, filenames in os.walk(path):
		for files in [f for f in filenames if f.endswith('.sgf')]:
			sgf_files.append(os.path.join(dirpath, files))
	return sgf_files
	
def process_board(board, player):
	"""
	Processes the board to identify groups of stones that have lost all their liberties.
	Future work merge into a more elegent fucntion and return a bard with 1s where a stone may not be placed.

	Input:
	board : a 361 sized list containing the position of the stones with 1 representing white, 2 black and 0 no stone
	player : the player who played the last move

	Returns:
	safe_black : stones that have atleast 1 liberty or are connected to a similar stone with atleast 1 liberty
	dead_black : stones with no liberties or connected to similar stones with no liberty
	"""
	p_opp = 1 if player == 2 else 2
	def check_alive(pos):
		pos = np.array(pos)
		neigh = np.clip([pos+[0, 1], pos+[1, 0], pos + [-1, 0], pos + [0, -1]], 0, 18)
		neigh_v_v = [(tuple(x), a[tuple(x)]) for x in neigh if not (tuple(x) in group)]
		neigh_v = [x[1] for x in neigh_v_v]
		if 0 in neigh_v:
			return "safe"
		elif p_opp in neigh_v:
			for x in neigh_v_v:
				if x[1] == p_opp and not(x[0] in group):
					if x[0] in safe:
						return "safe"
					group.append(x[0])
					res = check_alive(x[0])
					if res== "safe":
						return "safe"
					elif res=="dead":
						continue
		else:
			return "dead"

	a = np.array(board)
	a = a.reshape(19,19)
	stones = np.where(a == p_opp)
	stones = list(zip(stones[0], stones[1]))
	stones = np.array(stones)
	safe = []
	dead = []

	for pi in stones:
		if not(tuple(pi) in safe) and not(tuple(pi) in dead):
			group = [tuple(pi)]
			neighbours = np.clip([pi+[0, 1], pi+[1, 0], pi + [-1, 0], pi + [0, -1]], 0, 18)

			neighbours_v_v = [(tuple(x), a[tuple(x)]) for x in neighbours if tuple(x) != tuple(pi)]

			neighbours_v = [x[1] for x in neighbours_v_v]
			flag = 0
			if 0 in neighbours_v:
			    safe.extend(group)
			elif p_opp in neighbours_v:
				for x in neighbours_v_v:
					if x[1] == p_opp and not(x[0] in group) and flag == 0:
						if x[0] in safe:
							safe.extend(group)
							flag = 1
							break
						group.append(x[0])
						res = check_alive(x[0])
						if res == "safe":
							safe.extend(group)
							group = []
							flag = 1
							break
						elif res=="dead":
							continue
				if flag != 1:
					dead.extend(group)
					group = []
			else:
				dead.extend(group)
				group = []
	return safe, dead

def process_ko(dead, x, board, prev_board):
	"""
	Processes the board to identify a ko

	Input:
	dead : list of positions of stones that died on the previous turn
	x : int identity of the player that just played
	board : the state of the board after player x has played
	prev_board : state of the baord before player x has played

	Returns:
	ko : a list of lenght 361 with a 1 where a stone cannot be placed in the next move and 0 elsewhere
	"""
	ko = [0]*361
	if len(dead) == 1:
		x_opp = 2 if x==1 else 1
		board[dead[0][0]*19 + dead[0][1]] = x_opp
		safe, dead = process_board(board, x_opp)
		if len(dead) == 1:
			for x in dead:
				board[19*x[0] + x[1]] = 0
			if np.sum(np.array(board) == np.array(prev_board)) == len(board):
				ko[dead[0][0]*19 + dead[0][1]] = 1
	return ko

def process_file(filename):
	"""
	An Iterator that reads an sgf file for the moves and yeilds the the entire board
	move by move and the state of the board before said move.

	Input:
	filename : string location to the sgf file as a string

	Yields:
	prev_board : state of the board before a particular move
	x : int player who played the move
	y : int location of the move
	prev_ko : a list of len 361 with a 1 where a stone cannot be placed for this move and 0 elsewhere
	"""
	alphabets = string.ascii_lowercase[0:19]
	alph_num_dict = dict(zip(list(alphabets), range(19))) #create a dictionary mapping the alpabet co-ordinate with numerical one
	txt = []
	with open(filename, 'r') as f:	#read sgf file
		txt = f.read().splitlines()


	moves = []
	for x in txt:
		if x == '':
			continue
		if x[0] == ';':
			moves.extend(x[1:-1].split(';'))

	def pos(move):
		return 1 if move[0]=="W" else 2, 19*alph_num_dict[move[2]] + alph_num_dict[move[3]]

	ko = [0]*361
	board = [0]*(19*19)
	prev_board = board
	for move in moves:
		if move[2:4] == 'tt' or move[1:] == '[]' or move == '':
			continue
		x, y = pos(move)		# calculate the position on the board the next piece is placed
		prev_board = board[:]
		prev_ko = ko[:]
		board[y] = x
		safe, dead = process_board(board, x) # process board to see if any pieces have lost all liberties
		for x in dead:
			board[19*x[0] + x[1]] = 0
		ko = process_ko(dead, x, board[:], prev_board[:])
		yield prev_board, x, y, prev_ko

def process_path(path, f):
	print "Processing file {}".format(path),
	t = 0
	for i, (board, player, move, ko) in enumerate(process_file(path)):
		t += 1
		text = "E[{}]\n".format(i)
		black, white = get_stones(board[:])
		text += "B"+"".join(np.array2string(black, max_line_width=181, separator=';').split())+"\n"
		text += "W"+"".join(np.array2string(white, max_line_width=181, separator=';').split())+"\n"
		text += "P["+("W]" if player == 1 else "B]")+"\n"
		text += "M[{}]\n".format(move)
		_, k = get_stones(ko[:])
		text += "K"+"".join(np.array2string(k, max_line_width=181, separator=';').split())+"\n"
		f.write(text)
	print "DONE"
	return t

def process_data_set(folder, output, num_files = -1):
	"""
	process files in the list of paths and prints them in the output file in the following format
	E[move number]
	B[list of positions for black stones ';'-seperated]
	W[list of positions for white stones ';'-seperated]
	P[Player who plays now]
	M[Location of move]
	K[locations for ko]
	
	Input :
	folder : string location to the sgf files
	output : string file where data is to be written

	output:
	file at output
	"""
	paths = process_data_folder(folder)
	random.shuffle(paths)
	if num_files == -1:
		num_files = len(paths)
	if num_files < len(paths):
		paths = paths[0:num_files]
	total_examples = 0
	with open(output, 'w') as f:
		for path in paths:
			total_examples += process_path(path, f)
	print "Total Examples : ", total_examples
	

if __name__ == '__main__':
	data_folder = './games/'
	output_file = './output.adi'
	print "Processing SGF files in {}".format(data_folder)
	process_data_set(data_folder, output_file, 1000)
	print "Folder Processed, data written in {}".format(output_file)
