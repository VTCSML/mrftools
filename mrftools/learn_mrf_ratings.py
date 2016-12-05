import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from Graft import Graft
from read_ratings import read_ratings_from_batch_files

METHOD_COLORS = {'structured':'-r', 'naive': '-b', 'queue':'-g', 'graft':'-y'}
FILES_LIST = ['../../../ratings/jester-data-1.xls', '../../../ratings/jester-data-2.xls', '../../../ratings/jester-data-3.xls'] 

def main():
	edge_reg = 5
	node_reg = 0
	graft_iter = 300
	# edge_num = float('inf')
	edge_num = 50
	num_states = dict()
	variables = range(1,101)
	num_attributes = len(variables)
	for i in range(1,101):
		num_states[i] = 5
	max_num_states = 5
	print('FETCHING DATA')
	data = read_ratings_from_batch_files(FILES_LIST, 50)
	len_data = len(data)
	train_data = data[: int(.7 * len_data)]
	test_data = data[int(.7 * len_data) : len_data]
	print('LEARNING MRF')
	spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, None, 'structured')
	spg.on_verbose()
	spg.setup_learning_parameters(edge_reg, max_iter_graft=graft_iter, node_l1=node_reg, zero_threshold=.01)
	learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num)
	print('TESTING')
	likelihood = compute_likelihood(learned_mn, variables, test_data)
	print(likelihood)

if __name__ == '__main__':
	main()