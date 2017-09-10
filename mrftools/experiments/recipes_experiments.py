import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../util')
import argparse
from evaluate_methods import evaluate_methods
from generate_synthetic_data import generate_random_synthetic_data
import numpy as np

alphas = {1:([1, 0.5, 0.0],'o'), .75:([0.35, 1, 0.85],'v'), .5:([0.2, 0.6, .7],'<'), .25:([.9, .3, .7],'>'), 0:([.5, .7, .2],'s') }


def main():

	experiments_name = 'recipes'
	
	print('================================= ///////////////////START//////////////// =========================================')
	parser = argparse.ArgumentParser()
	parser.add_argument('--group_l1', dest='group_l1', required=True)
	parser.add_argument('--l2', dest='l2', default=0.5)
	parser.add_argument('--l1', dest='l1', default=0)
	parser.add_argument('--edge_num', dest='edge_num', default=100)
	parser.add_argument('--results_dir', dest='results_dir', default=5)

	shelve_dir = 'shelves'
	args = parser.parse_args()
	edge_reg = float(args.group_l1)
	node_reg = float(args.group_l1)

	################################################################### DATA PREPROCESSING GOES HERE --------->
	print('FETCHING DATA!')
	from read_ingredients import read_ingredients
	data, num_states, max_num_states, variables = read_ingredients()
	edges = []
	len_data = len(data)
	num_nodes = len(variables)
	print('variables')
	print(num_nodes)
	print('length data')
	print(len_data)
	shuffle(data)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]
	#############################################################################################<-------------

	max_update_step =  int(np.sqrt(len(variables)))
	evaluate_methods(num_states, variables, len(variables), max_num_states, data, len_data, args.results_dir, shelve_dir, args, \
		alphas, max_update_step, edge_reg, node_reg, experiments_name, int(args.edge_num), experiments_type='real')
	print('DONE!')

if __name__ == '__main__':
	main()