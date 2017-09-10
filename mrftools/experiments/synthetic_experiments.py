import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../util')
import argparse
from evaluate_methods import evaluate_methods
from generate_synthetic_data import generate_random_synthetic_data
import numpy as np

alphas = {1:([1, 0.5, 0.0],'o'), .75:([0.35, 1, 0.85],'v'), .5:([0.2, 0.6, .7],'<'), .25:([.9, .3, .7],'>'), 0:([.5, .7, .2],'s') }


def main():

	experiments_name = 'synthetic'

	print('================================= ///////////////////START//////////////// =========================================')
	parser = argparse.ArgumentParser()
	parser.add_argument('--nodes_num', dest='num_nodes', required=True)
	parser.add_argument('--state_num', dest='state_num', default=5)
	parser.add_argument('--len_data', dest='len_data', default=2000)
	parser.add_argument('--group_l1', dest='group_l1', required=True)
	parser.add_argument('--l2', dest='l2', default=0.5)
	parser.add_argument('--l1', dest='l1', default=0)
	parser.add_argument('--results_dir', dest='results_dir', default=5)

	shelve_dir = 'shelves'
	args = parser.parse_args()

	################################################################### DATA PREPROCESSING GOES HERE --------->
	num_nodes = int(args.num_nodes)
	edge_std = 1
	node_std = .5
	state_num = int(args.state_num)
	mrf_density = float(2)/((num_nodes - 1))
	len_data = int(args.len_data)
	edge_reg = float(args.group_l1)
	node_reg = float(args.group_l1)

	edge_num = int(3 * num_nodes) # MAX NUM EDGES TO GRAFT

	print('Genrating DATA!')
	model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, \
		state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
	

	#############################################################################################<-------------

	max_update_step =  int(np.sqrt(len(variables)))
	evaluate_methods(num_states, variables, len(variables), max_num_states, data, len_data, args.results_dir, shelve_dir, args, \
		alphas, max_update_step, edge_reg, node_reg, experiments_name, edge_num, edges=edges, experiments_type='synthetic')
	print('DONE!')

if __name__ == '__main__':
	main()