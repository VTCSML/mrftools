import time
import matplotlib.pyplot as plt
import numpy as np
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy
import time
from Graft import Graft
from read_ratings import read_ratings_from_batch_files

def main():
	# Define the l1 params intervals
	edge_reg_range = [0.06] #np.arange(0.01,0.25,0.05) 
	node_reg_range = [0.21] #np.arange(0.01,0.25,0.05)

	graft_iter = 2500 # Iterations per optimization over active set
	edge_num = 1 # Allowed max number of edges in the final network
	data_train_ratio = .8

	############################# LOAD DATA HERE ---->
	# Required data: 
	# data: a list of dicts [{variable1:state@x1, variable1:state@x1, ,...}, {variable1:state@x2, variable1:state@x2, ,...},...]
	# variables:  list of variables
	# num_states: dict() indicating the number of states of each variable
	# max_num_states : maximum number of states across all variables

	print('FETCHING DATA!')
	FILES_LIST = ['../../../ratings/jester-data-1.xls', '../../../ratings/jester-data-2.xls', '../../../ratings/jester-data-3.xls'] 
	data = read_ratings_from_batch_files(FILES_LIST, 50)
	len_data = len(data)
	train_data = data[: int(.8 * len_data)]
	test_data = data[int(.8 * len_data) : len_data]
	variables = list(data[0].keys())
	num_states = dict()
	for i in range(1,101):
		num_states[i] = 5
	max_num_states = 5

	############################

	len_data = len(data)
	train_data = data[: int(data_train_ratio * len_data)]
	test_data = data[int(data_train_ratio * len_data) : len_data]

	############################ LEARN USING SPG
	method = 'structured' #choose method as either 'structured', 'naive' or 'queue'
	best_spg_nll = float('inf')
	print('Searching for opt params')
	for edge_reg in edge_reg_range:
		for node_reg in node_reg_range:
			skip_iter = False
			print('PARAMS')
			print(edge_reg)
			print(node_reg)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, None, method) # Initialize SPG class
			spg.on_monitor_mn() # enable getting snapshots of partial MRFS and their graphs call spg.mn_snapshots and spg.graph_snapshots (Both are dict() with keys as time stamps)
			spg.on_verbose() # UnComment to log progress
			spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg) # Setup learning parameters
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, _ = spg.learn_structure(edge_num) # Learn structure
			exec_time = time.time() - t
			nll = compute_likelihood(learned_mn, len(variables), test_data, variables = variables) 
			if best_spg_nll>nll:
				print('NEW OPT FOUND AT:')
				print(str(node_reg) + ' , ' + str(edge_reg))
				best_spg_nll = nll
				opt_edge_reg = edge_reg
				opt_node_reg = node_reg
			print('======SPG NLL: ' + str(best_spg_nll) )
			print('======SPG Exec Time: ' + str(exec_time) + ' s')

	############################ LEARN USING CLASSIC GRAFT
	grafter = Graft(variables, num_states, max_num_states, train_data, None)
	grafter.on_monitor_mn()
	# grafter.on_verbose() #UnComment to log progress
	grafter.setup_learning_parameters(edge_l1 = opt_edge_reg, max_iter_graft=graft_iter, node_l1=opt_node_reg)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, _ = grafter.learn_structure(edge_num)
	exec_time = time.time() - t
	graft_nll = compute_likelihood(learned_mn, len(variables), test_data, variables = variables) 
	print('======GRAFT NLL: ' + str(graft_nll))
	print('======GRAFT Exec Time: ' + str(exec_time) + ' s')


if __name__ == '__main__':
	main()