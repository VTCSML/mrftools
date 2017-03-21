import time
import matplotlib.pyplot as plt
import numpy as np
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy
import time
from Graft import Graft
from read_ratings import read_ratings_from_batch_files

from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data

folder_name = '../../../real_results'
METHODS = ['structured', 'queue']
METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'black', 'graft':'blue'}

def main():
	# Define the l1 params intervals
	edge_reg_range = [0.01] #np.arange(0.01,0.25,0.05) 
	node_reg_range = [0.05] #np.arange(0.01,0.25,0.05)

	graft_iter = 2500 # Iterations per optimization over active set
	edge_num = 1 # Allowed max number of edges in the final network
	data_train_ratio = .8

	max_sufficient_stats_ratio = .1


	opt_edge_reg = edge_reg_range[0]
	opt_node_reg = node_reg_range[0]

	############################# LOAD DATA HERE ---->
	# Required data: 
	# data: a list of dicts [{variable1:state@x1, variable2:state@x1, ,...}, {variable1:state@x2, variable2:state@x2, ,...},...]
	# variables:  list of variables
	# num_states: dict() indicating the number of states of each variable
	# max_num_states : maximum number of states across all variables

	model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(100, 10, mrf_density=.05, state_min=5, state_max=5, edge_std=.5, node_std = .01)


	# print('FETCHING DATA!')
	# FILES_LIST = ['../../../ratings/jester-data-1.xls', '../../../ratings/jester-data-2.xls', '../../../ratings/jester-data-3.xls'] 
	# data = read_ratings_from_batch_files(FILES_LIST, 50)
	# len_data = len(data)
	# train_data = data[: int(.8 * len_data)]
	# test_data = data[int(.8 * len_data) : len_data]
	# variables = list(data[0].keys())
	# num_states = dict()
	# for i in range(1,101):
	# 	num_states[i] = 5
	# max_num_states = 5

	# data = [{0:1, 1:2, 2:3}, {0:1, 1:2, 2:3}, {0:1, 1:2, 2:3}, {0:1, 1:2, 2:3}]
	# max_num_states = 4
	# variables = [0,1,2]
	# num_states = {0:2, 1:3, 2:4}

	############################

	len_data = len(data)
	train_data = data[: int(data_train_ratio * len_data)]
	test_data = data[int(data_train_ratio * len_data) : len_data]

	ts_likelihoods, method_time_stamps, ts_loss = dict(), dict(), dict()

	############################ LEARN USING SPG
	for method in  METHODS: #choose method as either 'structured', 'naive' or 'queue'
		print(method)
		if method != 'structured':
			edge_reg = [opt_edge_reg]
			node_reg = [opt_node_reg]
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
				spg.on_limit_sufficient_stats(max_sufficient_stats_ratio)
				spg.on_show_metrics()
				print('learning')
				spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg) # Setup learning parameters
				t = time.time()
				learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, loss, _ = spg.learn_structure(edge_num) # Learn structure
				exec_time = time.time() - t
				nll = compute_likelihood(learned_mn, len(variables), test_data, variables = variables) 
				if best_spg_nll>nll:
					print('NEW OPT FOUND AT:')
					print(str(node_reg) + ' , ' + str(edge_reg))
					best_spg_nll = nll
					opt_edge_reg = edge_reg
					opt_node_reg = node_reg
					best_mn_snapshots = spg.mn_snapshots
					best_loss = loss

					print(best_loss)
				# print('======SPG NLL: ' + str(best_spg_nll))
				# print('======SPG Exec Time: ' + str(exec_time) + ' s')
		time_stamps = sorted(list(best_mn_snapshots.keys()))
		method_time_stamps[method] = time_stamps
		method_likelihoods = list()
		for t in time_stamps:
			nll = compute_likelihood(best_mn_snapshots[t], len(variables), test_data, variables = variables)
			method_likelihoods.append(nll)
		ts_likelihoods[method] = method_likelihoods
		ts_loss[method] = best_loss

	############################ LEARN USING CLASSIC GRAFT
	METHODS.append('graft')
	grafter = Graft(variables, num_states, max_num_states, train_data, None)
	grafter.on_limit_sufficient_stats(max_sufficient_stats_ratio)
	grafter.on_monitor_mn()
	grafter.on_show_metrics()
	# grafter.on_verbose() #UnComment to log progress
	grafter.setup_learning_parameters(edge_l1 = opt_edge_reg, max_iter_graft=graft_iter, node_l1=opt_node_reg)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, loss, _ = grafter.learn_structure(edge_num)
	exec_time = time.time() - t
	graft_nll = compute_likelihood(learned_mn, len(variables), test_data, variables = variables) 
	# print('======GRAFT NLL: ' + str(graft_nll))
	# print('======GRAFT Exec Time: ' + str(exec_time) + ' s')
	time_stamps = sorted(list(grafter.mn_snapshots.keys()))

	method_time_stamps['graft'] = time_stamps
	method_likelihoods = list()
	for t in time_stamps:
		nll = compute_likelihood(grafter.mn_snapshots[t], len(variables), test_data, variables = variables)
		method_likelihoods.append(nll)
	ts_likelihoods['graft'] = method_likelihoods
	ts_loss['graft'] = loss


	##### PLOTTING RESULTS
	### LOSS VS Time
	plt.close()
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	for i in range(len(METHODS)):
		ax1.plot(method_time_stamps[METHODS[i]], ts_loss[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='Loss-' + METHODS[i], linewidth=1)
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Loss')
	ax1.legend(loc='best', framealpha=0.5)
	plt.title('Loss VS Time')
	plt.savefig(folder_name + '/' + str(len(variables)) + '_Loss_.png')
	plt.close()

	### NLL VS Time
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		ax1.plot(method_time_stamps[METHODS[i]], ts_likelihoods[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='NLL-' + METHODS[i], linewidth=1)
	ax1.set_xlabel('Time')
	ax1.set_ylabel('NLL')
	ax1.legend(loc='best', framealpha=0.5)
	plt.title('NLL')
	plt.savefig(folder_name + '/' + str(len(variables)) + '_NLL_.png')
	plt.close()

if __name__ == '__main__':
	main()