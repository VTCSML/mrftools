import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from Graft import Graft


# METHODS = ['structured', 'naive', 'queue']
METHODS = ['structured']
def main():
	edge_reg = 0.05 #np.arange(0.01,0.25,0.05) 
	node_reg = 0.075
	len_data = 1000
	priority_graft_iter = 2500
	graft_iter = 2500
	suffstats_ratio = .05
	training_ratio = .6
	num_nodes = 10
	T_likelihoods = dict()
	print('================================= ///////////////////START//////////////// ========================================= ')
	print('======================================Simulating data...')
	model, variables, data, max_num_states, num_states, real_edges = generate_random_synthetic_data(len_data, num_nodes, 8, .05)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]
	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)
	print(variables)
	print(num_states)
	print(max_num_states)
	print('NUM VARIABLES')
	print(len(variables))
	print('NUM EDGES')
	print(len(real_edges))
	print('EDGES')
	print(real_edges)
	edge_num = float('inf')
	num_attributes = len(variables)
	recalls, precisions, active_sets= dict(), dict(), dict()

	# print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	# grafter = Graft(variables, num_states, max_num_states, data, list_order)
	# grafter.on_show_metrics()
	# # grafter.on_limit_sufficient_stats(suffstats_ratio)
	# # grafter.on_verbose()
	# grafter.setup_learning_parameters(edge_reg, max_iter_graft=graft_iter)
	# grafter.on_monitor_mn()
	# grafter.on_zero_treshold(zero_threshold=zero_threshold)
	# t = time.time()
	# learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure(edge_num, real_edges)
	# exec_time = time.time() - t
	# print('exec_time')
	# print(exec_time)
	# active_sets['graft'] = final_active_set

	################################### REMOVE THIS
	edges = real_edges
	# edges = final_active_set

	for method in METHODS:
		print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)


		spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
		spg.on_show_metrics()
		spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
		spg.on_monitor_mn()
		spg.on_verbose()
		t = time.time()
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
		exec_time = time.time() - t
		# precisions[method] = precision
		# recalls[method] = recall
		# time_stamps = sorted(list(spg.mn_snapshots.keys()))
		# M_time_stamps[method] = time_stamps
		# f1_scores[method] = f1_score


	# print('Missed edges')
	# print([x for x in active_sets['graft'] if x in real_edges and x not in active_sets['structured']])
if __name__ == '__main__':
	main()