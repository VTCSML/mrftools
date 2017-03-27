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
from grafting_util import initialize_priority_queue
import copy


# METHODS = ['structured', 'naive', 'queue']
METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'black', 'graft':'blue'}
METHODS = ['structured', 'queue']
def main():
	edge_reg = 0.009 #np.arange(0.01,0.25,0.05) 
	node_reg = 0.01
	len_data = 100
	priority_graft_iter = 5000
	suffstats_ratio = .05
	training_ratio = .6
	num_nodes = 25
	state_num = 5
	T_likelihoods = dict()
	edge_std = 5
	node_std = .0001

	mrf_density = .01
	print('================================= ///////////////////START//////////////// ========================================= ')
	print('======================================Simulating data...')

	precisions_nofreeze, recall_nofreeze, time_nofreeze, precisions_freeze, recall_freeze, time_freeze = list(), list(), list(), list(), list(), list()
	for i in range(1):

		model, variables, data, max_num_states, num_states, real_edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
		train_data = data[: int(training_ratio * len_data)]
		test_data = data[int(training_ratio * len_data) : len_data]
		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		original_pq = initialize_priority_queue(variables=variables)
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

		M_time_stamps = dict()
		objs = dict()
		f1_scores = dict()




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
			pq = copy.deepcopy(original_pq)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict = pq)
			spg.on_show_metrics()
			spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
			spg.on_monitor_mn()
			spg.on_verbose()
			spg.on_plot_queue('../../../pq_plot')
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
			exec_time = time.time() - t
			print('---->Exec time')
			print(exec_time)
			print('Loss')
			print(objec)
			M_time_stamps[method] = sorted(list(spg.mn_snapshots.keys()))
			objs[method] = objec
			f1_scores[method] = f1_score


		plt.close()
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		for i in range(len(METHODS)):
			print(METHODS[i])
			ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='Loss-' + METHODS[i], linewidth=1)
			ax2.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label='F1-'+METHODS[i])
		ax1.set_xlabel('Time')
		ax1.set_ylabel('Normalized Loss')
		ax2.set_ylabel('F1 Score')
		ax2.legend(loc=4, fancybox=True, framealpha=0.5)
		ax1.legend(loc='best', framealpha=0.5)
		plt.title('Loss-F1')
		plt.savefig('../../../pq_plot/' + str(len(variables)) + '_loss_.png')
		plt.close()




		# time_stamps = sorted(list(spg.mn_snapshots.keys()))
		# M_time_stamps[method] = time_stamps
		# f1_scores[method] = f1_score


	# print('Missed edges')
	# print([x for x in active_sets['graft'] if x in real_edges and x not in active_sets['structured']])
if __name__ == '__main__':
	main()