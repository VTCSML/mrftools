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
METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'black', 'graft':'blue'}
METHODS = ['graft']
def main():
	edge_reg = 0.04 #np.arange(0.01,0.25,0.05) 
	node_reg = 0.05
	len_data = 1000
	priority_graft_iter = 2500
	graft_iter = 100
	suffstats_ratio = .05
	training_ratio = .6
	num_nodes = 10
	state_num = 5
	T_likelihoods = dict()
	edge_std = 5
	node_std = .0001

	mrf_density = float(1)/((num_nodes - 1))
	print('================================= ///////////////////START//////////////// ========================================= ')
	print('======================================Simulating data...')

	precisions_nofreeze, recall_nofreeze, time_nofreeze, precisions_freeze, recall_freeze, time_freeze = list(), list(), list(), list(), list(), list()
	for i in range(1):

		model, variables, data, max_num_states, num_states, real_edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
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
		M_time_stamps = dict()
		objs = dict()
		f1_scores = dict()

		edges = real_edges

		print('================GRAFT')
		grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
		grafter.on_show_metrics()
		# grafter.on_verbose()
		grafter.setup_learning_parameters(edge_l1 = edge_reg, max_iter_graft=graft_iter, node_l1=node_reg)
		grafter.on_monitor_mn()
		t = time.time()
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
		print(final_active_set)
		print(recall)
		objs['graft'] = objec
		f1_scores['graft'] = f1_score
		exec_time = time.time() - t
		precisions['graft'] = precision
		recalls['graft'] = recall
		# mn_snapshots['graft'] = grafter.mn_snapshots
		# time_stamps = sorted(list(grafter.mn_snapshots.keys()))
		# M_time_stamps['graft'] = time_stamps
		# method_likelihoods = []
		# print(len(time_stamps))
		print(objec)

		################################### REMOVE THIS
		# edges = final_active_set

		# plt.close()
		# fig, ax1 = plt.subplots()
		# ax2 = ax1.twinx()
		# for i in range(len(METHODS)):
		# 	print(METHODS[i])
		# 	ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='Loss-' + METHODS[i], linewidth=1)
		# 	ax2.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label='F1-'+METHODS[i])
		# ax1.set_xlabel('Time')
		# ax1.set_ylabel('Normalized Loss')
		# ax2.set_ylabel('F1 Score')
		# ax2.legend(loc=4, fancybox=True, framealpha=0.5)
		# ax1.legend(loc='best', framealpha=0.5)
		# plt.title('Loss-F1')
		# plt.savefig('../../../pq_plot/' + str(len(variables)) + '_loss_.png')
		# plt.close()




		# time_stamps = sorted(list(spg.mn_snapshots.keys()))
		# M_time_stamps[method] = time_stamps
		# f1_scores[method] = f1_score


	# print('Missed edges')
	# print([x for x in active_sets['graft'] if x in real_edges and x not in active_sets['structured']])
if __name__ == '__main__':
	main()