import time
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from SelectiveStructuredPriorityGraft1 import SelectiveStructuredPriorityGraft1
from SelectiveStructuredPriorityGraft import SelectiveStructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from Graft import Graft
from grafting_util import initialize_priority_queue
import copy


# METHODS = ['structured', 'naive', 'queue']
METHOD_COLORS = {'queue':'red', 'best_k': 'blue', 'struct_best_k':'black', 'StrSSPG':'green'}
# METHODS = ['structured', 'queue']
METHODS = []
def main():
	len_data = 1000
	priority_graft_iter = 2500
	suffstats_ratio = .05
	training_ratio = .9
	num_nodes = 50
	state_num = 5
	T_likelihoods = dict()
	edge_std = 1
	node_std = 1

	edge_reg = 0.01 #np.arange(0.01,0.25,0.05) 
	node_reg = edge_reg
	l2 = .5
	l1 = 0

	METHOD_legend = dict()

	mrf_density = float(3)/((num_nodes - 1))
	print('================================= ///////////////////START//////////////// ========================================= ')
	print('======================================Simulating data...')

	precisions_nofreeze, recall_nofreeze, time_nofreeze, precisions_freeze, recall_freeze, time_freeze = list(), list(), list(), list(), list(), list()

	model, variables, data, max_num_states, num_states, real_edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std, mean_degree=2)
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
	num_attributes = len(variables)
	recalls, precisions, active_sets= dict(), dict(), dict()
	M_time_stamps = dict()
	objs = dict()
	f1_scores , mn_snapshots= dict(), dict()
	edges = real_edges
	METHODS = list()


	edge_num = int(3 * num_nodes) #MAX NUM EDGES TO GRAFT



	k = len(variables)
	alpha = 0.75
	max_update_step = int(np.sqrt(len(variables)))
	meth = str(alpha) + ' SOEG '
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	sspg.set_reassigned_nodes( m=5)
	sspg.on_structured()
	# sspg.set_select_unit(select_unit=select_unit)
	sspg.set_alpha(alpha=alpha)
	sspg.set_max_update_step(max_update_step=max_update_step)
	sspg.on_monitor_mn()
	# sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('len active set')
	print(len(final_active_set))
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = 'red'
	METHOD_legend[meth] = meth


	k = len(variables)
	alpha = 0.75
	max_update_step =  int(np.sqrt(len(variables)))
	meth = str(alpha) + ' OEG '
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	# sspg.on_structured()
	# sspg.set_select_unit(select_unit=select_unit)
	sspg.set_alpha(alpha=alpha)
	sspg.set_max_update_step(max_update_step=max_update_step)
	sspg.on_monitor_mn()
	# sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('len active set')
	print(len(final_active_set))
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = 'green'
	METHOD_legend[meth] = meth




	# k = len(variables)/4
	# alpha = 1
	# max_update_step = len(variables)
	# meth = str(alpha) + '_best_' + str(k)
	# print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	# pq = copy.deepcopy(original_pq)
	# sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	# sspg.on_show_metrics()
	# sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l2_coeff=l2)
	# sspg.set_top_relvant(k=k)
	# sspg.on_structured()
	# # sspg.set_select_unit(select_unit=select_unit)
	# sspg.set_alpha(alpha=alpha)
	# sspg.set_max_update_step(max_update_step=max_update_step)
	# sspg.on_monitor_mn()
	# sspg.on_verbose()
	# t = time.time()
	# learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	# exec_time = time.time() - t
	# print('---->Exec time')
	# print(exec_time)
	# print('Loss')
	# print(objec)
	# time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	# M_time_stamps[meth] = time_stamps
	# objs[meth] = objec
	# f1_scores[meth] = f1_score
	# METHODS.append(meth)
	# recalls[meth] = recall
	# METHOD_COLORS[meth] = 'red'


	# k = int(float(edge_num) / 3)
	# meth = 'struct_best_' + str(k)
	# print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	# pq = copy.deepcopy(original_pq)
	# sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	# sspg.on_show_metrics()
	# sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=0, l2_coeff=l2)
	# sspg.set_top_relvant(k=k)
	# sspg.on_monitor_mn()
	# sspg.on_verbose()
	# sspg.on_structured()
	# t = time.time()
	# learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	# exec_time = time.time() - t
	# print('---->Exec time')
	# print(exec_time)
	# print('Loss')
	# print(objec)
	# time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	# M_time_stamps[meth] = time_stamps
	# objs[meth] = objec
	# f1_scores[meth] = f1_score
	# METHODS.append(meth)
	# recalls[meth] = recall
	# METHOD_COLORS[meth] = 'black'


	# k = int(float(edge_num) / 3)
	# meth = 'grow_best_' + str(k)
	# print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	# pq = copy.deepcopy(original_pq)
	# sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	# sspg.on_show_metrics()
	# sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=0, l2_coeff=l2)
	# sspg.set_top_relvant(k=k)
	# sspg.on_monitor_mn()
	# sspg.on_verbose()
	# t = time.time()
	# learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	# exec_time = time.time() - t
	# print('---->Exec time')
	# print(exec_time)
	# print('Loss')
	# print(objec)
	# time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	# M_time_stamps[meth] = time_stamps
	# objs[meth] = objec
	# f1_scores[meth] = f1_score
	# METHODS.append(meth)
	# recalls[meth] = recall
	# METHOD_COLORS[meth] = 'black'

	var = 'jndn'
	plt.close()
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=r'$\alpha = $'+ METHOD_legend[METHODS[i]], linewidth=1)
		ax2.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Recall')
	ax2.legend(fancybox=True, framealpha=0.5, bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
	ax1.legend(loc='best', framealpha=0.5)
	plt.title('Loss-Recall')
	plt.savefig('../../../pq_plot/' + str(len(variables)) + '_Recall-Loss.eps', format='eps', dpi=1000)
	plt.close()


if __name__ == '__main__':
	main()