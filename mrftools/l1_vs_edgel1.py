import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from SelectiveStructuredPriorityGraft import SelectiveStructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from Graft import Graft
from grafting_util import initialize_priority_queue
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', dest='alpha', default=.01)
parser.add_argument('--lambda', dest='_lambda', default=.5)
parser.add_argument('--data_len', dest='data_len', default=5000)
parser.add_argument('--nodes_num', dest='nodes_num', default=20)
args = parser.parse_args()


# METHODS = ['structured', 'naive', 'queue']
METHOD_COLORS = {'queue':'red', 'best_k': 'blue', 'struct_best_k':'black', 'graft':'green'}
# METHODS = ['structured', 'queue']
METHODS = []

METHODS = ['graft' ]




def main():

	node_reg = 0
	len_data = int(args.data_len)
	alpha = float(args.alpha)
	_lambda = float(args._lambda)
	num_nodes = int(args.nodes_num)

	priority_graft_iter = 1500
	suffstats_ratio = .05
	training_ratio = .6
	state_num = 5
	T_likelihoods = dict()
	edge_std = 1
	node_std = 1

	F1_alpha = list()

	k = 5

	mrf_density = float(3)/((num_nodes - 1))
	print('================================= ///////////////////START//////////////// ========================================= ')
	print('======================================Simulating data...')

	precisions_nofreeze, recall_nofreeze, time_nofreeze, precisions_freeze, recall_freeze, time_freeze = list(), list(), list(), list(), list(), list()


	model, variables, data, max_num_states, num_states, real_edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
	shuffle(data)
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

	edge_reg = (1 - alpha) * _lambda 
	l1 = alpha * _lambda


	edge_num = np.inf #MAX NUM EDGES TO GRAFT

	num_attributes = len(variables)
	recalls, precisions, active_sets= dict(), dict(), dict()

	M_time_stamps = dict()
	objs = dict()
	f1_scores = dict()

	edges = real_edges



	print('>>>>>>>>>>>>>>>>>>>>>METHOD: queue')
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg, l1_coeff=l1)
	sspg.set_top_relvant(k=1)
	sspg.on_monitor_mn()
	sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	M_time_stamps['queue'] = sorted(list(sspg.mn_snapshots.keys()))
	objs['queue'] = objec
	f1_scores['queue'] = f1_score
	recalls['queue'] = recall
	METHODS.append('queue')


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: best_k')
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg,l1_coeff=l1)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	sspg.on_verbose()
	# sspg.on_structured()

	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	M_time_stamps['best_k'] = sorted(list(sspg.mn_snapshots.keys()))
	objs['best_k'] = objec
	f1_scores['best_k'] = f1_score
	recalls['best_k'] = recall
	METHODS.append('best_k')


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: struct_best_k')
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg, l1_coeff=l1)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	sspg.on_verbose()
	sspg.on_structured()

	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	M_time_stamps['struct_best_k'] = sorted(list(sspg.mn_snapshots.keys()))
	objs['struct_best_k'] = objec
	f1_scores['struct_best_k'] = f1_score
	recalls['struct_best_k'] = recall
	METHODS.append('struct_best_k')


	print('================GRAFT')
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	grafter.on_show_metrics()
	# grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1 = edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg, l1_coeff=l1)
	grafter.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
	print(final_active_set)
	print(recall)
	print('ALPHA')
	print(alpha)
	objs['graft'] = objec
	M_time_stamps['graft'] = sorted(list(grafter.mn_snapshots.keys()))
	f1_scores['graft'] = f1_score
	exec_time = time.time() - t
	precisions['graft'] = precision
	recalls['graft'] = recall
	METHODS.append('graft')

	F1_alpha.append(f1_scores)

	print(objs)
	


	# for method in METHODS:
	# 	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
	# 	pq = copy.deepcopy(original_pq)
	# 	spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict = pq)
	# 	spg.on_show_metrics()
	# 	spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
	# 	spg.on_monitor_mn()
	# 	spg.on_verbose()
	# 	# spg.on_plot_queue('../../../pq_plot')
	# 	t = time.time()
	# 	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
	# 	exec_time = time.time() - t
	# 	print('---->Exec time')
	# 	print(exec_time)
	# 	print('Loss')
	# 	print(objec)
	# 	M_time_stamps[method] = sorted(list(spg.mn_snapshots.keys()))
	# 	objs[method] = objec
	# 	f1_scores[method] = f1_score


	# METHODS.extend(['queue','best_k','struct_best_k', 'graft'])
	plt.close()
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='Loss-' + METHODS[i], linewidth=1)
		ax2.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label='Recall-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Recall')
	ax2.legend(fancybox=True, framealpha=0.5, bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
	ax1.legend(loc='best', framealpha=0.5)
	plt.title('Loss-Recall')
	plt.savefig('../../../results_l1/' + str(_lambda) + '/' + str(alpha) + '/Recall-Loss.eps', format='eps', dpi=1000)
	plt.close()


if __name__ == '__main__':
	main()