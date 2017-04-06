import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1, get_all_ss, initialize_priority_queue
from SelectiveStructuredPriorityGraft import SelectiveStructuredPriorityGraft
import time
from Graft import Graft
import copy
import itertools
import networkx as nx
import os
import sys
import argparse
import shelve

METHOD_COLORS = dict()

parser = argparse.ArgumentParser()
parser.add_argument('--group_l1', dest='group_l1', required=True)
parser.add_argument('--l2', dest='l2', default=0.5)
parser.add_argument('--edge_num', dest='edge_num', default=100)
args = parser.parse_args()


folder_name = 'results_ratings'
num_iterations = 1
is_real_loss = False

METHODS = [] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

def main():
	priority_graft_iter = 2500
	graft_iter = 1500

	print('================================= ///////////////////START//////////////// =========================================')

	training_ratio = .9
	edge_num = int(args.edge_num)

	group_l1 = float(args.group_l1)
	l2 = float(args.l2)
	edge_reg = group_l1
	node_reg = group_l1
	l1 = 0

	M_accuracies = dict()
	sorted_timestamped_mn = dict()
	edge_likelihoods = dict()

	################################################################### DATA PREPROCESSING GOES HERE --------->
	from read_ratings import read_ratings_from_batch_files

	FILES_LIST = ['../../../ratings/jester-data-1.xls', '../../../ratings/jester-data-2.xls', '../../../ratings/jester-data-3.xls'] 

	num_states = dict()
	variables = range(1,101)
	num_attributes = len(variables)
	for i in range(1,101):
		num_states[i] = 5
	max_num_states = 5
	print('FETCHING DATA!')
	data = read_ratings_from_batch_files(FILES_LIST, 50)
	len_data = len(data)
	print('LEN DATA')
	print(len_data)

	#############################################################################################<-------------

	shuffle(data)

	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]

	RES_SHELVE = shelve.open('shelves/rating_results_' + str(edge_num) + '_' + str(edge_reg)+ '_' + str(l2))

	params = {'edge_reg':edge_reg, 'node_reg':node_reg, 'len_data':len_data}

	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)

	ss_test = dict()

	# #Uncomment if real loss computation is required
	#########################
	# ss_test = get_all_ss(variables, num_states, train_data)
	#########################


	original_pq = initialize_priority_queue(variables=variables)

	print(num_states)
	recalls, precisions, sufficientstats, mn_snapshots, f1_scores, objs, test_nlls, train_nlls, M_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()

	print('>>>>>>>>>>>>>>>>>>>>>METHOD: queue')
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg,l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=1)
	sspg.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('Loss')
	print(objec)
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps['queue'] = time_stamps
	mn_snapshots['queue'] = sspg.mn_snapshots
	objs['queue'] = objec
	f1_scores['queue'] = f1_score
	METHODS.append('queue')
	recalls['queue'] = recall
	METHOD_COLORS['queue'] = [1, 1, 0.0]


	k = 50
	meth = 'best_' + str(k)
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('Loss')
	print(objec)
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = [1, 0.0, 0.0]

	k = 100
	meth = 'best_' + str(k)
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('Loss')
	print(objec)
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = [0, 1.0, 0.0]

	k = 100
	meth = 'struct_best_' + str(k)
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('Loss')
	print(objec)
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = [0, 0.75, 0.0]



	k = 200
	meth = 'best_' + str(k)
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	print('Loss')
	print(objec)
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = [0.75, 0.75, 0.75]


	# k = 200
	# meth = 'str_best_' + str(k)
	# print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	# pq = copy.deepcopy(original_pq)
	# sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	# sspg.on_show_metrics()
	# sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg ,  l1_coeff=l1,l2_coeff=l2)
	# sspg.set_top_relvant(k=k)
	# sspg.on_monitor_mn()
	# sspg.on_structured()
	# t = time.time()
	# learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	# exec_time = time.time() - t
	# print('---->Exec time')
	# print(exec_time)
	# print('Loss')
	# print(objec)
	# time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	# M_time_stamps[meth] = time_stamps
	# mn_snapshots[meth] = sspg.mn_snapshots
	# objs[meth] = objec
	# f1_scores[meth] = f1_score
	# METHODS.append(meth)
	# recalls[meth] = recall


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order, ss_shelve= True)
	grafter.on_show_metrics()
	# grafter.on_limit_sufficient_stats(.1)
	grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg, l1_coeff=l1, l2_coeff=l2)
	grafter.on_monitor_mn(is_real_loss=is_real_loss)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, ss_shelve=True, ratings_data=True, ss_time=225.0)
	objs['graft'] = objec
	f1_scores['graft'] = f1_score
	METHODS.append('graft')
	mn_snapshots['graft'] = grafter.mn_snapshots
	time_stamps = sorted(list(grafter.mn_snapshots.keys()))
	M_time_stamps['graft'] = time_stamps
	recalls['graft'] = recall
	print('Times')
	print(grafter.mn_snapshots.keys())
	METHOD_COLORS['graft'] = [0.0, 0.0, 0.0]


	#UNCOMMENT TO PLOT nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, label='LOSS-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('LOSS')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.title('LOSS VS Time')
	plt.savefig('../../../' + folder_name + '/'+ str(edge_num) + '/' + str(group_l1) + '/' + str(l2) + '/OBJ.eps', format='eps', dpi=1000)
	plt.close()
	results = {'methods':METHODS, 'time_stamps': M_time_stamps, 'train_nlls':train_nlls, 'test_nlls':test_nlls, 'recall':recalls, 'f1':f1_scores, 'objs':objs, 'params':params}
	RES_SHELVE.update(results)
	RES_SHELVE.close()


	#COMPUTE NLLS
	#########################################################
	for method in METHODS:
		print(method)
		test_nll_list = list()
		train_nll_list = list()
		mn_snaps = mn_snapshots[method]
		for t in M_time_stamps[method]:
			test_nll = compute_likelihood(mn_snaps[t], len(variables), test_data, variables = variables)
			# train_nll = compute_likelihood(mn_snaps[t], len(variables), train_data)
			test_nll_list.append(test_nll)
			# train_nll_list.append(train_nll)
		test_nlls[method] = test_nll_list
		# train_nlls[method] = train_nll_list
	#########################################################

	#UNCOMMENT TO PLOT test nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color= METHOD_COLORS[METHODS[i]], linewidth=2, label='TestNLL-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Test NLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.title('Test NLL VS Time')
	plt.savefig('../../../' + folder_name + '/' + str(edge_num) + '/' + str(group_l1) + '/' + str(l2) +  '/NLL.eps', format='eps', dpi=1000)
	plt.close()

if __name__ == '__main__':
	main()


