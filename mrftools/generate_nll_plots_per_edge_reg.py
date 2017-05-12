import time
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

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

METHOD_COLORS = {'best_10':'yellow', 'queue':'green', 'graft':'blue'}

METHOD_legend = dict()
METHOD_marker = dict()

parser = argparse.ArgumentParser()
parser.add_argument('--nodes_num', dest='num_nodes', required=True)
parser.add_argument('--group_l1', dest='group_l1', required=True)
parser.add_argument('--l2', dest='l2', default=0.5)
# parser.add_argument('--node_std', dest='node_std', default=.01)
parser.add_argument('--state_num', dest='state_num', default=5)
parser.add_argument('--len_data', dest='len_data', default=20000)
args = parser.parse_args()


folder_name = 'compare_nll_edge_reg_new'
folder_num = 'l1_metrics'
num_iterations = 1
is_real_loss = False


METHODS = [] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

def main():
	priority_graft_iter = 2500
	graft_iter = 2500
	# zero_threshold = 1e-3
	# min_precision = .05

	print('================================= ///////////////////START//////////////// =========================================')

	################################################################### DATA PREPROCESSING GOES HERE --------->
	num_nodes = int(args.num_nodes)
	group_l1 = float(args.group_l1)
	l2 = float(args.l2)
	edge_reg = group_l1
	node_reg = group_l1
	l1 = 0

	print('edge_reg')
	print(edge_reg)

	print('l2')
	print(l2)

	training_ratio = .95
	# edge_std = float(args.edge_std)
	# node_std = float(args.node_std)
	edge_std = 1
	node_std = .5

	state_num = int(args.state_num)
	mrf_density = float(2)/((num_nodes - 1))
	len_data = int(args.len_data)
	M_accuracies = dict()
	sorted_timestamped_mn = dict()
	edge_likelihoods = dict()
	print('======================================Simulating data...')
	model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
	shuffle(data)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]
	#############################################################################################<-------------

	RES_SHELVE = shelve.open('results_' + str(num_nodes) + '_' + str(edge_reg) + '_' + str(l2))

	params = {'num_nodes':num_nodes, 'edge_std':edge_std, 'node_std':node_std, 'state_num':state_num, 'len_data':len_data, 'mrf_density':mrf_density}

	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)

	ss_test = dict()

	max_update_step = int(np.sqrt(len(variables)))

	k = 10 * len(variables)

	# #Uncomment if real loss computation is required
	#########################
	# ss_test = get_all_ss(variables, num_states, train_data)
	#########################


	original_pq = initialize_priority_queue(variables=variables)

	print(variables)
	print(num_states)
	print(max_num_states)

	print('NUM VARIABLES')
	print(len(variables))

	print('NUM EDGES')
	print(len(edges))

	print('EDGES')
	print(edges)

	# edge_num = int(5 * num_nodes) # MAX NUM EDGES TO GRAFT
	edge_num = float('inf')

	recalls, precisions, sufficientstats, mn_snapshots, recalls,f1_scores, objs, test_nlls, train_nlls, M_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()

	# k = len(variables)
	alpha = 1
	# max_update_step = int(np.sqrt(len(variables)))
	meth = '$ '+ str(alpha) + ' $'
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
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
	print('Loss')
	print(objec)
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = sspg.mn_snapshots
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	recalls[meth] = recall
	METHOD_COLORS[meth] = [1, 0.5, 0.0]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = 'o'



	# k = len(variables)
	alpha = .75
	# max_update_step = int(np.sqrt(len(variables)))
	meth = '$ '+ str(alpha) + '$'
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	# sspg.set_select_unit(select_unit=select_unit)
	sspg.set_alpha(alpha=alpha)
	sspg.set_max_update_step(max_update_step=max_update_step)
	sspg.on_monitor_mn()
	sspg.on_structured()
	# sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
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
	METHOD_COLORS[meth] = [0.35, 1, 0.85]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = 'v'



	# k = len(variables)
	alpha = .5
	# max_update_step = int(np.sqrt(len(variables)))
	meth = '$ '+ str(alpha) + '$'
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	# sspg.set_select_unit(select_unit=select_unit)
	sspg.set_alpha(alpha=alpha)
	sspg.set_max_update_step(max_update_step=max_update_step)
	sspg.on_monitor_mn()
	sspg.on_structured()
	# sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
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
	METHOD_COLORS[meth] = [0.2, 0.6, .7]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = '<'


	# k = len(variables)
	alpha = .25
	# max_update_step = int(np.sqrt(len(variables)))
	meth = '$ '+ str(alpha) + '$'
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	# sspg.set_select_unit(select_unit=select_unit)
	sspg.set_alpha(alpha=alpha)
	sspg.set_max_update_step(max_update_step=max_update_step)
	sspg.on_monitor_mn()
	sspg.on_structured()
	# sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
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
	METHOD_COLORS[meth] = [.9, .3, .7]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = '>'



	# k = len(variables)
	alpha = 0
	# max_update_step = int(np.sqrt(len(variables)))
	meth = '$ '+ str(alpha) + '$'
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=l1, l2_coeff=l2)
	sspg.set_top_relvant(k=k)
	# sspg.set_select_unit(select_unit=select_unit)
	sspg.set_alpha(alpha=alpha)
	sspg.set_max_update_step(max_update_step=max_update_step)
	sspg.on_monitor_mn()
	sspg.on_structured()
	# sspg.on_verbose()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
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
	METHOD_COLORS[meth] = [.5, .7, .2]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = 's'
	

    ###########################################################################

	print('>>>>>>>>>>>>>>>>>>>>>METHOD: queue')
	meth = 'First Hit'
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg, l2_coeff=l2, l1_coeff=l1)
	sspg.set_top_relvant(k=1)
	sspg.on_monitor_mn()
	sspg.set_max_update_step(max_update_step=1)
	# sspg.on_verbose()
	sspg.on_structured()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num, edges=edges)
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
	METHOD_COLORS[meth] = [0, 0, 0]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = '8'


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	meth = 'EG'
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	grafter.on_show_metrics()
	grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg, l1_coeff=l1, l2_coeff=l2)
	grafter.on_monitor_mn(is_real_loss=is_real_loss)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	mn_snapshots[meth] = grafter.mn_snapshots
	time_stamps = sorted(list(grafter.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	recalls[meth] = recall
	METHOD_COLORS[meth] = [0.75, 0.75, 0.75]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = 'h'


	#UNCOMMENT TO PLOT F1 SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		f1 = f1_scores[METHODS[i]]
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0,  label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])

	ax1.set_xlabel('Time')
	ax1.set_ylabel('F1 Score')
	fontP = FontProperties()
	fontP.set_size('small')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	# ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.title('F1 VS Time')
	if f1[-1] > .4:
		plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) + '/F1*.eps',linewidth=2.5, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	else:
		plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) + '/F1.eps',linewidth=2.5, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.close()


	#UNCOMMENT TO PLOT Recall SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':',marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Recall Score')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.title('Recall VS Time')
	plt.savefig('../../../results_' + folder_name + '/'+ str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) +'/Recall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.close()


	#UNCOMMENT TO PLOT Recall SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]],fillstyle='full', markeredgewidth=0.0, label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Activation Iteration')
	ax1.set_ylabel('Recall Score')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.title('Recall VS Iter')
	plt.savefig('../../../results_' + folder_name + '/'+ str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) +'/IterRecall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.close()


	#UNCOMMENT TO PLOT nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('LOSS')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	# plt.title('LOSS VS Time')
	plt.savefig('../../../results_' + folder_name + '/'+ str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) + '/OBJ.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
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
			test_nll = compute_likelihood(mn_snaps[t], len(variables), test_data)
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
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(M_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(M_time_stamps[METHODS[i]], test_nlls
				[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Test NPLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	# plt.title('Test NLL VS Time')
	plt.savefig('../../../results_' + folder_name + '/' +str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) + '/NLL.eps', format='eps', dpi=1000)
	plt.close()

if __name__ == '__main__':
	main()

