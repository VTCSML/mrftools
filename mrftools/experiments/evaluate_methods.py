import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../util')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1, get_all_ss, initialize_priority_queue
from OnlineEdgeGrafting import OnlineEdgeGrafting
from Graft import Graft
import copy
import itertools
import networkx as nx
import os
import argparse
import shelve
from matplotlib.font_manager import FontProperties



def	evaluate_methods(num_states, variables, num_attributes, max_num_states, data, len_data, results_dir, shelve_dir, args, alphas, max_update_step, edge_reg, node_reg, \
	experiments_name, edge_num, edges=None, experiments_type='real', priority_graft_iter=2500, graft_iter=2500, training_ratio=.9):

	assert experiments_type in ['real', 'synthetic']
	METHOD_COLORS = dict()
	METHOD_legend = dict()
	METHOD_marker = dict()
	METHODS = [] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

	shuffle(data)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]
	RES_SHELVE = shelve.open(shelve_dir + '/' + experiments_name + '_results_' + str(edge_num) + '_' + str(edge_reg) + '_' + args.l2)
	params = {'edge_reg':edge_reg, 'node_reg':node_reg, 'len_data':len_data}
	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)
	ss_test = dict()
	original_pq = initialize_priority_queue(variables=variables)
	recalls, precisions, sufficientstats, mn_snapshots, f1_scores, objs, test_nlls, train_nlls, M_time_stamps, subsampled_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
	k = len(variables)

	for alpha in alphas.keys():
		meth = '$ '+ str(alpha) + '$ '
		print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
		pq = copy.deepcopy(original_pq)
		oeg = OnlineEdgeGrafting(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
		oeg.on_show_metrics()
		oeg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg , l1_coeff=float(args.l1), l2_coeff=float(args.l2))
		oeg.set_top_relvant(k=k)
		oeg.on_structured()
		# oeg.set_select_unit(select_unit=select_unit)
		oeg.set_alpha(alpha=alpha)
		oeg.set_max_update_step(max_update_step=max_update_step)
		oeg.on_monitor_mn()
		# oeg.on_verbose()
		t = time.time()
		if experiments_type == 'real':
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = oeg.learn_structure(edge_num)
		if experiments_type == 'synthetic':
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = oeg.learn_structure(edge_num, edges=edges)
		exec_time = time.time() - t
		print('---->Exec time')
		print(exec_time)
		time_stamps = sorted(oeg.timestamps)
		M_time_stamps[meth] = time_stamps
		subsampled_time_stamps[meth] = sorted(oeg.mn_snapshots.keys())
		mn_snapshots[meth] = oeg.mn_snapshots
		objs[meth] = objec
		if experiments_type == 'synthetic':
			f1_scores[meth] = f1_score
			recalls[meth] = recall
		METHODS.append(meth)
		METHOD_legend[meth] = meth
		METHOD_COLORS[meth] = alphas[alpha][0]
		METHOD_marker[meth] = alphas[alpha][1]
		test_nlls[meth] = get_test_nll(meth, mn_snapshots, subsampled_time_stamps, variables, test_data)


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: First Hit')
	meth = 'First Hit'
	pq = copy.deepcopy(original_pq)
	oeg = OnlineEdgeGrafting(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	oeg.on_show_metrics()
	oeg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg, l1_coeff=float(args.l1), l2_coeff=float(args.l2))
	oeg.set_top_relvant(k=1)
	oeg.on_monitor_mn()
	oeg.set_max_update_step(max_update_step=1)
	# oeg.on_verbose()
	oeg.on_structured()
	t = time.time()
	if experiments_type == 'real':
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = oeg.learn_structure(edge_num)
	if experiments_type == 'synthetic':
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = oeg.learn_structure(edge_num, edges=edges)
	exec_time = time.time() - t
	print('---->Exec time')
	print(exec_time)
	time_stamps = sorted(oeg.timestamps)
	M_time_stamps[meth] = time_stamps
	subsampled_time_stamps[meth] = sorted(oeg.mn_snapshots.keys())
	M_time_stamps[meth] = time_stamps
	mn_snapshots[meth] = oeg.mn_snapshots
	objs[meth] = objec
	if experiments_type == 'synthetic':
		f1_scores[meth] = f1_score
		recalls[meth] = recall
	METHODS.append(meth)
	METHOD_COLORS[meth] = [0, 0, 0]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = '8'
	test_nlls[meth] = get_test_nll(meth, mn_snapshots, subsampled_time_stamps, variables, test_data)


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	meth = 'EG'
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	grafter.on_show_metrics()
	grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg, l1_coeff=float(args.l1), l2_coeff=float(args.l2))
	grafter.on_monitor_mn(is_real_loss=False)
	t = time.time()
	if experiments_type == 'real':
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num)
	if experiments_type == 'synthetic':
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
	objs[meth] = objec
	METHODS.append(meth)
	mn_snapshots[meth] = grafter.mn_snapshots
	time_stamps = sorted(grafter.timestamps)
	M_time_stamps[meth] = time_stamps
	subsampled_time_stamps[meth] = sorted(grafter.mn_snapshots.keys())
	M_time_stamps[meth] = time_stamps
	if experiments_type == 'synthetic':
		f1_scores[meth] = f1_score
		recalls[meth] = recall
	METHOD_COLORS[meth] = [0.75, 0.75, 0.75]
	METHOD_legend[meth] = meth
	METHOD_marker[meth] = 'h'
	test_nlls[meth] = get_test_nll(meth, mn_snapshots, subsampled_time_stamps, variables, test_data)

	results = {'subsampled_time_stamps': subsampled_time_stamps, 'methods':METHODS, 'time_stamps': M_time_stamps, 'train_nlls':train_nlls, 'test_nlls':test_nlls, 'recall':recalls, 'f1':f1_scores, 'objs':objs, 'params':params}
	RES_SHELVE.update(results)
	RES_SHELVE.close()

	###############MAKE PLOTS

	print('>Making plots')

	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])

	ax1.set_xlabel('Time')
	ax1.set_ylabel('LOSS')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.savefig(results_dir + '/'+ str(len(variables)) + '/' + str(edge_reg) + '/' + args.l2 + '/OBJ.eps', format='eps', dpi=1000)
	plt.close()


	#UNCOMMENT TO PLOT test nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
			ax1.plot(subsampled_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color= METHOD_COLORS[METHODS[i]], linewidth=2, label=METHOD_legend[METHODS[i]])
		else:
			ax1.plot(subsampled_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color= METHOD_COLORS[METHODS[i]], linewidth=2, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Test NLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.savefig(results_dir + '/'+ str(len(variables)) + '/' + str(edge_reg) + '/' + args.l2 + '/NLL.eps', format='eps', dpi=1000)
	plt.close()

	if experiments_type == 'synthetic':
		#UNCOMMENT TO PLOT F1 SCORES EVOLUTION
		plt.close()
		fig, ax1 = plt.subplots()
		for i in range(len(METHODS)):
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
		plt.title('F1 VS Time')
		if f1[-1] > .4:
			plt.savefig(results_dir + '/'+ str(len(variables)) + '/' + str(edge_reg) + '/' + args.l2 + '/F1*.eps',linewidth=2.5, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
		else:
			plt.savefig(results_dir + '/'+ str(len(variables))+ '/' + str(edge_reg) + '/' + args.l2 + '/' + str(args.l2) + '/F1.eps',linewidth=2.5, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
		plt.close()

		#UNCOMMENT TO PLOT Recall SCORES EVOLUTION
		plt.close()
		fig, ax1 = plt.subplots()
		for i in range(len(METHODS)):
			if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
				ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=METHOD_legend[METHODS[i]])
			else:
				ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':',marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
		ax1.set_xlabel('Time')
		ax1.set_ylabel('Recall Score')
		lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.savefig(results_dir + '/'+ str(len(variables)) + '/' + str(edge_reg) + '/' + args.l2 + '/Recall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
		plt.close()

		fig, ax1 = plt.subplots()
		for i in range(len(METHODS)):
			if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
				ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]],fillstyle='full', markeredgewidth=0.0, label=METHOD_legend[METHODS[i]])
			else:
				ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
		ax1.set_xlabel('Activation Iteration')
		ax1.set_ylabel('Recall Score')
		lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.savefig(results_dir + '/'+ str(len(variables)) + '/' + str(edge_reg) + '/' + args.l2 + '/IterRecall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
		plt.close()

def get_test_nll(method, mn_snapshots, subsampled_time_stamps, variables, test_data):
	test_nll_list = list()
	train_nll_list = list()
	mn_snaps = mn_snapshots[method]
	for t in subsampled_time_stamps[method]:
		test_nll = compute_likelihood(mn_snaps[t], len(variables), test_data)
		test_nll_list.append(test_nll)
	return test_nll_list
