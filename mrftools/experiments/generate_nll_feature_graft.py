import time
import numpy as np
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../util')
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1, get_all_ss, initialize_priority_queue
from SelectiveStructuredPriorityGraft import SelectiveStructuredPriorityGraft
from FeatureGraft import FeatureGraft
import time
from Graft import Graft
import copy
import itertools
import networkx as nx
import os
import argparse
import shelve

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

METHOD_COLORS = {'best_10':'yellow', 'queue':'green', 'graft':'blue'}

METHOD_legend = dict()

parser = argparse.ArgumentParser()
parser.add_argument('--nodes_num', dest='num_nodes', required=True)
parser.add_argument('--group_l1', dest='group_l1', default=0.0)
parser.add_argument('--l2', dest='l2', default=0.1)
parser.add_argument('--l1', dest='l1', default=0.5)
# parser.add_argument('--node_std', dest='node_std', default=.01)
parser.add_argument('--state_num', dest='state_num', default=5)
parser.add_argument('--len_data', dest='len_data', default=2000)
args = parser.parse_args()


folder_name = 'feature'
folder_num = 'l1_metrics'
num_iterations = 1
is_real_loss = False


METHODS = [] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

def main():
	priority_graft_iter = 2500
	graft_iter = 2500

	print('================================= ///////////////////START//////////////// =========================================')

	################################################################### DATA PREPROCESSING GOES HERE --------->
	num_nodes = int(args.num_nodes)
	group_l1 = float(args.group_l1)
	l2 = float(args.l2)
	l1 = float(args.l1)
	edge_reg = group_l1
	node_reg = group_l1


	training_ratio = .95

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

	# RES_SHELVE = shelve.open('results_' + str(num_nodes) + '_' + str(edge_reg) + '_' + str(l2))

	# params = {'num_nodes':num_nodes, 'edge_std':edge_std, 'node_std':node_std, 'state_num':state_num, 'len_data':len_data, 'mrf_density':mrf_density}

	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)

	ss_test = dict()

	max_update_step = 4 * int(np.sqrt(len(variables)))

	k = 2 *len(variables)

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

	edge_num = int(4 * num_nodes) # MAX NUM EDGES TO GRAFT

	# num_feature = edge_num * state_num

	num_feature = 5

	recalls, precisions, sufficientstats, mn_snapshots, recalls,f1_scores, objs, test_nlls, train_nlls, M_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Feature Graft')
	meth = 'FG'
	fg = FeatureGraft(variables, num_states, max_num_states, train_data, list_order)
	fg.on_verbose()
	fg.on_show_metrics()
	# fg.on_synthetic(precison_threshold = min_precision)
	fg.setup_learning_parameters(max_iter_graft=priority_graft_iter, l1_coeff=l1, l2_coeff=l2)
	fg.on_monitor_mn()
	t = time.time()
	learned_mn, final_active_set, recall, precision, f1_score, objec, is_early_stop = fg.learn_structure(num_feature, edges=edges)

	objs[meth] = objec
	f1_scores[meth] = f1_score
	METHODS.append(meth)
	time_stamps = sorted(list(fg.mn_snapshots.keys()))
	M_time_stamps[meth] = time_stamps
	recalls[meth] = recall
	METHOD_COLORS[meth] = [0.75, 0.75, 0.75]
	METHOD_legend[meth] = meth

 #    ###########################################################################

	# print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	# meth = 'EG'
	# grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	# grafter.on_show_metrics()
	# grafter.on_verbose()
	# grafter.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg, l2_coeff=l2)
	# grafter.on_monitor_mn(is_real_loss=is_real_loss)
	# t = time.time()
	# learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
	# objs[meth] = objec
	# f1_scores[meth] = f1_score
	# METHODS.append(meth)
	# mn_snapshots[meth] = grafter.mn_snapshots
	# time_stamps = sorted(list(grafter.mn_snapshots.keys()))
	# M_time_stamps[meth] = time_stamps
	# recalls[meth] = recall
	# METHOD_COLORS[meth] = [0.75, 0.75, 0.75]
	# METHOD_legend[meth] = meth


	# #UNCOMMENT TO PLOT F1 SCORES EVOLUTION
	# plt.close()
	# fig, ax1 = plt.subplots()
	# for i in range(len(METHODS)):
	# 	print(METHODS[i])
	# 	f1 = f1_scores[METHODS[i]]
	# 	ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, linestyle=':', marker='o', fillstyle='full', edgecolor='none', label=METHOD_legend[METHODS[i]])

	# ax1.set_xlabel('Time')
	# ax1.set_ylabel('F1 Score')
	# fontP = FontProperties()
	# fontP.set_size('small')
	# lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	# # ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	# plt.title('F1 VS Time')
	# plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) + '/F1.eps',linewidth=2, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	# plt.close()

	#UNCOMMENT TO PLOT Recall SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, linestyle=':', marker='+',fillstyle='full', label=METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Recall Score')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title('Recall VS Time')
	plt.savefig('../../../results_' + folder_name + '/'+ str(len(variables)) + '/' + str(l1) + '/Recall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.close()

	#UNCOMMENT TO PLOT Recall SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, linestyle=':', marker='+',fillstyle='full', label=METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Iter')
	ax1.set_ylabel('Recall Score')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title('Recall VS Iter')
	plt.savefig('../../../results_' + folder_name + '/'+ str(len(variables)) + '/' + str(l1) + '/IterRecall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.close()

	#UNCOMMENT TO PLOT nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, label=METHOD_legend[METHODS[i]])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('LOSS')
	lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.title('LOSS VS Time')
	plt.savefig('../../../results_' + folder_name + '/'+ str(len(variables)) + '/' + str(l1) + '/OBJ.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.close()

	# results = {'methods':METHODS, 'time_stamps': M_time_stamps, 'train_nlls':train_nlls, 'test_nlls':test_nlls, 'recall':recalls, 'f1':f1_scores, 'objs':objs, 'params':params}
	# RES_SHELVE.update(results)
	# RES_SHELVE.close()


	# #COMPUTE NLLS
	# #########################################################
	# for method in METHODS:
	# 	print(method)
	# 	test_nll_list = list()
	# 	train_nll_list = list()
	# 	mn_snaps = mn_snapshots[method]
	# 	for t in M_time_stamps[method]:
	# 		test_nll = compute_likelihood(mn_snaps[t], len(variables), test_data)
	# 		# train_nll = compute_likelihood(mn_snaps[t], len(variables), train_data)
	# 		test_nll_list.append(test_nll)
	# 		# train_nll_list.append(train_nll)
	# 	test_nlls[method] = test_nll_list
	# 	# train_nlls[method] = train_nll_list
	# #########################################################

	# #UNCOMMENT TO PLOT test nll SCORES EVOLUTION
	# plt.close()
	# fig, ax1 = plt.subplots()
	# for i in range(len(METHODS)):
	# 	print(METHODS[i])
	# 	ax1.plot(M_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, label=METHODS[i])
	# ax1.set_xlabel('Time')
	# ax1.set_ylabel('Test NLL')
	# ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	# plt.title('Test NLL VS Time')
	# plt.savefig('../../../results_' + folder_name + '/' +str(len(variables)) + '/' + str(group_l1) + '/' + str(l2) + '/NLL.eps', format='eps', dpi=1000)
	# plt.close()

if __name__ == '__main__':
	main()

