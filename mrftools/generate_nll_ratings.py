import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1, get_all_ss, initialize_priority_queue
import time
from Graft import Graft
import copy
import itertools
import networkx as nx
import os
import sys
import argparse
import shelve

METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'black', 'graft':'blue'}
METHOD_COLORS_i = {'structured':'r', 'naive': 'g', 'queue':'y', 'graft':'b'}

parser = argparse.ArgumentParser()
parser.add_argument('--edge_reg', dest='edge_reg', default=.01)
args = parser.parse_args()


folder_name = 'compare_nll'
folder_num = 'l1_metrics'
num_iterations = 1
is_real_loss = False

METHODS = ['structured', 'queue'] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

def main():
	priority_graft_iter = 5000
	graft_iter = 5000
	# zero_threshold = 1e-3
	# edge_reg_range = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3,\
	# 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,\
	#  2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1]

	print('================================= ///////////////////START//////////////// =========================================')

	################################################################### DATA PREPROCESSING GOES HERE --------->
	training_ratio = .9
	edge_reg = float(args.edge_reg)
	M_accuracies = dict()
	sorted_timestamped_mn = dict()
	edge_likelihoods = dict()
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

	#############################################################################################<-------------

	shuffle(data)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]

	RES_SHELVE = shelve.open('shelves/rating_results_' + str(edge_reg))

	node_reg = 1.05 * edge_reg

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

	edge_num = 250 # MAX NUM EDGES TO GRAFT
	recalls, precisions, sufficientstats, mn_snapshots, f1_scores, objs, test_nlls, train_nlls, M_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()

	for method in METHODS:
		print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)

		pq = copy.deepcopy(original_pq)

		spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict=pq, ss_test=ss_test)
		spg.on_show_metrics()
		spg.on_verbose()
		# spg.on_synthetic(precison_threshold = min_precision, start_num = 3) ## EARLY STOP GRAFTING IF 4 EDGES ARE ADDED AND PRECISION < min_precision
		spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
		spg.on_monitor_mn(is_real_loss=is_real_loss)
		t = time.time()
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, _ = spg.learn_structure(edge_num)

		best_objec = objec[-1]
		best_mn_snapshots = copy.deepcopy(spg.mn_snapshots)
		time_stamps = sorted(list(best_mn_snapshots.keys()))
		# best_graph_snapshots = copy.deepcopy(spg.graph_snapshots)
		objs[method] = objec
		mn_snapshots[method] = best_mn_snapshots
		M_time_stamps[method] = time_stamps

		# #UNCOMMENT TO PLOT GRAPH EVOLUTION
		#########################################################
		# plt.close()
		# j = 0
		# graph_folder = '../../../results_' + folder_name + '/spg_graphs_' + str(num_nodes) + '/'
		# try:
		# 	os.mkdir(graph_folder)
		# except:
		# 	pass
		# for t in sorted(list(best_graph_snapshots.keys())):
		# 	j += 1
		# 	G = best_graph_snapshots[t]
		# 	pos = nx.shell_layout(G)
		# 	nx.draw(G, pos)
		# 	plt.title(str(t))
		# 	plt.savefig(graph_folder + str(j) + '_SPG_graph_.png')
		# 	plt.close()
		#########################################################

	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	grafter.on_show_metrics()
	# grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg)
	grafter.on_monitor_mn(is_real_loss=is_real_loss)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num)
	objs['graft'] = objec
	f1_scores['graft'] = f1_score
	METHODS.append('graft')
	mn_snapshots['graft'] = grafter.mn_snapshots
	time_stamps = sorted(list(grafter.mn_snapshots.keys()))
	M_time_stamps['graft'] = time_stamps

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

	# #UNCOMMENT TO PLOT GRAPH EVOLUTION
	#########################################################
	# plt.close()
	# j = 0
	# graph_folder = '../../../results_' + folder_name + '/graft_graphs_' + str(num_nodes) +'/'
	# try:
	# 	os.mkdir(graph_folder)
	# except:
	# 	pass
	# for t in sorted(list(grafter.graph_snapshots.keys())):
	# 	j += 1
	# 	G = grafter.graph_snapshots[t]
	# 	pos = nx.shell_layout(G)
	# 	nx.draw(G, pos)
	# 	plt.title(str(t))
	# 	plt.savefig(graph_folder + str(j) + '_G_graph_.png')
	# 	plt.close()
	#########################################################

	results = {'methods':METHODS, 'time_stamps': M_time_stamps, 'test_nlls':test_nlls, 'objs':objs, 'params':params}
	RES_SHELVE.update(results)
	RES_SHELVE.close()

	#UNCOMMENT TO PLOT test nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], test_nlls[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='TestNLL-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Test NLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.title('Test NLL VS Time')
	plt.savefig('../../../ratings_results_' + folder_name + '/' + str(edge_reg) + '_NLL_.png')
	plt.close()

	#UNCOMMENT TO PLOT loss SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='LOSS-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('LOSS')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True)
	plt.title('LOSS VS Time')
	plt.savefig('../../../ratings_results_' + folder_name + '/' + str(edge_reg) + '_OBJ_.png')
	plt.close()

if __name__ == '__main__':
	main()


