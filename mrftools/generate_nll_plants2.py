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

METHOD_COLORS = {'structured':'red', 'S-SSPG': 'black', 'SSPG': 'green', 'queue':'yellow', 'graft':'blue'}

parser = argparse.ArgumentParser()
parser.add_argument('--edge_reg', dest='edge_reg', default=.01)
parser.add_argument('--edge_num', dest='edge_num', default=1)
args = parser.parse_args()



folder_name = 'results_plants'
num_iterations = 1
is_real_loss = False

if not os.path.exists(folder_name):
	os.mkdir(folder_name)

METHODS = ['queue'] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

def main():
	priority_graft_iter = 5000
	graft_iter = 5000

	print('================================= ///////////////////START//////////////// =========================================')

	################################################################### DATA PREPROCESSING GOES HERE --------->
	training_ratio = .9
	edge_reg = float(args.edge_reg)
	edge_num = int(args.edge_num)

	M_accuracies = dict()
	sorted_timestamped_mn = dict()
	edge_likelihoods = dict()
	from read_plants import read_plants
	data, num_states, max_num_states, variables = read_plants()
	len_data = len(data)

	#############################################################################################<-------------

	shuffle(data)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]

	RES_SHELVE = shelve.open('shelves/rating_results_' + str(edge_num) + '_' + str(edge_reg))

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


	k = 10

	print('>>>>>>>>>>>>>>>>>>>>>METHOD: SSPG')
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	# sspg.on_verbose()
	# sspg.on_structured()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps['SSPG'] = time_stamps
	mn_snapshots['SSPG'] = sspg.mn_snapshots
	objs['SSPG'] = objec
	METHODS.append('SSPG')


	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Str-SSPG')
	pq = copy.deepcopy(original_pq)
	sspg = SelectiveStructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict = pq)
	sspg.on_show_metrics()
	sspg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
	sspg.set_top_relvant(k=k)
	sspg.on_monitor_mn()
	# sspg.on_verbose()
	sspg.on_structured()
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = sspg.learn_structure(edge_num)
	exec_time = time.time() - t
	time_stamps = sorted(list(sspg.mn_snapshots.keys()))
	M_time_stamps['S-SSPG'] = time_stamps
	mn_snapshots['S-SSPG'] = sspg.mn_snapshots
	objs['S-SSPG'] = objec
	METHODS.append('S-SSPG')


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

	results = {'methods':METHODS, 'time_stamps':M_time_stamps, 'test_nlls':test_nlls, 'objs':objs, 'params':params}
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
	plt.savefig(folder_name + '/' + str(edge_num) + '/' + str(edge_reg) + '/NLL_.png')
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
	plt.savefig(folder_name + '/' + str(edge_num) + '/' + str(edge_reg) +'/OBJ_.png')
	plt.close()

if __name__ == '__main__':
	main()


