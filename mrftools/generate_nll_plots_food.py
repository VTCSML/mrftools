
import matplotlib as mpl

mpl.use('Agg')

import time
import matplotlib.pyplot as plt
import numpy as np
# from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
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
from read_ingredients import read_ingredients
import shelve

plt.ioff()

METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'black', 'graft':'blue'}
METHOD_COLORS_i = {'structured':'r', 'naive': 'g', 'queue':'y', 'graft':'b'}


parser = argparse.ArgumentParser()
parser.add_argument('--edge_reg', dest='loaded_edge_reg', default=0.1)
args = parser.parse_args()
loaded_edge_reg = float(args.loaded_edge_reg)

print "Running with edge reg %f, %e" % (loaded_edge_reg, loaded_edge_reg)

folder_name = 'compare_nll_food_%e' % loaded_edge_reg

if not os.path.exists(folder_name):
	os.mkdir(folder_name)

num_iterations = 1
is_real_loss = False

METHODS = ['structured', 'queue'] # DON'T INCLUDE GRAFT IT WILL AUTOMATICALLY BE INCLUDED LATER

RES_SHELVE = shelve.open(folder_name + '/' + "food")


def main():
	priority_graft_iter = 5000
	graft_iter = 5000
	# zero_threshold = 1e-3
	min_precision = .2
	# edge_reg_range = [1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1]
	edge_reg_range = [loaded_edge_reg]

	print('================================= ///////////////////START//////////////// =========================================')

	################################################################### DATA PREPROCESSING GOES HERE --------->
	training_ratio = 1.0 - 1000.0 / 10000.0
	M_accuracies = dict()
	sorted_timestamped_mn = dict()
	edge_likelihoods = dict()
	print('======================================Loading data...')
	data, num_states, max_num_states, variables = read_ingredients()
	edges = []
	len_data = len(data)
	num_nodes = len(variables)
	shuffle(data)
	train_data = data[: int(training_ratio * len_data)]
	test_data = data[int(training_ratio * len_data) : len_data]

	print "%d training, %d testing" % (len(train_data), len(test_data))

	print "%d variables" % len(variables)

	#############################################################################################<-------------

	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)

	ss_test = dict()

	# # #Uncomment if real loss computation is required
	# #########################
	# ss_test = get_all_ss(variables, num_states, train_data)
	# #########################


	original_pq = initialize_priority_queue(variables=variables)

	print(variables)
	print(num_states)
	print(max_num_states)

	print('NUM VARIABLES')
	print(len(variables))

	edge_num = float('inf') # MAX NUM EDGES TO GRAFT
	recalls, precisions, sufficientstats, mn_snapshots, objs, nlls, M_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict()


	############## Get baseline NLL

	pq = copy.deepcopy(original_pq)

	edge_reg = 10000
	node_reg = edge_reg_range[0]

	spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, 'structured', pq_dict=pq,
								  ss_test=ss_test)
	spg.on_show_metrics()
	spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
	learned_mn, final_active_set, suff_stats_list, recall, precision, _, objec, is_early_stop = spg.learn_structure(
		edge_num, edges=edges)
	base_nll = compute_likelihood(learned_mn, len(variables), test_data, variables=variables)

	assert len(edges) == 0

	print "Base NLL with no edges: %e" % base_nll

	#############################


	for method in METHODS:
		if method == 'structured':
			opt_reached = False
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			j = 0
			best_params = (0,0)
			best_nll = float('inf')
			best_objec = float('inf')
			best_precision = 0
			_likelihoods = []
			first_opt_found = False
			for edge_reg in edge_reg_range:
				pq = copy.deepcopy(original_pq)

				node_reg = edge_reg

				print('======PARAMS')
				print("Edge_reg: %f" % edge_reg)
				print("Node reg: %f" % node_reg)
				j += 1
				spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict=pq, ss_test=ss_test)
				spg.on_show_metrics()
				# spg.on_verbose()
				# spg.on_synthetic(precison_threshold = min_precision, start_num = 5) ## EARLY STOP GRAFTING IF 4 EDGES ARE ADDED AND PRECISION < min_precision
				spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
				spg.on_monitor_mn(is_real_loss=is_real_loss)
				t = time.time()
				learned_mn, final_active_set, suff_stats_list, recall, precision, _, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
				print(not is_early_stop)
				# if not is_early_stop and recall[-1] == 0: # L1 coeff too big
				# 	break
				# nll_0 = compute_likelihood(spg.mn_snapshots[0], len(variables), test_data)
				nll = compute_likelihood(learned_mn, len(variables), test_data, variables=variables)
				_likelihoods.append(nll)
				# if not is_early_stop and f1_score[-1] > best_f1 and nll < best_nll and nll < nll_0:
					# best_nll = nll
				# if not is_early_stop and f1_score[-1] > best_f1:
				if not is_early_stop and nll < best_nll:
					print('NEW OPT EDGE L1. NLL: %e' % nll)
					opt_node_reg = node_reg
					opt_edge_reg = edge_reg
					print("opt_edge_reg %f" % opt_edge_reg)
					first_opt_found = True
					best_precision = precision[-1]
					best_mn_snapshots = copy.deepcopy(spg.mn_snapshots)
					best_graph_snapshots = copy.deepcopy(spg.graph_snapshots)
					# f1_scores[method] = f1_score
					objs[method] = objec
					best_params = (node_reg, edge_reg)
					time_stamps = sorted(list(best_mn_snapshots.keys()))
					M_time_stamps[method] = time_stamps
					mn_snapshots[method] = best_mn_snapshots
					# if f1_score[-1] >=.8: # good enough
					# 	print('NEW OPT NODE L1')
					# 	print(opt_node_reg)
					# 	break

			#UNCOMMENT IF PRINT EVOLUTION OF METRICS WITH EDGE L1
			#####################################################
			print('plotting')
			plt.close()
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			ax1.plot(edge_reg_range, _likelihoods, color='red', label='nll', linewidth=2)
			# ax2.plot(edge_reg_range, _recalls, color='green', linewidth=2, linestyle=':', label='recall')
			# ax2.plot(edge_reg_range, _precisions, color='blue', linewidth=2, linestyle=':', label='precision')
			ax2.set_ylim([-.1,1.1])
			ax1.set_ylabel('nll')
			# ax2.set_ylabel('precison/recall')
			ax1.set_xlabel('l1-coeff')
			ax1.legend(loc='best')
			ax2.legend(loc=4, fancybox=True, framealpha=0.5)
			ax1.set_xscale("log", nonposx='clip')
			ax2.set_xscale("log", nonposx='clip')
			plt.title('nll-precision' + '_best:' + str(best_params[0]) + ',' +str(best_params[1]))
			plt.savefig(folder_name + '/' + str(len(variables)) + '_best:' + str(best_params[0]) + ',' +str(best_params[1]) +'.png')
			plt.close()
			#####################################################

			if not first_opt_found:
				raise ValueError('OPT NOT FOUND')

			print('OPT PARAMS')
			print("opt_edge_reg: %f" % opt_edge_reg)
			print("opt_node_reg: %f" % opt_node_reg)
			opt_reached =True

			# #UNCOMMENT TO PLOT GRAPH EVOLUTION
			#########################################################
			# plt.close()
			# j = 0
			# graph_folder = folder_name + '/spg_graphs_' + str(num_nodes) + '/'
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

		else:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			pq = copy.deepcopy(original_pq)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict=pq, ss_test=ss_test)
			spg.on_show_metrics()
			spg.setup_learning_parameters(edge_l1=opt_edge_reg, max_iter_graft=priority_graft_iter, node_l1=opt_node_reg)
			spg.on_monitor_mn(is_real_loss=is_real_loss)
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
			exec_time = time.time() - t
			precisions[method] = precision
			recalls[method] = recall
			time_stamps = sorted(list(spg.mn_snapshots.keys()))
			mn_snapshots[method] = spg.mn_snapshots
			M_time_stamps[method] = time_stamps
			objs[method] = objec

	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	grafter.on_show_metrics()
	# grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1 = opt_edge_reg, max_iter_graft=graft_iter, node_l1=opt_node_reg)
	grafter.on_monitor_mn(is_real_loss=is_real_loss)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
	# print(final_active_set)
	# print(recall)
	objs['graft'] = objec
	exec_time = time.time() - t
	precisions['graft'] = precision
	recalls['graft'] = recall
	METHODS.append('graft')
	mn_snapshots['graft'] = grafter.mn_snapshots
	time_stamps = sorted(list(grafter.mn_snapshots.keys()))
	M_time_stamps['graft'] = time_stamps
	method_likelihoods = []
	# print(len(time_stamps))
	# print(len(objec))

	#COMPUTE NLLS
	#########################################################
	for method in METHODS:
		print(method)
		nll_list = list()
		mn_snaps = mn_snapshots[method]

		for t in M_time_stamps[method]:
			nll = compute_likelihood(mn_snaps[t], len(variables), train_data, variables=variables)
			nll_list.append(nll)
		nlls[method] = nll_list
	#########################################################

	#UNCOMMENT TO PLOT GRAPH EVOLUTION
	########################################################
	plt.close()
	j = 0
	graph_folder = folder_name + '/graft_graphs_' + str(num_nodes) +'/'
	try:
		os.mkdir(graph_folder)
	except:
		pass
	for t in sorted(list(grafter.graph_snapshots.keys())):
		j += 1
		G = grafter.graph_snapshots[t]
		pos = nx.shell_layout(G)
		nx.draw(G, pos)
		plt.title(str(t))
		plt.savefig(graph_folder + str(j) + '_G_graph_.png')
		plt.close()
	########################################################

	# #UNCOMMENT TO PLOT F1 SCORES EVOLUTION
	# plt.close()
	# fig, ax1 = plt.subplots()
	# for i in range(len(METHODS)):
	# 	print(METHODS[i])
	# 	ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label='F1-'+METHODS[i])
	# ax1.set_xlabel('Time')
	# ax1.set_ylabel('F1 Score')
	# ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
	# plt.title('F1 VS Time')
	# plt.savefig(folder_name + '/' + str(len(variables)) + '_F1_.png')
	# plt.close()

	#UNCOMMENT TO PLOT nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], nlls[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='NLL-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('NLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
	plt.title('NLL VS Time')
	plt.savefig(folder_name + '/' + str(len(variables)) + '_NLL_.png')
	plt.close()

	#UNCOMMENT TO PLOT nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='LOSS-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('LOSS')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
	plt.title('LOSS VS Time')
	plt.savefig(folder_name + '/' + str(len(variables)) + '_OBJ_.png')
	plt.close()

	results = {'methods': METHODS, 'nlls': nlls, 'objs': objs, 'mn_snapshots': mn_snapshots, 'time_stamps': M_time_stamps}
	RES_SHELVE.update(results)
	RES_SHELVE.close()

if __name__ == '__main__':
	main()