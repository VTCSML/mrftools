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
parser.add_argument('--nodes_num', dest='num_nodes', required=True)
parser.add_argument('--edge_std', dest='edge_std', default=2.5)
parser.add_argument('--node_std', dest='node_std', default=.001)
parser.add_argument('--state_num', dest='state_num', default=5)
parser.add_argument('--len_data', dest='len_data', default=50000)
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
	min_precision = .25
	edge_reg_range = [ 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3,\
	2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,\
	 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1]

	print('================================= ///////////////////START//////////////// =========================================')

	################################################################### DATA PREPROCESSING GOES HERE --------->
	num_nodes = int(args.num_nodes)
	training_ratio = .9
	edge_std = float(args.edge_std)
	node_std = float(args.node_std)
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

	RES_SHELVE = shelve.open('results_' + str(num_nodes))

	params = {'num_nodes':num_nodes, 'edge_std':edge_std, 'node_std':node_std, 'state_num':state_num, 'len_data':len_data, 'mrf_density':mrf_density}

	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)

	ss_test = dict()

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

	edge_num = float('inf') # MAX NUM EDGES TO GRAFT
	recalls, precisions, sufficientstats, mn_snapshots, f1_scores, objs, test_nlls, train_nlls, M_time_stamps = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()

	for method in METHODS:
		if method == 'structured':
			opt_reached = False
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			j = 0
			best_params = (0,0)
			best_nll = float('inf')
			best_objec = float('inf')
			best_f1 = 0
			first_opt_found = False
			for edge_reg in edge_reg_range:
				pq = copy.deepcopy(original_pq)

				node_reg = 1.15 * edge_reg

				print('======PARAMS')
				print(edge_reg)
				print(node_reg)
				j += 1
				spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict=pq, ss_test=ss_test)
				spg.on_show_metrics()
				# spg.on_verbose()
				spg.on_synthetic(precison_threshold = min_precision, start_num = 3) ## EARLY STOP GRAFTING IF 4 EDGES ARE ADDED AND PRECISION < min_precision
				spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
				spg.on_monitor_mn(is_real_loss=is_real_loss)
				t = time.time()
				learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
				print(not is_early_stop)
				if not is_early_stop and recall[-1] == 0: # L1 coeff too big
					break
				if not is_early_stop and f1_score[-1] > best_f1:
					print('NEW OPT EDGE L1')
					opt_node_reg = node_reg
					opt_edge_reg = edge_reg
					first_opt_found = True
					best_f1 = f1_score[-1]
					best_mn_snapshots = copy.deepcopy(spg.mn_snapshots)
					time_stamps = sorted(list(best_mn_snapshots.keys()))
					# best_graph_snapshots = copy.deepcopy(spg.graph_snapshots)

					f1_scores[method] = f1_score
					objs[method] = objec
					mn_snapshots[method] = best_mn_snapshots
					M_time_stamps[method] = time_stamps

					if f1_score[-1] >=.8: # good enough
						print('NEW OPT NODE L1')
						print(opt_node_reg)
						break

			# #UNCOMMENT IF PRINT EVOLUTION OF METRICS WITH EDGE L1
			######################################################
			# print('plotting')
			# plt.close()
			# fig, ax1 = plt.subplots()
			# ax2 = ax1.twinx()
			# ax1.plot(edge_reg_range, _likelihoods, color='red', label='nll', linewidth=2)
			# ax2.plot(edge_reg_range, _recalls, color='green', linewidth=2, linestyle=':', label='recall')
			# ax2.plot(edge_reg_range, _precisions, color='blue', linewidth=2, linestyle=':', label='precision')
			# ax2.set_ylim([-.1,1.1])
			# ax1.set_ylabel('nll')
			# ax2.set_ylabel('precison/recall')
			# ax1.set_xlabel('l1-coeff')
			# ax1.legend(loc='best')
			# ax2.legend(loc=4, fancybox=True, framealpha=0.5)
			# ax1.set_xscale("log", nonposx='clip')
			# ax2.set_xscale("log", nonposx='clip')
			# plt.title('nll-precision' + '_best:' + str(best_params[0]) + ',' +str(best_params[1]))
			# plt.savefig('../../../results_' + folder_num + '/' + str(len(variables)) + '_best:' + str(best_params[0]) + ',' +str(best_params[1]) +'.png')
			# plt.close()
			######################################################

			if not first_opt_found:
				raise ValueError('OPT NOT FOUND')

			# if best_f1 >= .8: # NO NEED TO FIND BETTER NODE L1
			# 	print('OPT PARAMS')
			# 	print(opt_edge_reg)
			# 	print(opt_node_reg)
			# 	opt_reached =True

			# if not opt_reached:

			# 	print('>>>>>>>>Getting best node reg...')
			# 	pq = copy.deepcopy(original_pq)
			# 	node_reg_range = [2 * opt_edge_reg, 1.9 * opt_edge_reg, 1.8 * opt_edge_reg, 1.7 * opt_edge_reg, 1.6 * opt_edge_reg, 1.5 * opt_edge_reg,1.4 * opt_edge_reg, 1.3 * opt_edge_reg, 1.2 * opt_edge_reg, 1.1 * opt_edge_reg, 1 * opt_edge_reg]
			# 	for node_reg in node_reg_range:
			# 		print('//////')
			# 		print(opt_edge_reg)
			# 		print(node_reg)
			# 		spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict=pq, ss_test=ss_test)
			# 		spg.on_show_metrics()
			# 		# spg.on_verbose()
			# 		spg.on_synthetic(precison_threshold = min_precision, start_num = 2)
			# 		spg.setup_learning_parameters(edge_l1=opt_edge_reg, max_iter_graft=graft_iter, node_l1=node_reg)
			# 		spg.on_monitor_mn(is_real_loss=is_real_loss)
			# 		# spg.on_plot_queue('../../../') # Uncomment to plot pq dict COMES WITH COMPUTATION OVERLOAD
			# 		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
			# 		if not is_early_stop:
			# 			if f1_score[-1] > best_f1:
			# 				print('NEW OPT NODE L1!')
			# 				opt_node_reg = node_reg
			# 				best_f1 = f1_score[-1]
			# 				best_mn_snapshots = copy.deepcopy(spg.mn_snapshots)
			# 				f1_scores[method] = f1_score
			# 				objs[method] = objec
			# 				time_stamps = sorted(list(best_mn_snapshots.keys()))
			# 				M_time_stamps[method] = time_stamps
			# 				mn_snapshots[method] = best_mn_snapshots
			# 				print('OPT PARAMS')
			# 				print(opt_edge_reg)
			# 				print(opt_node_reg)

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

		else:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			print('OPT PARAMS')
			print(opt_edge_reg)
			print(opt_node_reg)
			pq = copy.deepcopy(original_pq)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, pq_dict=pq, ss_test=ss_test)
			spg.on_show_metrics()
			spg.setup_learning_parameters(edge_l1=opt_edge_reg, max_iter_graft=priority_graft_iter, node_l1=opt_node_reg)
			spg.on_monitor_mn(is_real_loss=is_real_loss)
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
			time_stamps = sorted(list(spg.mn_snapshots.keys()))
			mn_snapshots[method] = spg.mn_snapshots
			M_time_stamps[method] = time_stamps
			f1_scores[method] = f1_score
			objs[method] = objec

	print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
	grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
	grafter.on_show_metrics()
	# grafter.on_verbose()
	grafter.setup_learning_parameters(edge_l1 = opt_edge_reg, max_iter_graft=graft_iter, node_l1=opt_node_reg)
	grafter.on_monitor_mn(is_real_loss=is_real_loss)
	t = time.time()
	learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
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
			test_nll = compute_likelihood(mn_snaps[t], len(variables), test_data)
			train_nll = compute_likelihood(mn_snaps[t], len(variables), train_data)
			test_nll_list.append(test_nll)
			train_nll_list.append(train_nll)
		test_nlls[method] = test_nll_list
		train_nlls[method] = train_nll_list
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

	opt_params = (opt_node_reg, opt_edge_reg)
	params['opt_params'] = opt_params
	print('opt_params')
	print(opt_params)

	results = {'methods':METHODS, 'train_nlls':train_nlls, 'test_nlls':test_nlls, 'f1':f1_scores, 'objs':objs, 'params':params}
	RES_SHELVE.update(results)
	RES_SHELVE.close()



	#UNCOMMENT TO PLOT F1 SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label='F1-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('F1 Score')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
	plt.title('F1 VS Time')
	plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_F1_.png')
	plt.close()

	#UNCOMMENT TO PLOT test nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], test_nlls[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='TestNLL-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Test NLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
	plt.title('Test NLL VS Time')
	plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_NLL_.png')
	plt.close()


	#UNCOMMENT TO PLOT train nll SCORES EVOLUTION
	plt.close()
	fig, ax1 = plt.subplots()
	for i in range(len(METHODS)):
		print(METHODS[i])
		ax1.plot(M_time_stamps[METHODS[i]], train_nlls[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='TrainNLL-'+METHODS[i])
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Train NLL')
	ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
	plt.title('Train NLL VS Time')
	plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_NLL_.png')
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
	plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_OBJ_.png')
	plt.close()

if __name__ == '__main__':
	main()

