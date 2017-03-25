import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1, get_all_ss
import time
from Graft import Graft
import copy
import itertools
import networkx as nx
import os

np.set_printoptions(threshold=np.nan)

METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'black', 'graft':'blue'}
METHOD_COLORS_i = {'structured':'r', 'naive': 'g', 'queue':'y', 'graft':'b'}

folder_name = 'compare_loss_queue'
folder_num = 'l1_metrics'
num_iterations = 1

def main():
	priority_graft_iter = 5000
	graft_iter = 5000
	T_likelihoods = dict()
	# zero_threshold = 1e-3
	training_ratio = .7
	edge_std = 3
	node_std = .0001
	state_num = 5
	l2_coeff = 0
	num_nodes_range = range(10, 500, 10)
	min_precision = .5

	edge_reg_range = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1]
	# opt_edge_reg = .1
	# opt_node_reg = .1

	T_likelihoods = dict()
	M_time_stamps = dict()
	print('================================= ///////////////////START//////////////// =========================================')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		total_edge_num = (num_nodes ** 2 - num_nodes) / 2
		# mrf_density = min(mrf_density, float(2)/(num_nodes-1))
		mrf_density = float(1)/((num_nodes - 1))
		len_data = 25000
		# METHODS = ['naive', 'structured', 'queue']
		METHODS = ['structured', 'queue']
		M_accuracies = dict()
		sorted_timestamped_mn = dict()
		edge_likelihoods = dict()
		print('======================================Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
		target_vars = list(set(itertools.chain.from_iterable(edges)))
		train_data = data[: int(training_ratio * len_data)]
		test_data = data[int(training_ratio * len_data) : len_data]
		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		shuffle(list_order)


		ss_test = dict()

		# print('ss')
		# for var1 in variables:
		# 	for var2 in variables:
		# 		if var1 < var2:
		# 			edge = (var1, var2)
		# 			edge_sufficient_stats = 1
		# 			table = np.ones((num_states[edge[0]], num_states[edge[1]]))
		# 			# table[states[edge[0]], states[edge[1]]] = 1
		# 			tmp = np.asarray(table.reshape((-1, 1)))
		# 			edge_sufficient_stats += tmp
		# 			ss_test[edge] = edge_sufficient_stats
		# print('ss end')



		ss_test = get_all_ss(variables, num_states, train_data)

		print(variables)
		print(num_states)
		print(max_num_states)

		print('NUM VARIABLES')
		print(len(variables))

		print('NUM EDGES')
		print(len(edges))

		print('EDGES')
		print(edges)
		edge_num = float('inf')
		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots, f1_scores, objs = dict(), dict(), dict(), dict(), dict(), dict()
		for method in METHODS:
			if method == 'structured':
				opt_reached = False
				_likelihoods = []
				_recalls = []
				_precisions= []
				print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
				j = 0
				best_params = (0,0)
				best_nll = float('inf')
				best_objec = float('inf')
				best_precision = 0
				best_f1 = 0
				for edge_reg in edge_reg_range:
					node_reg = 1.1 * edge_reg
					print('======PARAMS')
					print(edge_reg)
					print(node_reg)
					j += 1
					spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, ss_test=ss_test)
					spg.on_show_metrics()
					# spg.on_verbose()

					spg.on_synthetic(precison_threshold = min_precision, start_num = 2)

					spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
					spg.on_monitor_mn()
					t = time.time()
					learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
					print(is_early_stop)
					print(recall)
					print(precision)
					print(spg.active_set)
					_recalls.append(recall[-1])
					_precisions.append(precision[-1])
					# nll = compute_likelihood_1(learned_mn, len(variables), test_data)
					# nll = compute_likelihood(learned_mn, len(variables), test_data)
					# _likelihoods.append(nll)
					# if not is_early_stop and nll < best_nll:
					if not is_early_stop and recall[-1] == 0:
						break
					if not is_early_stop and f1_score[-1] > best_f1 + .05 and objec[-1] < best_objec:
						best_objec = objec[-1]
						best_f1 = f1_score[-1]
						best_precision = precision[-1] 
						print('NEW OPT FOUND')
						# best_nll = nll
						best_mn_snapshots = copy.deepcopy(spg.mn_snapshots)
						best_graph_snapshots = copy.deepcopy(spg.graph_snapshots)
						f1_scores[method] = f1_score
						objs[method] = objec
						opt_node_reg = node_reg
						opt_edge_reg = edge_reg
						best_params = (node_reg, edge_reg)
						time_stamps = sorted(list(best_mn_snapshots.keys()))
						M_time_stamps[method] = time_stamps
						print(len(time_stamps))
						print(len(objec))
						print('OPT PARAMS')
						print(opt_edge_reg)
						print(opt_node_reg)
						print(best_params)
						if f1_score[-1] >=.8:
							print('OPT reached')
							break
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


				if best_f1 >= .8:
					print('OPT PARAMS')
					print(opt_edge_reg)
					print(opt_node_reg)
					opt_reached =True

				if not opt_reached:

					print('#########################################Getting best node reg#################################################')
					edge_reg = best_params[1]
					best_precision = 0
					best_f1 = 0
					pass_loop = False
					best_nll = float('inf')
					node_reg_range = [1.5 * edge_reg, 1.25 * edge_reg, 1.2 * edge_reg, 1.175 * edge_reg, 1.15 * edge_reg, 1.125 * edge_reg, 1.1 * edge_reg , 1.05 * edge_reg, 1 * edge_reg]
					for node_reg in node_reg_range:
						print('//////')
						print(edge_reg)
						print(node_reg)
						spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, ss_test=ss_test)
						spg.on_show_metrics()
						# spg.on_verbose()
						spg.on_synthetic(precison_threshold = min_precision, start_num = 2)
						spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
						spg.on_monitor_mn(is_real_loss=True)
						# spg.on_plot_queue('../../../')
						t = time.time()
						learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
						if not is_early_stop:
							exec_time = time.time() - t
							precisions[method] = precision
							recalls[method] = recall
							# nll1 = compute_likelihood(spg.mn_snapshots[min(list(spg.mn_snapshots.keys()))], len(variables), test_data)
							# nll = compute_likelihood(learned_mn, len(variables), test_data)
							mn_snapshots[method] = spg.mn_snapshots
							# if nll1 > nll and nll <= best_nll:
							if  f1_score[-1] > best_f1:
								# best_nll = nll
								print('NEW BEST!')
								print(best_nll)
								opt_edge_reg = edge_reg
								opt_node_reg = node_reg
								best_precision = precision[-1]
								best_f1 = f1_score[-1]
								best_mn_snapshots = copy.deepcopy(spg.mn_snapshots)
								best_graph_snapshots = copy.deepcopy(spg.graph_snapshots)
								f1_scores[method] = f1_score
								objs[method] = objec
								time_stamps = sorted(list(best_mn_snapshots.keys()))
								M_time_stamps[method] = time_stamps
								method_likelihoods = []
								print(len(time_stamps))
								print(len(objec))
								print('OPT PARAMS')
								print(opt_edge_reg)
								print(opt_node_reg)


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

			else:
				print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
				spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method, ss_test=ss_test)
				spg.on_show_metrics()
				spg.setup_learning_parameters(edge_l1=opt_edge_reg, max_iter_graft=priority_graft_iter, node_l1=opt_node_reg)
				spg.on_monitor_mn(is_real_loss=True)
				t = time.time()
				learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = spg.learn_structure(edge_num, edges=edges)
				exec_time = time.time() - t
				precisions[method] = precision
				recalls[method] = recall
				time_stamps = sorted(list(spg.mn_snapshots.keys()))
				M_time_stamps[method] = time_stamps
				f1_scores[method] = f1_score
				objs[method] = objec
				print(len(time_stamps))
				print(len(objec))



		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
		grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
		grafter.on_show_metrics()
		# grafter.on_verbose()
		grafter.setup_learning_parameters(edge_l1 = opt_edge_reg, max_iter_graft=graft_iter, node_l1=opt_node_reg)
		grafter.on_monitor_mn()
		t = time.time()
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
		print(final_active_set)
		print(recall)
		objs['graft'] = objec
		f1_scores['graft'] = f1_score
		exec_time = time.time() - t
		precisions['graft'] = precision
		recalls['graft'] = recall
		METHODS.append('graft')
		mn_snapshots['graft'] = grafter.mn_snapshots
		time_stamps = sorted(list(grafter.mn_snapshots.keys()))
		M_time_stamps['graft'] = time_stamps
		method_likelihoods = []
		print(len(time_stamps))
		print(len(objec))


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



		# accuracies = []

		# for t in time_stamps:
		# 	nll = compute_likelihood(grafter.mn_snapshots[t], len(variables), test_data)
		# 	method_likelihoods.append(nll)
		# T_likelihoods['graft'] = method_likelihoods

		# M_accuracies['graft'] = accuracies

		# plt.close()
		# fig, ax1 = plt.subplots()
		# ax1.plot(M_time_stamps['structured'], objs['structured'], color=METHOD_COLORS['structured'], label='nll_structured', linewidth=1)
		# ax2.plot(M_time_stamps['structured'], f1_scores['structured'], METHOD_COLORS['structured'], linewidth=1, linestyle=':', label='f1-score_'+ 'structured')
		# ax1.set_xlabel('time')
		# ax1.set_ylabel('loss')
		# ax2.set_ylabel('f1_score')
		# ax1.legend(loc='best')
		# ax2.legend(loc=4, fancybox=True, framealpha=0.5)
		# plt.title('loss-F1')
		# plt.xlabel('iterations')
		# plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_nll_.png')
		# plt.close()


		# fig, ax1 = plt.subplots()
		# ax2 = ax1.twinx()
		# for i in range(len(METHODS)):
		# 	ax1.plot(M_time_stamps[METHODS[i]], T_likelihoods[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='nll_' +METHODS[i], linewidth=1)
		# 	ax2.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', label='f1-score_'+METHODS[i])
		# ax1.set_xlabel('time')
		# ax1.set_ylabel('nll')
		# ax2.set_ylabel('f1_score')
		# ax1.legend(loc='best')
		# ax2.legend(loc=4, fancybox=True, framealpha=0.5)
		# plt.title('nll-precision')
		# plt.xlabel('iterations')
		# plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_nll_.png')
		# plt.close()

		# plt.close()
		# fig, ax1 = plt.subplots()
		# ax2 = ax1.twinx()
		# for i in range(len(METHODS)):
		# 	print(METHODS[i])
		# 	ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='Loss-' + METHODS[i], linewidth=1)
		# 	ax2.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', marker='o', label='F1-'+METHODS[i])
		# ax1.set_xlabel('Time')
		# ax1.set_ylabel('Normalized Loss')
		# ax2.set_ylabel('F1 Score')
		# ax2.legend(loc=4, fancybox=True, framealpha=0.5)
		# ax1.legend(loc='best', framealpha=0.5)
		# plt.title('Loss-F1')
		# plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_loss_.png')
		# plt.close()


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


		plt.close()
		fig, ax1 = plt.subplots()
		for i in range(len(METHODS)):
			print(METHODS[i])
			ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=2, label='Loss-'+METHODS[i])
		ax1.set_xlabel('Time')
		ax1.set_ylabel('Loss')
		ax1.legend(loc='best', framealpha=0.5, fancybox=True,)
		plt.title('Loss VS Time')
		plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_Loss_.png')
		plt.close()

if __name__ == '__main__':
	main()