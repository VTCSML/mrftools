import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1
import time
from Graft import Graft
import copy
import itertools

np.set_printoptions(threshold=np.nan)

METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'blue', 'graft':'blue'}
METHOD_COLORS_i = {'structured':'r', 'naive': 'g', 'queue':'y', 'graft':'b'}

folder_num = 'monitor_ss'
num_iterations = 1

def main():

	# edge_reg = .1
	# node_reg = .5
	priority_graft_iter = 5000
	graft_iter = 5000
	# num_nodes_range = range(8, 500, 8)
	T_likelihoods = dict()
	zero_threshold = 1e-3

	training_ratio = .7
	mrf_density = .01
	edge_std = 2.5
	node_std = .0001
	state_num = 4
	l2_coeff = 0
	num_nodes_range = range(50, 100, 10)
	min_precision = .15

	# num_nodes_range = range(16, 500, 8)
	# num_nodes_range = [10]
	# num_nodes_range = range(5, 500, 10)

	edge_reg_range = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1] #np.arange(1e-5, 5e-2, 5e-5)#[.04]#[1, 5e-1, 1e-1]
	# node_reg_range = [1e-2] #np.arange(1e-3, 1e-1, 1e-3)#[.06]#[1, 9e-1, 6e-1, 3e-1, 1e-1]

	# reg_range = np.arange(0,1,0.01)

	# edge_reg_range = np.arange(50,500,10)#[.04]#[1, 5e-1, 1e-1]
	# node_reg_range = np.arange(50,500,10)

	T_likelihoods = dict()
	M_time_stamps = dict()
	suff_stats_at_70 = dict()
	print('================================= ///////////////////START//////////////// =========================================')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		mrf_density = float(1)/(num_nodes-1)

		len_data = num_nodes * 1000

		time_likelihoods = dict()
		mean_time_likelihoods = dict()
		std_time_likelihoods = dict()

		time_recall = dict()
		mean_time_recall = dict()
		std_time_recall = dict()

		time_precision = dict()
		mean_time_precision = dict()
		std_time_precision = dict()

		time_num_edges = dict()
		mean_time_num_edges = dict()
		std_time_num_edges = dict()

		max_time = 0

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
		# edge_num = len(edges) + 10
		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots, f1_scores, cumm_iters = dict(), dict(), dict(), dict(), dict(), dict()

		opt_edge_reg = edge_reg_range[0]
		opt_node_reg = 1.5 * edge_reg_range[0]
		
		for method in METHODS:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			if method == 'structured':
				max_recall = -1
				pass_loop = False
				for edge_reg in edge_reg_range:
					node_reg = 1.15 * edge_reg
					print('//////////////')
					print('reg params')
					print(edge_reg)
					print(node_reg)
					spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
					spg.on_show_metrics()
					# spg.on_verbose()

					spg.on_synthetic(precison_threshold = min_precision, start_num = 10)
					spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
					# spg.on_zero_treshold(zero_threshold=zero_threshold)
					spg.on_monitor_mn()
					t = time.time()
					# print(edges)
					learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, _, is_early_stop = spg.learn_structure(edge_num, edges=edges)
					# print(final_active_set)
					# print(recall)
					print(not is_early_stop)
					if not is_early_stop:
						print('Success')
						print(f1_score[-1])
						# f1_score = []
						# for i in range(len(precision)):
						# 	if precision[i]==0 and recall[i]==0:
						# 		f1_score.append(0)
						# 	else:
						# 		f1_score.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))

						suff_stats_at_70[num_nodes] = spg.ss_at_70

						exec_time = time.time() - t
						# print('exec_time')
						# print(exec_time)
						# print('Converged')
						# print(spg.is_converged)
						mn_snapshots[method] = spg.mn_snapshots
						time_stamps = sorted(list(spg.mn_snapshots.keys()))
						M_time_stamps[method] = time_stamps
						last_f1score = f1_score[-1]
						# for t in time_stamps:
						# 	nll = compute_likelihood_1(spg.mn_snapshots[t], len(variables), tesiterationst_data)
						# last_f1score = f1_score[-1]
						if recall[-1] > max_recall:
							print('New best')
							# print(last_f1score)
							opt_edge_reg = edge_reg
							opt_node_reg = node_reg
							max_recall = recall[-1]
							best_mn_snapshots = spg.mn_snapshots
							cumm_iter = [spg.total_iter_num[0]]
							[cumm_iter.append(cumm_iter[i-1] + spg.total_iter_num[i]) for i in range(1,len(spg.total_iter_num))]
							cumm_iters[method] = cumm_iter
							f1_scores[method] = f1_score
							sufficientstats[method] = suff_stats_list
							precisions[method] = precision
							recalls[method] = recall
							if last_f1score > .9:
								print('OPT found')
								pass_loop = True
								break

				# time_stamps = sorted(list(best_mn_snapshots.keys()))
				# M_time_stamps[method] = time_stamps
			else:
				spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
				spg.on_show_metrics()
				spg.setup_learning_parameters(edge_l1=opt_edge_reg, max_iter_graft=priority_graft_iter, node_l1=opt_node_reg)
				# spg.on_zero_treshold(zero_threshold=zero_threshold)
				spg.on_monitor_mn()
				t = time.time()
				# print(edges)
				learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, _, is_early_stop = spg.learn_structure(edge_num, edges=edges)
				f1_scores[method] = f1_score
				sufficientstats[method] = suff_stats_list
				precisions[method] = precision
				recalls[method] = recall
				cumm_iter = [spg.total_iter_num[0]]
				[cumm_iter.append(cumm_iter[i-1] + spg.total_iter_num[i]) for i in range(1,len(spg.total_iter_num))]
				cumm_iters[method] = cumm_iter

		plt.close()
		for i in range(len(METHODS)):
			print(METHODS[i])
			print(sufficientstats[METHODS[i]])
			print(recalls[METHODS[i]])
			# ax1.plot(M_time_stamps[METHODS[i]], M_accuracies[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='accuracy_' +METHODS[i], linewidth=1)
			plt.plot(recalls[METHODS[i]], sufficientstats[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=2)

		plt.xlabel('Recall')
		plt.ylabel('Suff Stats')

		plt.legend(loc='best')
		plt.title('Recall-SS')
		plt.savefig('../../../results_' + folder_num + '/' + str(len(variables)) + '_ReacllVSss_.png')
		plt.close()

		for i in range(len(METHODS)):
			print(METHODS[i])
			print(sufficientstats[METHODS[i]])
			print(recalls[METHODS[i]])
			# ax1.plot(M_time_stamps[METHODS[i]], M_accuracies[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='accuracy_' +METHODS[i], linewidth=1)
			plt.plot(sufficientstats[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=2)

		plt.xlabel('# active edges')
		plt.ylabel('Suff Stats')

		plt.legend(loc='best')
		plt.title('Edges-SS')
		plt.savefig('../../../results_' + folder_num + '/' + str(len(variables)) + '_EdgesVSss_.png')
		plt.close()


		for i in range(len(METHODS)):
			print(METHODS[i])
			print(sufficientstats[METHODS[i]])
			print(recalls[METHODS[i]])
			# ax1.plot(M_time_stamps[METHODS[i]], M_accuracies[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='accuracy_' +METHODS[i], linewidth=1)
			plt.plot(recalls[METHODS[i]], cumm_iters[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=2)

		plt.xlabel('Recall')
		plt.ylabel('cumm_iter')

		plt.legend(loc='best')
		plt.title('Recall-SearchIterations')
		plt.savefig('../../../results_' + folder_num + '/' + str(len(variables)) + '_ReacllVsIter_.png')
		plt.close()

	pernode_sufficientstats = []
	[pernode_sufficientstats.append(suff_stats_at_70[num_nodes]) for num_nodes in num_nodes_range]

	for i in range(len(METHODS)):
		plt.plot(num_nodes_range, pernode_sufficientstats, color=METHOD_COLORS[METHODS[i]], label='suff_stats_' + METHODS[i], linewidth=2)

	plt.xlabel('num_nodes')
	plt.ylabel('suff_stats')

	plt.legend(loc='best')
	plt.title('suff_stats VS num_nodes')
	plt.savefig('../../../results_' + folder_num + '/' + 'numnodes_ss_.png')
	plt.close()

if __name__ == '__main__':
	main()