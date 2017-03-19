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

METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'yellow', 'graft':'blue'}
METHOD_COLORS_i = {'structured':'r', 'naive': 'g', 'queue':'y', 'graft':'b'}

folder_name = 'compare_accuracy'
folder_num = 'l1_metrics'

def main():
	priority_graft_iter = 10000
	graft_iter = 10000
	T_likelihoods = dict()
	zero_threshold = 1e-3
	training_ratio = .7
	mrf_density = .01
	edge_std = 2.5
	node_std = .001
	state_num = 4
	l2_coeff = 0
	# num_nodes_range = range(16, 500, 8)
	num_nodes_range = [95]
	# num_nodes_range = range(35, 100, 10)
	min_precision = .1


	# edge_reg_range = [ 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2] #[1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1] #np.arange(1e-5, 5e-2, 5e-5)#[.04]#[1, 5e-1, 1e-1]
	# node_reg_range = [1e-2]

	node_reg_range = [0.0275, 0.0265]

	edge_reg_range = [0.025]

	T_likelihoods = dict()
	M_time_stamps = dict()
	print('================================= ///////////////////START//////////////// =========================================')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		total_edge_num = (num_nodes ** 2 - num_nodes) / 2
		mrf_density = min(mrf_density, float(2)/(num_nodes-1))

		len_data = 10000

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

		METHODS = ['structured']
		M_accuracies = dict()
		sorted_timestamped_mn = dict()
		edge_likelihoods = dict()
		print('======================================Simulating data...')
		# model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(len_data, num_cluster, 8, 8)
		model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
		target_vars = list(set(itertools.chain.from_iterable(edges)))

		train_data = data[: int(training_ratio * len_data)]
		test_data = data[int(training_ratio * len_data) : len_data]
		#######################
		# test_data = train_data
		#######################
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
		# edge_num = len(edges) + 20
		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots, f1_scores = dict(), dict(), dict(), dict(), dict()

		opt_edge_reg = edge_reg_range[0]
		opt_node_reg = edge_reg_range[0]
		
		for method in METHODS:
			max_f1score = 0
			pass_loop = False
			best_nll = float('inf')
			# for reg in reg_range:
			# 	edge_reg
			# 	node_reg
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			for edge_reg in edge_reg_range:
				if pass_loop:
					break
				node_reg_range = [1.25 * edge_reg, 1.2 * edge_reg, 1.175 * edge_reg, 1.15 * edge_reg, 1.125 * edge_reg, 1.1 * edge_reg , 1.05 * edge_reg, edge_reg]
				for node_reg in node_reg_range:
					print('//////')
					print(edge_reg)
					print(node_reg)
					spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
					spg.on_show_metrics()
					# spg.on_verbose()

					spg.on_synthetic(precison_threshold = min_precision)

					spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
					# spg.on_zero_treshold(zero_threshold=zero_threshold)
					spg.on_monitor_mn()
					t = time.time()
					# print(edges)
					learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, is_early_stop = spg.learn_structure(edge_num, edges=edges)
					# print(final_active_set)
					# print(recall)
					if not is_early_stop:
						# f1_score = []
						# for i in range(len(precision)):
						# 	if precision[i]==0 and recall[i]==0:
						# 		f1_score.append(0)
						# 	else:
						# 		f1_score.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
						exec_time = time.time() - t
						precisions[method] = precision
						recalls[method] = recall
						nll1 = compute_likelihood(spg.mn_snapshots[min(list(spg.mn_snapshots.keys()))], len(variables), test_data)
						nll = compute_likelihood(learned_mn, len(variables), test_data)
						mn_snapshots[method] = spg.mn_snapshots
						last_f1score = f1_score[-1]
						# last_f1score = f1_score[-1]
						if nll1 > nll and nll <= best_nll:
							best_nll = nll
							print('NEW BEST CONGRATUALITIONS WALID! YOU MADE IT! YOUR PARENTS MUST BE PROUD OF YOU! WHAT AN ACCOMPLISHMENT!! THIS IS THE FUTURE OF MACHINE LEARNING...')
							print(best_nll)
							opt_edge_reg = edge_reg
							opt_node_reg = node_reg
							max_f1score = last_f1score
							best_mn_snapshots = spg.mn_snapshots
							f1_scores[method] = f1_score
							if last_f1score > .95:
								pass_loop = True
								break

			time_stamps = sorted(list(best_mn_snapshots.keys()))
			M_time_stamps[method] = time_stamps

			method_likelihoods = []
			accuracies = []
			print('OPT PARAMS')
			print(opt_edge_reg)
			print(opt_node_reg)
			for t in time_stamps:
				# nll = compute_likelihood(best_mn_snapshots[t], len(variables), test_data)
				nll = compute_likelihood(best_mn_snapshots[t], len(variables), test_data)
				method_likelihoods.append(nll)
		M_accuracies[method] = accuracies
		T_likelihoods[method] = method_likelihoods


		print('Real model')
		real_model_nll = compute_likelihood(model, len(variables), test_data)
		# print('Real model')
		# real_model_nll = compute_likelihood_1(model, len(variables), test_data)


		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
		grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
		grafter.on_show_metrics()
		# grafter.on_verbose()
		grafter.setup_learning_parameters(edge_l1 = opt_edge_reg, max_iter_graft=graft_iter, node_l1=opt_node_reg)
		grafter.on_monitor_mn()
		t = time.time()
		# grafter.on_zero_treshold(zero_threshold=zero_threshold)
		# print(edges)
		learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, is_early_stop = grafter.learn_structure(edge_num, edges=edges)
		print(final_active_set)
		print(recall)
		f1_scores['graft'] = f1_score
		exec_time = time.time() - t
		precisions['graft'] = precision
		recalls['graft'] = recall
		# print('exec_time')
		# print(exec_time)
		METHODS.append('graft')
		mn_snapshots['graft'] = grafter.mn_snapshots
		time_stamps = sorted(list(grafter.mn_snapshots.keys()))
		M_time_stamps['graft'] = time_stamps
		method_likelihoods = []
		accuracies = []
		for t in time_stamps:
			# nll = compute_likelihood(grafter.mn_snapshots[t], len(variables), test_data)
			nll = compute_likelihood(grafter.mn_snapshots[t], len(variables), test_data)
			method_likelihoods.append(nll)
		T_likelihoods['graft'] = method_likelihoods
		M_accuracies['graft'] = accuracies



		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		for i in range(len(METHODS)):
			# ax1.plot(M_time_stamps[METHODS[i]], M_accuracies[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='accuracy_' +METHODS[i], linewidth=1)
			ax1.plot(M_time_stamps[METHODS[i]], T_likelihoods[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='nll_' +METHODS[i], linewidth=1)
			ax2.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', label='f1-score_'+METHODS[i])

		# ax1.plot(M_time_stamps[METHODS[i]], real_model_nll * np.ones(len(M_time_stamps['graft'])), color='green', label='nll_real_model', linewidth=1)
		ax1.set_xlabel('time')
		ax1.set_ylabel('nll')
		ax2.set_ylabel('f1_score')

		ax1.legend(loc='best')
		ax2.legend(loc=4, fancybox=True, framealpha=0.5)
		plt.title('nll-precision')
		plt.xlabel('iterations')
		plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_nll_.png')
		plt.close()



if __name__ == '__main__':
	main()