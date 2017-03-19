import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data, convert_to_binary_data
from random import shuffle
from scipy import signal as sg
from grafting_util import compute_likelihood
import time
from Graft import Graft


def main():
	edge_reg = .1
	feature_reg = 1
	node_reg = 0
	len_data = 1000
	graft_iter = 1000
	num_nodes_range = range(8, 500, 8)
	T_likelihoods = dict()
	training_ratio = .8
	zero_threshold = 1e-2
	edge_num = float('inf')
	print('================================= ///////////////////START//////////////// ========================================= ')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		max_time = 0

		for iteration in range(1):

			print('======================================Simulating data...')
			# model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(len_data, num_cluster, 8, 8)
			model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, 10, .1)
			binary_data, binary_variables, binary_to_orginial_hash, binary_max_num_states, binary_num_states = convert_to_binary_data(variables, num_states, data)
			train_data = data[: int(training_ratio * len_data)]
			test_data = data[int(training_ratio * len_data) : len_data]

			binary_train_data = binary_data[: int(training_ratio * len_data)]
			binary_test_data = binary_data[int(training_ratio * len_data) : len_data]

			list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)

			print(variables)
			print(num_states)
			print(max_num_states)

			print('NUM VARIABLES')
			print(len(variables))

			print('NUM EDGES')
			print(len(edges))

			print('EDGES')
			print(edges)

			mn_snapshots, time_stamps, M_time_stamps, T_likelihoods = dict(), dict(), dict(), dict()

			print('>>>>>>>>>>>>>>>>>>>>>METHOD: Edge Graft' )
			grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
			# grafter.on_show_metrics()
			# grafter.on_verbose()
			grafter.setup_learning_parameters(edge_l1 = edge_reg, node_l1= node_reg, max_iter_graft=graft_iter)
			grafter.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure( edges=edges)
			# grafter.on_zero_treshold(zero_threshold=zero_threshold)
			exec_time = time.time() - t
			# precisions['graft'] = precision
			# recalls['graft'] = recall

			mn_snapshots['edge_graft'] = grafter.mn_snapshots
			time_stamps = sorted(list(grafter.mn_snapshots.keys()))
			M_time_stamps['edge_graft'] = time_stamps

			for t in time_stamps:
				# nll = compute_likelihood(grafter.mn_snapshots[t], len(variables), test_data)
				nll = compute_likelihood_1(grafter.mn_snapshots[t], len(variables), test_data)
				method_likelihoods.append(nll)

			T_likelihoods['edge_graft'] = method_likelihoods


			# print('exec_time')
			# print(exec_time)

			# print('Final active space')
			# print(final_active_set)
			# mn_snapshots['graft'] = grafter.mn_snapshots

			# likelihood = compute_likelihood(learned_mn, variables, test_data)

			# print('likelihood')
			# print(likelihood)

			# print('Recall')
			# print(recall)

			# print('Precision')
			# print(precision)


			print('>>>>>>>>>>>>>>>>>>>>>METHOD: Feature Graft' )
			list_order = range(0,(len(binary_variables) ** 2 - len(binary_variables)) / 2, 1)

			binary_grafter = Graft(binary_variables, binary_num_states, binary_max_num_states, binary_train_data, list_order)
			# grafter.on_verbose()
			binary_grafter.setup_learning_parameters(edge_reg, max_iter_graft=graft_iter)
			binary_grafter.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision = binary_grafter.learn_structure( edges=edges)
			# grafter.on_zero_treshold(zero_threshold=zero_threshold)
			exec_time = time.time() - t
			# precisions['graft'] = precision
			# recalls['graft'] = recall
			print('exec_time')
			print(exec_time)
			print('Final active space')
			print(final_active_set)

			mn_snapshots['feature_graft'] = grafter.mn_snapshots
			time_stamps = sorted(list(grafter.mn_snapshots.keys()))
			M_time_stamps['feature_graft'] = time_stamps

			for t in time_stamps:
				# nll = compute_likelihood(grafter.mn_snapshots[t], len(variables), test_data)
				nll = compute_likelihood_1(grafter.mn_snapshots[t], len(variables), test_data)
				method_likelihoods.append(nll)

			T_likelihoods['feature_graft'] = method_likelihoods


			# likelihood = compute_likelihood(learned_mn, binary_variables, binary_test_data)

			# print('likelihood')
			# print(likelihood)

			# print('Recall')
			# print(recall)

			# print('Precision')
			# print(precision)


		fig, ax1 = plt.subplots()
		for i in range(len(METHODS)):
			# ax1.plot(M_time_stamps[METHODS[i]], M_accuracies[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label='accuracy_' +METHODS[i], linewidth=1)
			ax1.plot(M_time_stamps['edge_graft'], T_likelihoods['edge_graft'], color='green', label='edge_graft' , linewidth=1)
			ax1.plot(M_time_stamps['feature_graft'], T_likelihoods['feature_graft'], color='red', label='feature_graft' , linewidth=1)

			# ax2.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], METHOD_COLORS[METHODS[i]], linewidth=1, linestyle=':', label='f1-score_'+METHODS[i])

		# ax1.plot(M_time_stamps[METHODS[i]], real_model_nll * np.ones(len(M_time_stamps['graft'])), color='green', label='nll_real_model', linewidth=1)
		ax1.set_xlabel('time')
		ax1.set_ylabel('nll')
		# ax2.set_ylabel('f1_score')

		ax1.legend(loc='best')
		plt.title('nll')
		plt.xlabel('iterations')
		plt.savefig('../../../results_' + folder_num + '/' + str(len(variables)) + '_nll_.png')
		plt.close()


if __name__ == '__main__':
	main()