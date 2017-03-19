import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from L1_learner import L1_learner


def main():
	l1_coeff, l2_coeff = 1, 0
	edge_reg = 1
	node_reg = 1
	len_data = 10000
	graft_iter = 10000
	num_nodes_range = range(8, 500, 8)
	T_likelihoods = dict()
	zero_threshold = 1e-2
	print('================================= ///////////////////START//////////////// ========================================= ')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		max_time = 0


		for iteration in range(1):

			print('======================================Simulating data...')
			# model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(len_data, num_cluster, 8, 8)
			model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, 10, .1)
			train_data = data[: int(.9 * len_data)]
			test_data = data[int(.9 * len_data) : len_data]

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

			num_attributes = len(variables)

			print('>>>>>>>>>>>>>>>>>>>>>METHOD: Full L1' )

			l1_learner = L1_learner(variables, num_states, max_num_states, train_data, list_order)
			l1_learner.on_show_metrics()
			# grafter.on_verbose()
			l1_learner.setup_learning_parameters(l1_coeff=l1_coeff, max_iter_graft=graft_iter)

			l1_learner.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations, is_early_stop = l1_learner.learn_structure(edge_num, edges=edges)

			# print('final_active_set')
			# print(final_active_set)

			print('Recall')
			print(recall)

			print('Precision')
			print(precision)


			print('>>>>>>>>>>>>>>>>>>>>>METHOD: Group L1' )

			l1_learner = L1_learner(variables, num_states, max_num_states, train_data, list_order)
			l1_learner.on_show_metrics()
			# grafter.on_verbose()
			l1_learner.setup_learning_parameters(edge_l1=edge_reg, node_l1=node_reg, max_iter_graft=graft_iter)
			l1_learner.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations, is_early_stop = l1_learner.learn_structure(edges=edges)

			# print('final_active_set')
			# print(final_active_set)

			print('Recall')
			print(recall)

			print('Precision')
			print(precision)


if __name__ == '__main__':
	main()