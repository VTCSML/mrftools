import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood, compute_accuracy
import time
from Graft import Graft
from read_ratings import read_ratings_from_batch_files

FILES_LIST = ['../../../ratings/jester-data-1.xls', '../../../ratings/jester-data-2.xls', '../../../ratings/jester-data-3.xls'] 

def main():
	edge_reg_range = [0.06] #np.arange(0.01,0.25,0.05)
	node_reg_range = [0.21] #np.arange(0.01,0.25,0.05)

	graft_iter = 2500
	zero_threshold = 1e-3
	# edge_num = float('inf')
	edge_num = 150
	num_states = dict()
	variables = range(1,101)
	num_attributes = len(variables)
	for i in range(1,101):
		num_states[i] = 5
	max_num_states = 5
	print('FETCHING DATA!')
	data = read_ratings_from_batch_files(FILES_LIST, 50)
	len_data = len(data)
	train_data = data[: int(.8 * len_data)]
	test_data = data[int(.8 * len_data) : len_data]
	opt_params = (0, 0)
	max_accuracy = 0
	max_predicted_nodes = 0
	for edge_reg in edge_reg_range:
		for node_reg in node_reg_range:
			skip_iter = False
			print('PARAMS')
			print(edge_reg)
			print(node_reg)
			# print('LEARNING MRF')
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, None, 'structured')
			spg.on_verbose()
			spg.on_monitor_mn()
			spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=graft_iter, node_l1=node_reg)
			# spg.on_zero_treshold(zero_threshold=zero_threshold)
			# spg.on_show_metrics()
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, _ = spg.learn_structure(edge_num)
			# print('TESTING')
			# likelihood = compute_likelihood(learned_mn, variables, test_data)
			total_accuracy = 0

			predicted_nodes = 0

			for item in range(1,101):
				print(item)
				curr_neighbors = list(learned_mn.get_neighbors(item)) 
				# print(curr_neighbors)
				# print(len(curr_neighbors) > 0)
				if curr_neighbors:
					predicted_nodes += 1
					accuracy, true_states, predicted_states = compute_accuracy(learned_mn, variables, test_data, item, 5)
					# print('HERE')
					# print('Accuracy')
					print(accuracy)
					total_accuracy += accuracy
					if (predicted_nodes > 2) and (float(total_accuracy) / float(predicted_nodes) < .5):
						# break
						pass

			mean_accuracy = float(total_accuracy) / float(predicted_nodes)

			print('Mean accuracy')
			print(mean_accuracy)
			print('predicted_nodes')
			print(predicted_nodes)

			if max_accuracy < mean_accuracy:
				max_accuracy = mean_accuracy
				opt_params = (node_reg, edge_reg)
				max_predicted_nodes = predicted_nodes


	# print('Real states')
	# print(true_states)

	# print('Predicted states')
	# print(predicted_states)

	print('max_accuracy')
	print(max_accuracy)
	print('opt_params')
	print(opt_params)
	print('predicted_nodes')
	print(max_predicted_nodes)

if __name__ == '__main__':
	main()