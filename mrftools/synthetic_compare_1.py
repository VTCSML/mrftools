import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood

METHODS = ['naive', 'structured', 'queue']
METHOD_COLORS = {'structured':'-r', 'naive': '-b', 'queue':'-g'}

def main():
	edge_reg = .075
	num_cluster_range = range(3, 15, 1)
	print('================================= ///////////////////START//////////////// ========================================= ')
	for num_cluster in num_cluster_range:
		print('Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(5000, num_cluster, 6, 10)
		train_data = data[: len(data) - 201]
		test_data = data[len(data) - 200 : len(data) - 1]

		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		shuffle(list_order)

		print('EDGES')
		print(edges)
		edge_num = len(edges) + 7
		num_attributes = len(variables)
		for method in METHODS:
			print('METHOD: ' + method)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
			spg.on_show_metrics()
			# spg.on_verbose()
			spg.on_plot_queue('../../../DataDump/pq_plot')
			spg.setup_learning_parameters(edge_reg)
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges=edges)
			curr_likelihood = compute_likelihood(learned_mn, num_attributes, test_data)
			plt.figure(0)
			plt.plot(recall, METHOD_COLORS[method], label=method, linewidth=5)
			plt.figure(1)
			plt.plot(precision, METHOD_COLORS[method], label=method, linewidth=5)
			plt.figure(2)
			plt.plot(suff_stats_list, METHOD_COLORS[method], label=method, linewidth=5)
		plt.figure(0)
		plt.legend(loc='best')
		plt.title('Recall')
		plt.savefig('../../../DataDump/recall_precision_1/Recall_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()
		plt.figure(1)
		plt.legend(loc='best')
		plt.title('Precision')
		plt.savefig('../../../DataDump/recall_precision_1/Precision_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()
		plt.figure(2)
		plt.legend(loc='best')
		plt.title('Sufficient stats')
		plt.savefig('../../../DataDump/recall_precision_1/Sufficient_stats' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()

if __name__ == '__main__':
	main()