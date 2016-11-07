import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood
import time

METHODS = ['naive', 'structured', 'queue']
METHOD_COLORS = {'structured':'-r', 'naive': '-b', 'queue':'-g'}

def main():
	edge_reg = .1
	num_cluster_range = range(3, 15, 1)
	print('================================= ///////////////////START//////////////// ========================================= ')
	for num_cluster in num_cluster_range:
		print('Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(20000, num_cluster, 6, 7)
		train_data = data[: len(data) - 201]
		test_data = data[len(data) - 200 : len(data) - 1]
		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		shuffle(list_order)

		print('EDGES')
		print(edges)
		edge_num = len(edges)
		num_attributes = len(variables)
		recalls, precisions, sufficientstats = list(), list(), list()
		for method in METHODS:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
			spg.on_show_metrics()
			# spg.on_verbose()
			spg.on_plot_queue('../../../DataDump/pq_plot')
			spg.setup_learning_parameters(edge_reg, max_iter_graft=50)
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges=edges)
			exec_time = time.time() - t
			print('exec_time')
			print(exec_time)
			curr_likelihood = compute_likelihood(learned_mn, num_attributes, test_data)
			recalls.append(recall)
			precisions.append(precision)
			sufficientstats.append(suff_stats_list)

		for i in range(len(METHODS)):
			plt.plot(recalls[i], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)  
		plt.legend(loc='best')
		plt.title('Recall')
		plt.savefig('../../../DataDump/recall_precision_1/Recall_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()

		for i in range(len(METHODS)):
			plt.plot(precisions[i], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)  
		plt.legend(loc='best')
		plt.title('Precision')
		plt.savefig('../../../DataDump/recall_precision_1/Precision_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()

		for i in range(len(METHODS)):
			plt.plot(sufficientstats[i], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)  
		plt.legend(loc='best')
		plt.title('Sufficient Stats')
		plt.savefig('../../../DataDump/recall_precision_1/SufficientStats_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()



		# 	plt.figure(1)
		# 	plt.plot(recall, METHOD_COLORS[method], label=method, linewidth=5)
		# 	plt.figure(0)
		# 	plt.plot(precision, METHOD_COLORS[method], label=method, linewidth=5)
		# 	plt.figure(2)
		# 	plt.plot(suff_stats_list, METHOD_COLORS[method], label=method, linewidth=5)
		# plt.figure(1)
		# plt.legend(loc='best')
		# plt.title('Recall')
		# plt.savefig('../../../DataDump/recall_precision_1/Recall_' + str(len(variables)) +'Nodes_MRF.png')
		# plt.figure(0)
		# plt.legend(loc='best')
		# plt.title('Precision')
		# plt.savefig('../../../DataDump/recall_precision_1/Precision_' + str(len(variables)) +'Nodes_MRF.png')
		# plt.figure(2)
		# plt.legend(loc='best')
		# plt.title('Sufficient stats')
		# plt.savefig('../../../DataDump/recall_precision_1/Sufficient_stats' + str(len(variables)) +'Nodes_MRF.png')
		# plt.close()

if __name__ == '__main__':
	main()