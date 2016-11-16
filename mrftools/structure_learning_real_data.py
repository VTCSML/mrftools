import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from Graft import Graft


METHODS = ['naive', 'structured', 'queue']
METHOD_COLORS = {'structured':'-r', 'naive': '-b', 'queue':'-g', 'graft':'-y'}


def main():
	edge_reg = .07
	num_cluster_range = range(2, 15, 1)
	print('================================= ///////////////////START//////////////// ========================================= ')
	for num_cluster in num_cluster_range:
		print('Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(5000, num_cluster, 6, 9)
		print('EDGES')
		print(edges)
		edge_num = len(edges)
		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		train_data = data[: len(data) - 1001]
		test_data = data[len(data) - 1000 : len(data) - 1]
		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
		grafter = Graft(variables, num_states, max_num_states, data, list_order)
		# grafter.on_show_metrics()
		# grafter.on_verbose()
		# grafter.on_monitor_mn()
		grafter.setup_learning_parameters(edge_reg, max_iter_graft=500)
		learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure(edge_num)
		edges = grafter.active_set
		shuffle(list_order)
		num_attributes = len(variables)
		recalls, precisions, sufficientstats = dict(), dict(), dict()
		for method in METHODS:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
			spg.on_show_metrics()
			# spg.on_verbose()
			# spg.on_plot_queue('../../../DataDump/pq_plot')
			# spg.on_monitor_mn()
			spg.setup_learning_parameters(edge_reg, max_iter_graft=500)
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges=edges)
			recalls[method] = recall
			precisions[method] = precision
			sufficientstats[method] = suff_stats_list


		for i in range(len(METHODS)):
			plt.plot(recalls[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)  
		plt.legend(loc='best')
		plt.title('Recall')
		plt.savefig('../../../DataDump/recall_precision_real_data/Recall_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()

		for i in range(len(METHODS)):
			plt.plot(precisions[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)  
		plt.legend(loc='best')
		plt.title('Precision')
		plt.savefig('../../../DataDump/recall_precision_real_data/Precision_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()

		for i in range(len(METHODS)):
			plt.plot(sufficientstats[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)  
		plt.legend(loc='best')
		plt.title('Sufficient Stats')
		plt.savefig('../../../DataDump/recall_precision_real_data/SufficientStats_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()



if __name__ == '__main__':
	main()