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

folder_num = 'l1_metrics'

def main():
	priority_graft_iter = 500
	graft_iter = 500
	zero_threshold = 1e-3
	training_ratio = .8
	mrf_density = .01
	edge_std = 2.5
	node_std = .001
	state_num = 4
	l2_coeff = 0
	num_nodes_range = [10]
	num_nodes_range = range(85, 100, 10)

	min_precision = .05


	edge_reg_range = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1] #np.arange(1e-5, 5e-2, 5e-5)#[.04]#[1, 5e-1, 1e-1]
	node_reg_range = [1e-2]

	# reg_range = np.arange(0,1,0.01)

	# edge_reg_range = np.arange(50,500,10)#[.04]#[1, 5e-1, 1e-1]
	# node_reg_range = np.arange(50,500,10)
	print('================================= ///////////////////START//////////////// =========================================')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		total_edge_num = (num_nodes ** 2 - num_nodes) / 2
		mrf_density = min(mrf_density, float(2)/(num_nodes-1))

		len_data = 10000
		# METHODS = ['naive', 'structured', 'queue']
		METHODS = ['structured']
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
		# edge_num = len(edges) * 2
		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots, f1_scores = dict(), dict(), dict(), dict(), dict()
		
		for method in METHODS:	
			likelihoods = []
			recalls = []
			precisions= []
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			j = 0
			best_params = (0,0)
			best_nll = float('inf')
			for edge_reg in edge_reg_range:
				node_reg = 1.1 * edge_reg
				print('======PARAMS')
				print(edge_reg)
				print(node_reg)
				j += 1
				spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
				spg.on_show_metrics()
				spg.on_verbose()
				spg.on_synthetic(precison_threshold = min_precision, start_num = 10)
				spg.setup_learning_parameters(edge_l1=edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
				spg.on_monitor_mn()
				t = time.time()
				learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, is_early_stop = spg.learn_structure(edge_num, edges=edges)
				print(is_early_stop)
				print(recall)
				print(precision)
				print(spg.active_set)
				recalls.append(recall[-1])
				precisions.append(precision[-1])
				nll = compute_likelihood_1(learned_mn, len(variables), test_data)
				likelihoods.append(nll)
				if nll < best_nll:
					best_nll = nll
					best_params = (node_reg, edge_reg)


				# if len(final_active_set) == 0:
				# 	break
			print('plotting')
			plt.close()
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			ax1.plot(edge_reg_range, likelihoods, color='red', label='nll', linewidth=2)
			ax2.plot(edge_reg_range, recalls, color='green', linewidth=2, linestyle=':', label='recall')
			ax2.plot(edge_reg_range, precisions, color='blue', linewidth=2, linestyle=':', label='precision')
			ax2.set_ylim([-.1,1.1])
			ax1.set_ylabel('nll')
			ax2.set_ylabel('precison/recall')
			ax1.set_xlabel('l1-coeff')
			ax1.legend(loc='best')
			ax2.legend(loc=4, fancybox=True, framealpha=0.5)
			ax1.set_xscale("log", nonposx='clip')
			ax2.set_xscale("log", nonposx='clip')
			plt.title('nll-precision' + '_best:' + str(best_params[0]) + ',' +str(best_params[1]))
			plt.savefig('../../../results_' + folder_num + '/' + str(len(variables)) + '_best:' + str(best_params[0]) + ',' +str(best_params[1]) +'.png')
			plt.close()
if __name__ == '__main__':
	main()