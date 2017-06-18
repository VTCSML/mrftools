import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from grafting_util import compute_likelihood
import time
from Graft import Graft
import sys

METHOD_COLORS = {'structured':'red', 'naive': 'blue', 'queue':'green', 'graft':'yellow'}

test_num = '6'
num_iterations = 1

def main():
	edge_reg = .05
	node_reg = .05
	priority_graft_iter = 2500
	graft_iter = 2500
	# num_nodes_range = range(8, 500, 8)
	T_likelihoods = dict()
	zero_threshold = 1e-3
	training_ratio = .8
	mrf_density = .1
	edge_std = 5
	node_std = 1
	state_num = 4
	l2_coeff = 0
	len_data = 50000
	# num_nodes_range = range(16, 500, 8)
	num_nodes_range = [24]

	print('================================= ///////////////////START//////////////// ========================================= ')

	# for num_cluster in num_cluster_range:
	for num_nodes in num_nodes_range:

		time_likelihoods = dict()
		mean_time_likelihoods = dict()
		std_time_likelihoods = dict()


		time_f1 = dict()
		mean_time_f1 = dict()
		std_time_f1 = dict()

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


		for iteration in range(num_iterations):

			# METHODS = ['naive', 'structured', 'queue']

			METHODS = ['structured']
			sorted_timestamped_mn = dict()
			edge_likelihoods = dict()
			print('======================================Simulating data...')
			# model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(len_data, num_cluster, 8, 8)
			model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
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
			f1s, recalls, precisions, sufficientstats, mn_snapshots = dict(), dict(), dict(), dict(), dict()

			for method in METHODS:
				sys.stdout.flush()
				print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
				spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
				spg.on_show_metrics()
				spg.on_verbose()
				# spg.on_plot_queue('../../../qplot')
				spg.setup_learning_parameters(edge_reg, max_iter_graft=priority_graft_iter, node_l1=node_reg)
				# spg.on_zero_treshold(zero_threshold=zero_threshold)
				spg.on_monitor_mn()
				t = time.time()
				print(edges)
				learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges=edges)
				# print('recall')
				# print(recall)
				# print('precision')
				# print(precision)
				f1_score = []
				for i in range(len(precision)):
						if precision[i]==0 and recall[i]==0:
							f1_score.append(0)
						else:
							f1_score.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))

				exec_time = time.time() - t
				precisions[method] = precision
				recalls[method] = recall
				f1s[method] = f1_score
				print('exec_time')
				print(exec_time)
				print('Converged')
				print(spg.is_converged)
				mn_snapshots[method] = spg.mn_snapshots

			print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
			grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
			grafter.on_show_metrics()
			grafter.on_verbose()
			grafter.setup_learning_parameters(edge_l1 = edge_reg, max_iter_graft=graft_iter, node_l1=node_reg)
			grafter.on_monitor_mn()
			t = time.time()
			print(edges)
			learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure(edge_num, edges=edges)
			# grafter.on_zero_treshold(zero_threshold=zero_threshold)
			exec_time = time.time() - t
			for i in range(len(precision)):
				if precision[i]==0 and recall[i]==0:
					f1_score.append(0)
				else:
					f1_score.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
			precisions['graft'] = precision
			recalls['graft'] = recall
			f1s['graft'] = f1_score
			print('exec_time')
			print(exec_time)
			METHODS.append('graft')
			mn_snapshots['graft'] = grafter.mn_snapshots

			time_stamps = list()

			for method in METHODS:
				if iteration == 0:
					if max_time < max(list(mn_snapshots[method].keys())):
						max_time = max(list(mn_snapshots[method].keys()))
				timestamped_mn_vec = mn_snapshots[method].items()
				timestamped_mn_vec.sort(key=lambda x: x[0])
				# timestamped_mn_vec.sort(key=lambda x: x[1])
				T_likelihoods[method] = [(x[0], compute_likelihood(x[1], num_attributes, test_data)) for x in timestamped_mn_vec]
				print(T_likelihoods[method])
			if iteration == 0:
				max_time = 1.25 * max_time
				step_size = float(max_time) / 25
				time_range = np.arange(0,max_time + step_size,step_size)

			for method in METHODS:
				curr_likelihoods = list()
				method_T_likelihoods = T_likelihoods[method]
				for t in time_range:
					try:
						val = next(x for x in enumerate(method_T_likelihoods) if x[1][0] > t)
						curr_likelihoods.append(val[1][1])
					except:
						curr_likelihoods.append(min(curr_likelihoods))
				if iteration == 0:
					time_likelihoods[method] = np.array([curr_likelihoods]).T
				else:
					print(len(time_likelihoods[method]))
					print(len(np.array([curr_likelihoods]).T))
					time_likelihoods[method] = np.concatenate( (time_likelihoods[method],np.array([curr_likelihoods]).T), axis=1)
			for method in METHODS:
				curr_likelihoods = [x[1] for x in T_likelihoods[method]]
				edge_likelihoods[method] = curr_likelihoods

			############################


			for method in METHODS:
				curr_f1, curr_recall, curr_precision, curr_num_edges = [], [], [], []
				time_stamps = [x[0] for x in T_likelihoods[method]]
				# print(method)
				# print('/////////////////////////')
				# # print(recalls[method])
				# print('/////////////////////////')
				# # print(time_stamps)
				# print('/////////////////////////')
				# print(zip(time_stamps, recalls[method]))
				# print('/////////////////////////')
				# print(time_range)
				# # print([x for x in enumerate(zip(time_stamps, recalls[method]))])
				# # print('/////////////////////////')
				# print([x for x in enumerate(time_stamps)])
				for t in time_range:
					try:
						recall_val_ind = next(x[0] for x in enumerate(zip(time_stamps, recalls[method])) if x[1][0] > t)
						curr_recall.append(recalls[method][recall_val_ind - 1])
					except:
						curr_recall.append(recalls[method][-1])

					try:
						f1_val_ind = next(x[0] for x in enumerate(zip(time_stamps, f1s[method])) if x[1][0] > t)
						curr_f1.append(recalls[method][f1_val_ind - 1])
					except:
						curr_f1.append(f1s[method][-1])

					try:
						precision_val_ind = next(x[0] for x in enumerate(zip(time_stamps, precisions[method])) if x[1][0] > t)
						curr_precision.append(recalls[method][recall_val_ind - 1])
					except:
						curr_precision.append(recalls[method][-1])
					try:
						num_edges = next(x for x in enumerate(time_stamps) if x[1] > t)
						curr_num_edges.append(num_edges[0] - 1)
					except:
						curr_num_edges.append(len(time_stamps))

				if iteration == 0:
					time_f1[method] = np.asarray([curr_f1]).T
					time_recall[method] = np.asarray([curr_recall]).T
					time_precision[method] = np.asarray([curr_precision]).T
					time_num_edges[method] = np.asarray([curr_num_edges]).T
				else:
					time_f1[method] = np.concatenate( (time_f1[method],np.asarray([curr_f1]).T), axis=1)
					time_recall[method] = np.concatenate( (time_recall[method],np.asarray([curr_recall]).T), axis=1)
					time_precision[method] = np.concatenate( (time_precision[method],np.asarray([curr_precision]).T), axis=1)
					time_num_edges[method] = np.concatenate( (time_num_edges[method],np.asarray([curr_num_edges]).T), axis=1)

				# print(curr_recall)
				
			############################



		###################################################

		for method in METHODS:

			mean_time_likelihoods[method] = time_likelihoods[method].mean(1)
			std_time_likelihoods[method] = time_likelihoods[method].std(1)

			mean_time_f1[method] = time_f1[method].mean(1)
			std_time_f1[method] = time_f1[method].std(1)

			mean_time_recall[method] = time_recall[method].mean(1)
			std_time_recall[method] = time_recall[method].std(1)

			mean_time_precision[method] = time_precision[method].mean(1)
			std_time_precision[method] = time_precision[method].std(1)

			mean_time_num_edges[method] = time_num_edges[method].mean(1)
			std_time_num_edges[method] = time_num_edges[method].std(1)

		###################################################


		# print('PLOTTING time VS nllikelihoods')
		for i in range(len(METHODS)):

			# plt.plot(time_range, time_likelihoods[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)

			##################
			plt.plot(time_range, mean_time_likelihoods[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=1)
			plt.fill_between(time_range, mean_time_likelihoods[METHODS[i]] - std_time_likelihoods[METHODS[i]], mean_time_likelihoods[METHODS[i]] + std_time_likelihoods[METHODS[i]], alpha=0.2, color=METHOD_COLORS[METHODS[i]], lw=2)
			#################
		plt.legend(loc='best')
		plt.title('Likelihoods')
		plt.xlabel('time')
		plt.ylabel('likelihood')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + '_TimeVSlikelihoods.png')
		plt.close()

		# ########################## pNLL USE TLIKELIHOODS NOT TIMERANGE
		# for i in range(len(METHODS)):

		# 	# plt.plot(edge_likelihoods[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)

		# 	##################
		# 	plt.plot(range(len(time_range)), mean_time_likelihoods[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=1)
		# 	plt.fill_between(range(len(time_range)), mean_time_likelihoods[METHODS[i]] - std_time_likelihoods[METHODS[i]], mean_time_likelihoods[METHODS[i]] + std_time_likelihoods[METHODS[i]], alpha=0.2, color=METHOD_COLORS[METHODS[i]], lw=2)
		# 	#################

		# plt.legend(loc='best')
		# plt.title('Likelihoods')
		# plt.xlabel('NUM oF edges activated')
		# plt.ylabel('likelihood')
		# plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + '_EdgesVSlikelihoods.png')
		# plt.close()



		########################## NUM EDGES
		for i in range(len(METHODS)):
			time_stamps = [x[0] for x in T_likelihoods[METHODS[i]]]
			edges_created = range(len(time_stamps))
			# plt.plot(time_stamps, edges_created, METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)

			##################
			plt.plot(time_range, mean_time_num_edges[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=1)
			plt.fill_between(time_range, mean_time_num_edges[METHODS[i]] - std_time_num_edges[METHODS[i]], mean_time_num_edges[METHODS[i]] + std_time_num_edges[METHODS[i]], alpha=0.2, color=METHOD_COLORS[METHODS[i]], lw=2)
			#################

		plt.legend(loc='best')
		plt.title('Edge activation')
		plt.xlabel('time')
		plt.ylabel('# edges')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + '_EdgesVStime.png')
		plt.close()

		########################## F1
		for i in range(len(METHODS)):
			# plt.plot(time_stamps, recalls[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
			##################
			plt.plot(time_range, mean_time_f1[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=1)
			plt.fill_between(time_range, mean_time_f1[METHODS[i]] - std_time_f1[METHODS[i]], mean_time_f1[METHODS[i]] + std_time_f1[METHODS[i]], alpha=0.2, color=METHOD_COLORS[METHODS[i]], lw=2)
			#################

		plt.legend(loc='best')
		plt.title('F1')
		plt.xlabel('time')
		plt.ylabel('F1')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + '_F1_.png')
		plt.close()

		########################## RECALL
		for i in range(len(METHODS)):
			# plt.plot(time_stamps, recalls[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
			##################
			plt.plot(time_range, mean_time_recall[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=1)
			plt.fill_between(time_range, mean_time_recall[METHODS[i]] - std_time_recall[METHODS[i]], mean_time_recall[METHODS[i]] + std_time_recall[METHODS[i]], alpha=0.2, color=METHOD_COLORS[METHODS[i]], lw=2)
			#################

		plt.legend(loc='best')
		plt.title('Recall')
		plt.xlabel('time')
		plt.ylabel('recall')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + '_Recall_.png')
		plt.close()

		########################## PRECISION
		for i in range(len(METHODS)):
			time_stamps = [x[0] for x in T_likelihoods[METHODS[i]]]
			edges_created = range(len(time_stamps))
			# plt.plot(time_stamps, precisions[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
			##################
			plt.plot(time_range, mean_time_precision[METHODS[i]], color=METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=1)
			plt.fill_between(time_range, mean_time_precision[METHODS[i]] - std_time_precision[METHODS[i]], mean_time_precision[METHODS[i]] + std_time_precision[METHODS[i]], alpha=0.2, color=METHOD_COLORS[METHODS[i]], lw=2)
			#################

		plt.legend(loc='best')
		plt.title('Precision')
		plt.xlabel('time')
		plt.ylabel('precision')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + '_Precision.png')
		plt.close()

if __name__ == '__main__':
	main()