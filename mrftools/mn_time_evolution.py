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

METHOD_COLORS = {'structured':'-r', 'naive': '-b', 'queue':'-g', 'graft':'-y'}

test_num = '1'

def main():
	edge_reg = 1
	node_reg = 0
	len_data = 20000
	graft_iter = 2500
	num_cluster_range = range(10, 500, 1)
	T_likelihoods = dict()
	print('================================= ///////////////////START//////////////// ========================================= ')
	for num_cluster in num_cluster_range:
		
		METHODS = ['naive', 'structured', 'queue']
		time_likelihoods = dict()
		sorted_timestamped_mn = dict()
		edge_likelihoods = dict()
		print('======================================Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(len_data, num_cluster, 8, 7)
		train_data = data[: int(.7 * len_data)]
		test_data = data[int(.7 * len_data) : len_data]
		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		shuffle(list_order)

		print(variables)
		print(num_states)
		print(max_num_states)

		print('NUM VARIABLES')
		print(len(variables))

		print('NUM EDGES')
		print(len(edges))
		# edge_num = float('inf')
		edge_num = len(edges) + 10
		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots = dict(), dict(), dict(), dict()
		for method in METHODS:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
			spg.on_show_metrics()
			# spg.on_verbose()
			spg.on_plot_queue('../../../DataDump/pq_plot')
			spg.setup_learning_parameters(edge_reg, max_iter_graft=graft_iter, node_l1=node_reg, zero_threshold=1e-3)
			spg.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges=edges)
			exec_time = time.time() - t
			precisions[method] = precision
			recalls[method] = recall
			print('exec_time')
			print(exec_time)
			print('Converged')
			print(spg.is_converged)
			mn_snapshots[method] = spg.mn_snapshots

		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
		grafter = Graft(variables, num_states, max_num_states, data, list_order)
		grafter.on_show_metrics()
		# grafter.on_verbose()
		grafter.setup_learning_parameters(edge_reg, max_iter_graft=graft_iter)
		grafter.on_monitor_mn()
		t = time.time()
		learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure(edge_num, edges)
		exec_time = time.time() - t
		precisions['graft'] = precision
		recalls['graft'] = recall
		print('exec_time')
		print(exec_time)
		METHODS.append('graft')
		mn_snapshots['graft'] = grafter.mn_snapshots

		time_stamps = list()

		max_time = 0
		for method in METHODS:
			if max_time < max(list(mn_snapshots[method].keys())):
				max_time = max(list(mn_snapshots[method].keys()))
			timestamped_mn_vec = mn_snapshots[method].items()
			timestamped_mn_vec.sort(key=lambda x: x[0])
			T_likelihoods[method] = [(x[0], compute_likelihood(x[1], variables, test_data)) for x in timestamped_mn_vec]

		# step_size = .1

		step_size = float(max_time) / 10

		time_range = np.arange(0,max_time + step_size,step_size)
		# print('Getting likelihoods')
		for method in METHODS:
			# print('>' + method)
			curr_likelihoods = list()
			method_T_likelihoods = T_likelihoods[method]
			for t in time_range:
				try:
					val = next(x for x in enumerate(method_T_likelihoods) if x[1][0] > t)
					curr_likelihoods.append(val[1][1])
					# print('/////')
					# print(t)
					# print(val[1][0])
					# print(val[1][1])
				except:
					curr_likelihoods.append(min(curr_likelihoods))
			time_likelihoods[method] = curr_likelihoods

		for method in METHODS:
			# print('>' + method)
			curr_likelihoods = [x[1] for x in T_likelihoods[method]]
			edge_likelihoods[method] = curr_likelihoods

		# print('PLOTTING time VS nllikelihoods')
		for i in range(len(METHODS)):
			plt.plot(time_range, time_likelihoods[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Likelihoods')
		plt.xlabel('time')
		plt.ylabel('likelihood')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + 'timeVSlikelihoods.png')
		plt.close()

		# print('PLOTTING NUM oF edges activated Vs nllikelihood')
		for i in range(len(METHODS)):
			plt.plot(edge_likelihoods[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Likelihoods')
		plt.xlabel('NUM oF edges activated')
		plt.ylabel('likelihood')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + 'EdgesVSlikelihoods.png')
		plt.close()

		# print('PLOTTING NUM of edges activated VS time')
		for i in range(len(METHODS)):
			time_stamps = [x[0] for x in T_likelihoods[METHODS[i]]]
			edges_created = range(len(time_stamps))
			plt.plot(time_stamps, edges_created, METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Edge activation')
		plt.xlabel('time')
		plt.ylabel('# edges')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + 'EdgesVStime.png')
		plt.close()

		# print('PLOTTING NUM of edges activated VS time')
		for i in range(len(METHODS)):
			time_stamps = [x[0] for x in T_likelihoods[METHODS[i]]]
			edges_created = range(len(time_stamps))
			plt.plot(time_stamps, recalls[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Edge activation')
		plt.xlabel('time')
		plt.ylabel('# edges')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + 'Recall_.png')
		plt.close()

		# print('PLOTTING NUM of edges activated VS time')
		for i in range(len(METHODS)):
			time_stamps = [x[0] for x in T_likelihoods[METHODS[i]]]
			edges_created = range(len(time_stamps))
			plt.plot(time_stamps, precisions[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Edge activation')
		plt.xlabel('time')
		plt.ylabel('# edges')
		plt.savefig('../../../results_' + test_num + '/' + str(len(variables)) + 'Precision.png')
		plt.close()

if __name__ == '__main__':
	main()