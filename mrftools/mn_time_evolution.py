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


def main():
	edge_reg = .5
	len_data = 5000
	num_cluster_range = range(2, 15, 1)

	MNs, Ts = dict(), dict()

	print('================================= ///////////////////START//////////////// ========================================= ')
	for num_cluster in num_cluster_range:
		METHODS = ['naive', 'structured', 'queue']
		time_likelihoods = dict()
		sorted_timestamped_mn = dict()
		edge_likelihoods = dict()
		print('Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(len_data, num_cluster, 5, 5)
		train_data = data[: int(.8 * len_data)]
		test_data = data[int(.8 * len_data) : len_data]
		list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
		shuffle(list_order)

		print('EDGES')
		print(edges)
		edge_num = len(edges) + 10
		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots = dict(), dict(), dict(), dict()
		for method in METHODS:
			print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + method)
			spg = StructuredPriorityGraft(variables, num_states, max_num_states, train_data, list_order, method)
			spg.on_show_metrics()
			# spg.on_verbose()
			spg.on_plot_queue('../../../DataDump/pq_plot')
			spg.setup_learning_parameters(edge_reg, max_iter_graft=300)
			spg.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges=edges)
			exec_time = time.time() - t
			print('exec_time')
			print(exec_time)
			print('Converged')
			print(spg.is_converged)
			mn_snapshots[method] = spg.mn_snapshots

		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
		grafter = Graft(variables, num_states, max_num_states, data, list_order)
		grafter.on_show_metrics()
		# grafter.on_verbose()
		grafter.setup_learning_parameters(edge_reg, max_iter_graft=300)
		grafter.on_monitor_mn()
		t = time.time()
		learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure(edge_num, edges)
		exec_time = time.time() - t
		print('exec_time')
		print(exec_time)
		METHODS.append('graft')
		mn_snapshots['graft'] = grafter.mn_snapshots

		time_stamps = list()

		max_time = 0
		for method in METHODS:
			print(method)
			print(max(list(mn_snapshots[method].keys())))
			if max_time < max(list(mn_snapshots[method].keys())):
				max_time = max(list(mn_snapshots[method].keys()))
			timestamped_mn_vec = mn_snapshots[method].items()
			timestamped_mn_vec.sort(key=lambda x: x[0])
			sorted_timestamped_mn[method] = timestamped_mn_vec

			MNs[method] = [x[1] for x in timestamped_mn_vec]
			Ts[method] = [x[0] for x in timestamped_mn_vec]


		# step_size = .1

		step_size = float(max_time) / 25

		time_range = np.arange(0,max_time + step_size,step_size)
		print('Getting likelihoods')
		for method in METHODS:
			print('>' + method)
			curr_likelihoods = list()
			method_sorted_timestamped_mn = sorted_timestamped_mn[method]
			for t in time_range:
				try:
					ind = next(x for x, v in enumerate(method_sorted_timestamped_mn) if v[0] > t)
					mn = method_sorted_timestamped_mn[ind][1]
					curr_likelihoods.append(compute_likelihood(mn, num_attributes, test_data))
				except:
					curr_likelihoods.append(min(curr_likelihoods))
			time_likelihoods[method] = curr_likelihoods



		for method in METHODS:
			print('>' + method)
			curr_likelihoods = list()
			for mn in MNs[method]:
					curr_likelihoods.append(compute_likelihood(mn, num_attributes, test_data))
			edge_likelihoods[method] = curr_likelihoods



		print('PLOTTING time VS nllikelihoods')
		for i in range(len(METHODS)):
			plt.plot(time_range, time_likelihoods[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Likelihoods')
		plt.xlabel('time')
		plt.ylabel('likelihood')
		plt.savefig('../../../DataDump/likelihoods/timeVSlikelihoods_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()


		print('PLOTTING NUM oF edges activated Vs nllikelihood')
		for i in range(len(METHODS)):
			plt.plot(edge_likelihoods[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Likelihoods')
		plt.xlabel('NUM oF edges activated')
		plt.ylabel('likelihood')
		plt.savefig('../../../DataDump/likelihoods/EdgesVSlikelihoods_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()


		print('PLOTTING NUM of edges activated VS time')
		for i in range(len(METHODS)):
			plt.plot(Ts[METHODS[i]], METHOD_COLORS[METHODS[i]], label=METHODS[i], linewidth=4)
		plt.legend(loc='best')
		plt.title('Likelihoods')
		plt.xlabel('NUM oF edges activated')
		plt.ylabel('time')
		plt.savefig('../../../DataDump/likelihoods/EdgesVStime_' + str(len(variables)) +'Nodes_MRF.png')
		plt.close()




if __name__ == '__main__':
	main()