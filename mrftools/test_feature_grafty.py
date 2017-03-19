import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data, generate_random_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft
from FeatureGraft import FeatureGraft
from grafting_util import compute_likelihood, compute_accuracy_synthetic, compute_likelihood_1
import time
from Graft import Graft
import copy
import itertools

np.set_printoptions(threshold=np.nan)

folder_name = 'feature_graft'
num_iterations = 1
METHOD_COLORS = {'structured':'red', 'naive': 'green', 'queue':'yellow', 'graft':'blue', 'FeatureGraft':'black', 'full_l1':'green'}

def main():

	priority_graft_iter = 5000
	graft_iter = 5000
	T_likelihoods = dict()
	zero_threshold = 1e-3
	training_ratio = .8
	edge_std = 2.5
	node_std = .0001
	state_num = 4
	min_precision = .1 # WORKS WITH MAX NUMBER OF EDGES
	# num_nodes_range = [10]
	num_nodes_range = range(10, 100, 10)

	# l1_coeff_range = np.arange(0.01,.1,0.01)
	# l1_coeff_range = [0]

	l1_coeff_range = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1]


	edge_reg_range = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1] #np.arange(1e-5, 5e-2, 5e-5)#[.04]#[1, 5e-1, 1e-1]
	node_reg_range = [1e-2]

	T_likelihoods = dict()
	M_time_stamps = dict()
	print('================================= ///////////////////START//////////////// =========================================')

	for num_nodes in num_nodes_range:

		mrf_density = float(1)/(2 * num_nodes-1)

		max_edge_num = (num_nodes ** 2 - num_nodes) / 2
		# mrf_density = min(.05, float(num_nodes) / (10 * max_edge_num))

		len_data = min(100000, num_nodes * 500)

		time_likelihoods = dict()
		mean_time_likelihoods = dict()
		std_time_likelihoods = dict()

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
		METHODS = ['FeatureGraft']
		M_accuracies = dict()
		sorted_timestamped_mn = dict()
		edge_likelihoods = dict()
		print('======================================Simulating data...')
		model, variables, data, max_num_states, num_states, edges = generate_random_synthetic_data(len_data, num_nodes, mrf_density=mrf_density, state_min=state_num, state_max=state_num, edge_std=edge_std, node_std = node_std)
		target_vars = list(set(itertools.chain.from_iterable(edges)))

		model.init_search_space()

		for var in variables:
			print(var)
			p = dict()
			for state0 in range(num_states[var]):
				_p0 =[]
				[_p0.append(x) for x in data if x[var]==state0]

				p0 = float(len(_p0)) / len_data

				# print(_p)
				p[state0] = p0
				# p[(state0,state1)] = _p
			print(p)


		for edge in model.search_space:
			print(edge)
			p = dict()
			for state0 in range(num_states[edge[0]]):
				for state1 in range(num_states[edge[1]]):

					_p0 =[]
					[_p0.append(x) for x in data if x[edge[0]]==state0]

					_p1 =[]
					[_p1.append(x) for x in data if x[edge[1]]==state1]

					p0 = float(len(_p0)) / len_data

					p1 = float(len(_p1)) / len_data

					p_test = p0 * p1

					_p =[]
					[_p.append(x) for x in data if x[edge[0]]==state0 and x[edge[1]]==state1]
					# print(_p)
					p[(state0,state1)] = (float(len(_p)) / len_data, p_test)
					# p[(state0,state1)] = _p
			print(p)

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

		edge_num = state_num ** 2 * len(edges) + 10 # Allow precision to go up

		num_attributes = len(variables)
		recalls, precisions, sufficientstats, mn_snapshots, f1_scores = dict(), dict(), dict(), dict(), dict()

		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Feature Graft' )
		solved_feature_graft = False
		best_f1, best_l1 = 0, 0
		best_mn, best_f1_list = None, None
		for l1_coeff in l1_coeff_range:
			print('//////////////////////////////')
			print('l1_coeff')
			print(l1_coeff)
			fg = FeatureGraft(variables, num_states, max_num_states, train_data, list_order)
			# fg.on_verbose()
			fg.on_show_metrics()
			fg.on_synthetic(precison_threshold = min_precision)

			fg.setup_learning_parameters(max_iter_graft=priority_graft_iter, l1_coeff=l1_coeff)
			fg.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, recall, precision, f1_score, is_early_stop = fg.learn_structure(edge_num, edges=edges)

			nll = compute_likelihood(learned_mn, len(variables), test_data)
			nll1 = compute_likelihood(fg.mn_snapshots[min(list(fg.mn_snapshots.keys()))], len(variables), test_data)


			print('finished?')
			print((not is_early_stop))
			if not is_early_stop:
				exec_time = time.time() - t
				if nll < nll1:
					best_f1 = f1_score[-1]
					best_mn = fg.mn_snapshots
					best_f1_list = f1_score
					best_l1 = l1_coeff
					solved_feature_graft = True
					if best_f1 > .75:
						break

		mn_snapshots['FeatureGraft'] = best_mn
		f1_scores['FeatureGraft'] = best_f1_list
		time_stamps = sorted(list(best_mn.keys()))
		M_time_stamps['FeatureGraft'] = time_stamps
		method_likelihoods = []
		accuracies = []
		for t in time_stamps:
			nll = compute_likelihood_1(best_mn[t], len(variables), test_data)
			method_likelihoods.append(nll)

		M_accuracies['FeatureGraft'] = accuracies
		T_likelihoods['FeatureGraft'] = method_likelihoods


		print('>>>>>>>>>>>>>>>>>>>>>METHOD:  Full L1' )

		fg = FeatureGraft(variables, num_states, max_num_states, train_data, list_order)
		# fg.on_verbose()
		fg.on_show_metrics()
		fg.on_full_l1()
		METHODS.append('full_l1')
		fg.setup_learning_parameters(max_iter_graft=priority_graft_iter, l1_coeff=best_l1)
		fg.on_monitor_mn()

		t = time.time()
		learned_mn, final_active_set, recall, precision, f1_score, is_early_stop = fg.learn_structure(edge_num, edges=edges)
		exec_time = time.time() - t

		mn_snapshots['full_l1'] = fg.mn_snapshots
		f1_scores['full_l1'] = f1_score
		time_stamps = sorted(list(fg.mn_snapshots.keys()))
		M_time_stamps['full_l1'] = time_stamps
		method_likelihoods = []
		accuracies = []
		for t in time_stamps:
			nll = compute_likelihood_1(fg.mn_snapshots[t], len(variables), test_data)
			method_likelihoods.append(nll)

		M_accuracies['full_l1'] = accuracies
		T_likelihoods['full_l1'] = method_likelihoods


		print('Real model old')
		real_model_nll = compute_likelihood(model, len(variables), test_data)
		print('Real model new')
		real_model_nll = compute_likelihood_1(model, len(variables), test_data)

		print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
		edge_num = len(edges) + 10
		solved_graft = False
		best_f1 = 0
		METHODS.append('graft')
		for edge_reg in edge_reg_range:
			node_reg = 1.15 * edge_reg
			print('//////////////')
			print('reg params')
			print(edge_reg)
			print(node_reg)
			# print(node_reg)
			# print(edge_reg)
			grafter = Graft(variables, num_states, max_num_states, train_data, list_order)
			grafter.on_show_metrics()
			# grafter.on_verbose()
			grafter.setup_learning_parameters(edge_l1 = edge_reg, max_iter_graft=graft_iter, node_l1=node_reg)
			grafter.on_synthetic(precison_threshold = min_precision)
			grafter.on_monitor_mn()
			t = time.time()
			learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, is_early_stop  = grafter.learn_structure(edge_num, edges=edges)
			print((not is_early_stop))
			if not is_early_stop:
				# print(final_active_set)
				# print(recall)
				f1_scores['graft'] = f1_score
				exec_time = time.time() - t
				precisions['graft'] = precision
				recalls['graft'] = recall

				nll = compute_likelihood(learned_mn, len(variables), test_data)
				nll1 = compute_likelihood(grafter.mn_snapshots[min(list(grafter.mn_snapshots.keys()))], len(variables), test_data)

				if nll < nll1:
					best_f1 = f1_score[-1]
					best_mn = grafter.mn_snapshots
					best_f1_list = f1_score
					solved_graft = True
					if best_f1 > .75:
						break

		mn_snapshots['graft'] = best_mn
		f1_scores['graft'] = best_f1_list
		time_stamps = sorted(list(best_mn.keys()))
		M_time_stamps['graft'] = time_stamps

		method_likelihoods = []
		accuracies = []
		for t in time_stamps:
			nll = compute_likelihood_1(best_mn[t], len(variables), test_data)
			method_likelihoods.append(nll)
		T_likelihoods['graft'] = method_likelihoods
		M_accuracies['graft'] = accuracies


		plt.close()
		if solved_graft and solved_feature_graft:
			fig, ax1 = plt.subplots()
			ax2 = ax1.twinx()
			for i in range(len(METHODS)):
				method = METHODS[i]
				# print(len(M_time_stamps[METHODS[i]]))
				# print(len(T_likelihoods[METHODS[i]]))
				# print(f1_scores)
				# print(f1_scores[METHODS[i]])
				# print(len(f1_scores[METHODS[i]]))
				ax1.plot(M_time_stamps[method], T_likelihoods[method], color=METHOD_COLORS[method], label='nll_' + method, linewidth=1)
				if method != 'full_l1':
					# print(len(f1_scores[method]))
					# print(len(M_time_stamps[method]))
					ax2.plot(M_time_stamps[method], f1_scores[method], METHOD_COLORS[method], linewidth=1, linestyle=':', label='f1-score_'+ method)

			ax1.plot(M_time_stamps['FeatureGraft'], real_model_nll * np.ones(len(T_likelihoods['FeatureGraft'])), color='green', label='nll_real_model', linewidth=1)

			ax1.set_xlabel('time')
			ax1.set_ylabel('nll')
			ax2.set_ylabel('f1_score')

			ax1.legend(loc='best')
			ax2.legend(loc=4, fancybox=True, framealpha=0.5)
			plt.title('nll-precision')
			plt.xlabel('iterations')
			plt.savefig('../../../results_' + folder_name + '/' + str(len(variables)) + '_nll_FeatureVsGroup.png')
			plt.close()



if __name__ == '__main__':
	main()
