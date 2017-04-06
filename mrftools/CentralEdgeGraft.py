from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
from grafting_util import initialize_priority_queue, reset_unary_factors, reset_edge_factors
from graph_mining_util import make_graph, select_edge_to_inject
import copy
import operator
from graph_mining_util import draw_graph, make_graph
import scipy.sparse as sps
import itertools
import os
import networkx as nx
import random
from scipy.misc import logsumexp
from grafting_util import compute_likelihood, sanity_check_likelihood

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# + Functionality:
#   Pairwise MRF stracture learning using structured priority grafting
#
# + TODO : add logs and reference
# 
# + Author: (walidch)
# + Reference : 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CentralEdgeGraft():
    """
    Structured priority grafting class
    """
    def __init__(self, variables, num_states, max_num_states, data, list_order, ss_test=dict(), pq_dict = None):
        """
        Initialize StructuredPriorityGraft class
        """
        self.variables = variables
        self.num_states = num_states
        self.max_num_states = max_num_states
        self.mn = MarkovNet()
        self.mn.initialize_unary_factors(variables, num_states)

        if list_order == None:
            self.search_space = self.mn.search_space
        else:
            self.search_space = [self.mn.search_space[i] for i in list_order]
        self.initial_search_space = copy.deepcopy(self.search_space)

        self.data = data

        self.sufficient_stats_test = ss_test

        # for edge in self.search_space:
        #     self.sufficient_stats_test[edge], _ = self.get_sufficient_stats_per_edge(self.mn, edge)

        self.sufficient_stats, self.padded_sufficient_stats = self.mn.get_unary_sufficient_stats(self.data , self.max_num_states)

        if pq_dict == None:
            self.pq = initialize_priority_queue(search_space=self.search_space)
        else:
            self.pq = pq_dict

        self.search_space = set(self.search_space)

        self.edges_list = list()
        self.l1_coeff = 0
        self.l2_coeff = 0
        self.node_l1 = 0
        self.edge_l1 = .1
        self.max_iter_graft = 500
        self.is_limit_sufficient_stats = False
        self.sufficient_stats_ratio = 1.0
        self.current_sufficient_stats_ratio = .0
        self.active_set = []
        self.edge_regularizers, self.node_regularizers = dict(), dict()
        self.graph_snapshots, self.mn_snapshots = dict(), dict()
        self.is_show_metrics = False
        self.is_plot_queue = False
        self.is_verbose = False
        self.priority_decrease_decay_factor = .75
        self.plot_path = '.'
        self.is_monitor_mn = False
        self.is_converged = False
        self.graph = nx.Graph()
        self.k_graph = nx.Graph()
        self.frozen = dict()
        self.total_iter_num = list()
        for var in self.variables:
            self.graph.add_node(var)
            self.frozen[var] = list()
        self.subgraphs = dict()
        self.is_remove_zero_edges = False
        self.is_synthetic = False
        self.zero_threshold = 1e-2
        self.precison_threshold = .7
        self.start_num = 4
        self.ss_at_70 = 1
        self.is_freeze_neighbors = False
        self.frozen_list = list()
        self.reordered =0
        self.correctly_reordered = 0
        self.is_real_loss = False
        self.is_added = True
        self.k = 5
        self.k_relevant = dict()
        self.candidate_graph = dict()
        self.structured = False
        self.centrality = dict()
        for var in self.variables:
            self.centrality[var] = 0
        self.alpha = .5

        self.m = 5

    def set_alpha(self, alpha):
        self.alpha = alpha

    def on_structured(self):
        self.structured = True

    def set_top_relvant(self, k=5):
        self.k = k

    def on_synthetic(self, precison_threshold=.7, start_num=4):
        self.is_synthetic = True
        self.precison_threshold = precison_threshold
        self.start_num = start_num

    def set_priority_factors(self, priority_increase_decay_factor, priority_decrease_decay_factor):
        """
        Set priority factors
        """
        self.priority_increase_decay_factor = priority_increase_decay_factor
        self.priority_decrease_decay_factor = priority_decrease_decay_factor

    def on_limit_sufficient_stats(self, max_sufficient_stats_ratio):
        """
        Set a limit for the maximum amount of sufficient statistics to be computed
        """
        self.is_limit_sufficient_stats = True
        self.sufficient_stats_ratio = max_sufficient_stats_ratio

    def on_zero_treshold(self, zero_threshold=1e-2):
        """
        Set a limit for the maximum amount of sufficient statistics to be computed
        """
        self.is_remove_zero_edges = True
        self.zero_threshold = zero_threshold

    def on_monitor_mn(self, is_real_loss=False):
        """
        Enable monitoring Markrov net
        """
        self.is_monitor_mn = True
        self.is_real_loss = is_real_loss
        # self.full_mn = MarkovNet()
        # self.full_mn.mn.initialize_unary_factors(variables, num_states)

    def on_freeze_neighbors(self):
        """
        Enable monitoring Markrov net
        """
        self.is_freeze_neighbors = True

    def setup_learning_parameters(self, edge_l1=0, node_l1=0, l1_coeff=0, l2_coeff=0, max_iter_graft=500):
        """
        Set grafting parameters
        """
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.node_l1 = node_l1
        self.edge_l1 = edge_l1
        self.max_iter_graft = max_iter_graft

    def on_show_metrics(self):
        self.is_show_metrics = True

    def on_plot_queue(self, plot_path='.'):
        self.is_plot_queue = True
        self.plot_path = plot_path

    def on_verbose(self):
        self.is_verbose = True

    def is_limit_sufficient_stats_reached(self):
        """
        Check if limit of sufficient stats is reached
        """
        return not self.is_limit_sufficient_stats or ( (len(self.sufficient_stats) - len(self.variables) ) / float(len(self.search_space)) < self.sufficient_stats_ratio )

    def save_mn(self, exec_time=0):
        learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
        learned_mn.load_factors_from_matrices()
        # nll = compute_likelihood(learned_mn, len(self.variables), self.data)
        self.mn_snapshots[exec_time] = learned_mn

    def save_graph(self, exec_time=0):
        graph = copy.deepcopy(self.graph)
        self.graph_snapshots[exec_time] = graph

    def set_regularization_indices(self, unary_indices, pairwise_indices):
        for node in self.variables:
            self.node_regularizers[node] = (unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]])
        for edge in self.active_set:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]

    def reinit_weight_vec(self, unary_indices, pairwise_indices, weights_opt, vector_length_per_edge, old_node_regularizers, old_edge_regularizers ):
        # tmp_weights_opt = np.random.randn(len(weights_opt) + vector_length_per_edge)
        tmp_weights_opt = np.zeros(len(weights_opt) + vector_length_per_edge)
        for edge in self.active_set[:len(self.active_set) - 1]:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]
            try:
                tmp_weights_opt[list(self.edge_regularizers[edge].flatten())] = weights_opt[list(old_edge_regularizers[edge].flatten())]
                # print(tmp_weights_opt)
            except:
                pass
        if len(old_node_regularizers) > 0:
            for node in self.variables:
                # self.node_regularizers.extend(unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]])
                self.node_regularizers[node] = unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]]
                try:
                    tmp_weights_opt[list(self.node_regularizers[node])] = weights_opt[list(old_node_regularizers[node])]
                    # print(tmp_weights_opt)
                except:
                    pass
        old_edge_regularizers = copy.deepcopy(self.edge_regularizers)
        old_node_regularizers = copy.deepcopy(self.node_regularizers)

        # print(tmp_weights_opt)

        return tmp_weights_opt, old_node_regularizers, old_edge_regularizers



    def learn_structure(self, num_edges, edges=list()):
        """
        Main function for grafting
        """
        data_len = len(self.data)
        np.random.seed(0)
        self.edges = edges

        num_features = 0

        for var in self.variables:
            num_features += len(self.mn.unary_potentials[var]) 
        for edge in self.search_space:
            num_features += len(self.mn.unary_potentials[edge[0]]) *  len(self.mn.unary_potentials[edge[1]])

        if self.is_monitor_mn:
            exec_time_origin = time.time()
        self.aml_optimize = self.setup_grafting_learner(len(self.data))

        vector_length_per_var = self.max_num_states
        vector_length_per_edge = self.max_num_states ** 2
        len_search_space = len(self.search_space)

        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        self.set_regularization_indices(unary_indices, pairwise_indices)

        old_node_regularizers = self.node_regularizers
        old_edge_regularizers = self.edge_regularizers

        objec = list()
        if self.is_monitor_mn:
            recall, precision, suff_stats_list, self.total_iter_num, f1_score = [0,0], [0,0], [0,0], [0,0], [0,0]
            is_ss_at_70_regeistered = False

        metric_exec_time = 0
        tmp_weights_opt = np.zeros(self.aml_optimize.weight_dim)
        weights_opt, tmp_metric_exec_time = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, data_len, verbose=False, loss=objec, ss_test = self.sufficient_stats_test, search_space = self.search_space, len_data = data_len, bp = self.aml_optimize.belief_propagators[0], is_real_loss = self.is_real_loss)
        metric_exec_time += tmp_metric_exec_time

        objec.extend(objec)

        if self.is_monitor_mn:
            exec_time =  time.time() - exec_time_origin - metric_exec_time
            self.save_mn()
            self.save_mn(exec_time=exec_time)
            self.save_graph(exec_time=exec_time)

        if self.is_plot_queue:
            columns, rows, values = list(), list(), list()
            loop_num = 0
            pq_history = dict()

        is_activated_edge, activated_edge, iter_number = self.activation_test()

        old_edge_regularizers = []

        while (len(self.pq) > 0 or is_activated_edge or len(self.k_relevant)>0 ) and len(self.active_set) < num_edges and self.is_limit_sufficient_stats_reached(): # Stop if all edges are added or no edge is added at the previous iteration
            
            if len(self.pq) == 0 and not is_activated_edge :
                activated_edge = max(self.k_relevant.iteritems(), key=operator.itemgetter(1))[0]
                self.k_relevant.pop(activated_edge, None)

            self.active_set.append(activated_edge)
            if self.is_verbose:
                # print(precision)
                self.print_update(activated_edge)

            if self.is_synthetic and precision[-1] < self.precison_threshold and len(self.active_set) > self.start_num:
                learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
                learned_mn.load_factors_from_matrices()
                return learned_mn, None, suff_stats_list, recall, precision, f1_score, objec, True

            if self.is_plot_queue:
                loop_num = self.update_plot_info(loop_num, columns, rows, values, pq_history, edges)

            self.graph.add_edge(activated_edge[0], activated_edge[1])
            self.aml_optimize = self.setup_grafting_learner(len(self.data))

            unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
            self.set_regularization_indices(unary_indices, pairwise_indices)

            tmp_weights_opt, old_node_regularizers, old_edge_regularizers= self.reinit_weight_vec(unary_indices, pairwise_indices, weights_opt, vector_length_per_edge, old_node_regularizers, old_edge_regularizers)

            weights_opt, tmp_metric_exec_time = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, data_len, verbose=False, loss=objec, ss_test = self.sufficient_stats_test, search_space = self.search_space, len_data = data_len, bp = self.aml_optimize.belief_propagators[0], is_real_loss = self.is_real_loss)
            metric_exec_time += tmp_metric_exec_time


            if self.is_monitor_mn:
                exec_time = time.time() - exec_time_origin - metric_exec_time
                self.save_mn(exec_time=exec_time)
                self.save_graph(exec_time=exec_time)

            self.total_iter_num.append(iter_number)
            if self.is_show_metrics:
                self.update_metrics(edges, recall, precision, f1_score, suff_stats_list)
                if f1_score[-1] >= .7:
                    self.ss_at_70 = (len(self.sufficient_stats) - len(self.variables)) / len(self.initial_search_space)
                    is_ss_at_70_regeistered = True
            
            is_activated_edge, activated_edge, iter_number = self.activation_test()

        if is_activated_edge == False:
                self.is_converged = True

        final_active_set = self.active_set

        if self.is_remove_zero_edges:
            # REMOVE NON RELEVANT EDGES
            final_active_set = self.remove_zero_edges()
            self.active_set = final_active_set

        if self.is_plot_queue:
            # self.make_queue_plot_ground_truth(values, rows, columns, loop_num, len_search_space)
            self.make_queue_plot_synthetic_truth(pq_history, final_active_set, loop_num, len_search_space)

        learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
        learned_mn.load_factors_from_matrices()

        if self.is_show_metrics:
            self.print_metrics(recall, precision, f1_score)
            print('Correctly re-ordered')
            try:
                print(float(self.reordered) )
                print( float(self.correctly_reordered) / float(self.reordered) )
            except:
                print('N/A')
            return learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, False

        if self.is_verbose:
            print('Final Active Set')
            print(final_active_set)

        return learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, False

    def update_k_relevant(self, bp):
        #Update relevant edges
        for rel_edge in self.k_relevant.keys():
            sufficient_stat_edge = self.sufficient_stats[rel_edge]
            belief = bp.var_beliefs[rel_edge[0]] + np.matrix(bp.var_beliefs[rel_edge[1]]).T
            gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(sufficient_stat_edge) / len(self.data)).squeeze()
            gradient_norm = np.sqrt(gradient.dot(gradient))
            length_normalizer = np.sqrt(len(gradient))
            passed = (gradient_norm / length_normalizer) > self.edge_l1
            if not passed:
                direct_penalty = 1 - gradient_norm/(length_normalizer * self.edge_l1)
                self.frozen_list.append((rel_edge, direct_penalty))
                self.k_relevant.pop(rel_edge, None)
                self.candidate_graph.setdefault(rel_edge[0], []).remove(rel_edge[1])
                self.candidate_graph.setdefault(rel_edge[1], []).remove(rel_edge[0])
            else:
                self.k_relevant[rel_edge] = (float(self.centrality[rel_edge[0]] + self.centrality[rel_edge[1]]) / 2, (gradient_norm / length_normalizer))

    # def update_pq(self, bp):
    #     for node in self.candidate_graph.keys():
    #         if len(self.candidate_graph[node])>0 and self.graph.degree(node)>0:
    #             for cand_neighbor in self.candidate_graph[node]:
    #                 candidate_edge = (min(node, cand_neighbor), max(node, cand_neighbor))
    #                 for real_neighbor in self.graph[node]:
    #                     real_edge = (min(node, real_neighbor), max(node, real_neighbor))
    #                     edge = (min(cand_neighbor, real_neighbor), max(cand_neighbor, real_neighbor))
    #                     belief = bp.var_beliefs[real_edge[0]] + np.matrix(bp.var_beliefs[real_edge[1]]).T
    #                     # belief = np.exp(belief.T.reshape((-1, 1)).tolist()).squeeze()
    #                     gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(self.sufficient_stats[real_edge]) / len(self.data)).squeeze()
    #                     gradient_norm = np.sqrt(gradient.dot(gradient))
    #                     length_normalizer = np.sqrt(len(gradient))
    #                     deviation = (gradient_norm / (length_normalizer * self.edge_l1))
    #                     if edge in self.pq:
    #                         print('//////////////////')
    #                         print('edge to update')
    #                         print(edge)
    #                         print('rel edge gradient')
    #                         print(deviation)
    #                         self.pq.updateitem(edge, self.pq[edge] + 1 - (self.k_relevant[(min(node, cand_neighbor), max(node, cand_neighbor))] / self.edge_l1))
    #                         print('candidate edge energy')
    #                         print(self.k_relevant[candidate_edge]/ self.edge_l1)
    #                         print('is relevant')
    #                         print(edge in self.edges)
    #                         print('//////////////////')

    def update_pq(self, bp, candidate_edge):

        self.centrality = nx.betweenness_centrality(self.graph)
        self.m_central_nodes = [x[0] for x in sorted(self.centrality.iteritems(), key=operator.itemgetter(1), reverse=True)[:self.m]]
                for node_1 in self.m_central_nodes:
                    for node_2 in self.m_central_nodes:
                        if (node_1,node_2) in self.pq:
                            self.pq.updateitem((node_1,node_2), self.pq[(node_1,node_2)] - 1)
        node_0 = candidate_edge[0]
        node_1 = candidate_edge[1]
        belief = bp.var_beliefs[candidate_edge[0]] + np.matrix(bp.var_beliefs[candidate_edge[1]]).T
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(self.sufficient_stats[candidate_edge]) / len(self.data)).squeeze()
        gradient_norm = np.sqrt(gradient.dot(gradient))
        length_normalizer = np.sqrt(len(gradient))
        candidate_deviation = (gradient_norm / (length_normalizer * self.edge_l1))



        for real_neighbor in self.graph[node_0]:
            # real_edge = (min(node_0, real_neighbor), max(node_0, real_neighbor))

            # belief = bp.var_beliefs[real_edge[0]] + np.matrix(bp.var_beliefs[real_edge[1]]).T
            # gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(self.sufficient_stats[real_edge]) / len(self.data)).squeeze()
            # gradient_norm = np.sqrt(gradient.dot(gradient))
            # length_normalizer = np.sqrt(len(gradient))
            # real_edge_deviation = (gradient_norm / (length_normalizer * self.edge_l1))

            # deviation = real_edge_deviation + candidate_deviation

            edge = (min(node_1, real_neighbor), max(node_1, real_neighbor))
            if edge in self.pq:
                # print('//////////////////////')
                # print('updated edge')
                # print(edge)
                # print(self.pq[edge])
                # # print('deviation')
                # # print(deviation)
                # print('Is relevant?')
                # print(edge in self.edges)

                # print(self.pq)
                # update = self.pq[edge] + deviation

                # self.pq.updateitem(edge, self.pq[edge] - deviation)

                self.pq.updateitem(edge, self.pq[edge] + 1)

                # self.pq.updateitem(edge, self.pq[edge] - 1)
                # print('rank of edge in pq')
                # print(sorted(list(self.pq.values())).index(self.pq[edge]))
                # print('len pq')
                # print(self.pq[edge])
                # print('//////////////////////')




        for real_neighbor in self.graph[node_1]:
            # real_edge = (min(node_1, real_neighbor), max(node_1, real_neighbor))
            # belief = bp.var_beliefs[real_edge[0]] + np.matrix(bp.var_beliefs[real_edge[1]]).T
            # gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(self.sufficient_stats[real_edge]) / len(self.data)).squeeze()
            # gradient_norm = np.sqrt(gradient.dot(gradient))
            # length_normalizer = np.sqrt(len(gradient))
            # real_edge_deviation = (gradient_norm / (length_normalizer * self.edge_l1))

            # deviation = real_edge_deviation + candidate_deviation

            edge = (min(node_0, real_neighbor), max(node_0, real_neighbor))
            if edge in self.pq:
                # print('//////////////////////')
                # print('updated edge')
                # print(edge)
                # print(self.pq[edge])
                # # print('deviation')
                # # print(deviation)
                # print('Is relevant?')
                # print(edge in self.edges)
                # update = self.pq[edge] + deviation
                # print(self.pq)

                # self.pq.updateitem(edge, self.pq[edge] - deviation)
                self.pq.updateitem(edge, self.pq[edge] + 1)

                # self.pq.updateitem(edge, self.pq[edge] - 1)
                # print('rank of edge in pq')
                # print(sorted(list(self.pq.values())).index(self.pq[edge]))
                # print('len pq')
                # print(self.pq[edge])
                # print('//////////////////////')

    def activation_test(self):
        """
        Test edges for activation
        """
        refill = False
        priority_min = 1e+5
        iteration_activation = 0
        tmp_list = list()
        bp = self.aml_optimize.belief_propagators[0]
        bp.load_beliefs()
        len_data = len(self.data)

        self.update_k_relevant(bp)

        while len(self.pq)>0:
            while len(self.pq)>0:

                item = self.pq.popitem()# Get edges by order of priority
                edge = item[0]
                iteration_activation += 1
                if edge in self.sufficient_stats:
                    sufficient_stat_edge = self.sufficient_stats[edge]
                else:
                    t1 = time.time()
                    sufficient_stat_edge, padded_sufficient_stat_edge =  self.get_sufficient_stats_per_edge(self.aml_optimize.belief_propagators[0].mn,edge)
                    self.sufficient_stats[edge] = sufficient_stat_edge
                    self.padded_sufficient_stats[edge] = padded_sufficient_stat_edge
                belief = bp.var_beliefs[edge[0]] + np.matrix(bp.var_beliefs[edge[1]]).T
                gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(sufficient_stat_edge) / len_data).squeeze()
                gradient_norm = np.sqrt(gradient.dot(gradient))
                length_normalizer = np.sqrt(len(gradient))
                normalized_gradient = gradient_norm / length_normalizer
                passed = normalized_gradient > self.edge_l1

                if passed:
                    self.is_added = True
                    # self.k_relevant[edge] = (float(self.centrality[edge[0]] + self.centrality[edge[1]]) / 2, normalized_gradient)
                    self.k_relevant[edge] = normalized_gradient

                    if self.structured and len(self.active_set) > 5:
                        self.update_pq(bp, edge)

                    self.candidate_graph.setdefault(edge[0], []).append(edge[1])
                    self.candidate_graph.setdefault(edge[1], []).append(edge[0])

                    if len(self.k_relevant) == self.k:
                        # centrality_normalizer = sum([self.k_relevant[x][0] for x in self.k_relevant.keys()])
                        # gradient_normalizer = sum([self.k_relevant[x][1] for x in self.k_relevant.keys()])
                        # if centrality_normalizer == 0:
                        #     centrality_normalizer = 1
                        # max_score = 0
                        # for edge in self.k_relevant.keys():
                        #     edge_score = self.alpha * self.k_relevant[edge][0] / centrality_normalizer + (1 - self.alpha ) * self.k_relevant[edge][1] / gradient_normalizer
                        #     if edge_score > max_score:
                        #         max_score = edge_score
                        #         selected_edge = edge

                        selected_edge = max(self.k_relevant.iteritems(), key=operator.itemgetter(1))[0]
                        self.k_relevant.pop(selected_edge, None)
                        self.search_space.remove(selected_edge)
                        self.candidate_graph.setdefault(selected_edge[0], []).remove(selected_edge[1])
                        self.candidate_graph.setdefault(selected_edge[1], []).remove(selected_edge[0])
                        # self.is_added = True
                        return True, selected_edge, iteration_activation
                else:
                    direct_penalty = 1 - gradient_norm/(length_normalizer * self.edge_l1)
                    self.frozen_list.append((edge, direct_penalty))

            if self.is_added:
                self.is_added = False
                for frozen_items in self.frozen_list:
                    self.pq.additem(frozen_items[0], frozen_items[1] )
                refill = True
                self.frozen_list = list()
        return False, (0, 0), iteration_activation


    def setup_grafting_learner(self, len_data):
        """
        Initialize learner with training data
        """
        self.mn = MarkovNet()
        self.mn.initialize_unary_factors(self.variables, self.num_states)
        for edge in self.active_set:
            self.mn.set_edge_factor(edge, np.ones((len(self.mn.unary_potentials[edge[0]]), len(self.mn.unary_potentials[edge[1]]))))
        aml_optimize = ApproxMaxLikelihood(self.mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(self.l1_coeff, self.l2_coeff, self.node_l1, self.edge_l1)
        aml_optimize.init_grafting()
        unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        tau_q = np.zeros(aml_optimize.weight_dim)
        for var in self.mn.variables:
            i = aml_optimize.belief_propagators[0].mn.var_index[var]
            inds = unary_indices[:, i]
            tau_q[inds] = self.padded_sufficient_stats[var] / len_data
        for edge in self.active_set:
            i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
            inds = pairwise_indices[:, :, i]
            tau_q[inds] = self.padded_sufficient_stats[edge] / len_data
        aml_optimize.set_sufficient_stats(tau_q)
        return aml_optimize

    def get_linked_inactive_tested_edges(self, edge):
        linked_inactive_tested_edges = list()

        reachable_nodes = self.update_subgraphs(edge)

        for node in reachable_nodes:
            linked_inactive_tested_edges.extend(self.frozen[node]) # GET ALL FROZEN EDGES THAT INCLUDE node as one end
            for item in self.frozen[node]:# REMOVE all FROZEN EDGES THAT INCLUDE node as one end
                edge = item[0]
                if edge[0] != node:
                    self.frozen.setdefault(edge[0], []).remove(item)
                else:
                    self.frozen.setdefault(edge[1], []).remove(item)
            # print(self.frozen[node])
            self.frozen[node] = list() # REMOVE all FROZEN EDGES THAT INCLUDE node as one end

        linked_inactive_tested_edges = list(set(linked_inactive_tested_edges))

        return linked_inactive_tested_edges



    def update_subgraphs(self, edge):

        descendants = list(set(list(nx.shortest_path(self.graph,source=edge[0])) + list(nx.shortest_path(self.graph,source=edge[1]))))

        # subgraph_0, subgraph_1 = None, None # MERGE SUBGRAPHS TOGETHER
        # key_0, key_1 = 'k1', 'k2'
        # for subgraph_key in self.subgraphs.keys():
        #     if edge[0] in self.subgraphs[subgraph_key]:
        #         subgraph_0 = self.subgraphs[subgraph_key]
        #         key_0 = subgraph_key
        #     if edge[1] in self.subgraphs[subgraph_key]:
        #         subgraph_1 = self.subgraphs[subgraph_key]
        #         key_1 = subgraph_key
        # if key_0 == key_1:
        #     return []
        # descendants = list()
        # if subgraph_0 == None and subgraph_1 == None:
        #     self.subgraphs[edge[0]] = [edge[0], edge[1]]
        # if subgraph_0 == None and subgraph_1 != None:
        #     self.subgraphs.setdefault(key_1, []).append(edge[0])
        #     descendants = self.subgraphs[key_1]
        # if subgraph_0 != None and subgraph_1 == None:
        #     self.subgraphs.setdefault(key_0, []).append(edge[1])
        #     descendants = self.subgraphs[key_0]
        # if subgraph_0 != None and subgraph_1 != None:
        #     self.subgraphs.setdefault(key_0, []).extend(self.subgraphs[key_1])
        #     self.subgraphs.pop(key_1, None)
        #     descendants = self.subgraphs[key_0]

        return descendants # RETURN nodes that can be reachable from any of edge's two nodes


    def print_update(self, activated_edge):
        """
        print update
        """
        print('ACTIVATED EDGE')
        print(activated_edge)
        print('LENGTH CURRENT ACTIVE SPACE')
        print(len(self.active_set))

    def update_metrics(self, edges, recall, precision, f1_score, suff_stats_list):
        """
        UPDATE METRICS
        """
        try:
            curr_recall = float(len([x for x in self.active_set if x in edges]))/len(edges)
        except:
            curr_recall = 0
        recall.append(curr_recall)

        try:
            curr_precision = float(len([x for x in edges if x in self.active_set]))/len(self.active_set)
        except:
            curr_precision = 0

        precision.append(curr_precision)

        if curr_precision==0 or curr_recall==0:
            curr_f1_score = 0
        else:
            curr_f1_score = (2 * curr_precision * curr_recall) / (curr_precision + curr_recall)

        f1_score.append(curr_f1_score)

        suff_stats_list.append(float(len(self.sufficient_stats) - len(self.variables))/ len(self.initial_search_space))

    def remove_zero_edges(self):
        """
        Filter out near zero edges
        """
        bp = self.aml_optimize.belief_propagators[0]
        self.aml_optimize.belief_propagators[0].mn.update_edge_tensor()
        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        final_active_set = list()
        # print('REMOVING ZERO EDGES')
        for edge in self.active_set:
            # print('EDGE')
            # print(edge)

            i = self.aml_optimize.belief_propagators[0].mn.edge_index[edge]
            edge_weights = self.aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:self.aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :self.aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
            # length_normalizer = float(1) / len(edge_weights)
            # print(np.sqrt(edge_weights.dot(edge_weights)))
            if np.sqrt(edge_weights.dot(edge_weights)) / np.sqrt(len(edge_weights))  > self.zero_threshold:
                final_active_set.append(edge)
        return final_active_set

    def print_metrics(self, recall, precision, f1_score):
        """
        Print metrics
        """
        print('Average Selection iterations')
        try:
            print(float(sum(self.total_iter_num))/len(self.total_iter_num))
        except:
            print('N/A')
        print('SUFFICIENT STATS')
        print(len(self.sufficient_stats)- len(self.variables))
        print('Recall')
        try:
            print(recall[-1])
        except:
            print(0)
        print('Precision')
        try:
            print(precision[-1])
        except:
            print(0)
        print('F1-score')
        try:
            print(f1_score[-1])
        except:
            print(0)

    def get_sufficient_stats_per_edge(self, mn, edge):
        """
        Compute joint states reoccurrences in the data
        """
        edge_padded_sufficient_stats = np.asarray(np.zeros((self.max_num_states, self.max_num_states)).reshape((-1, 1)))
        edge_sufficient_stats = np.asarray(np.zeros((len(mn.unary_potentials[edge[0]]), (len(mn.unary_potentials[edge[1]])))).reshape((-1, 1)))
        for states in self.data:
            padded_table = np.zeros((self.max_num_states, (self.max_num_states)))
            padded_table[states[edge[0]], states[edge[1]]] = 1
            padded_tmp = np.asarray(padded_table.reshape((-1, 1)))
            table = np.zeros((len(mn.unary_potentials[edge[0]]), (len(mn.unary_potentials[edge[1]]))))
            table[states[edge[0]], states[edge[1]]] = 1
            tmp = np.asarray(table.reshape((-1, 1)))
            edge_sufficient_stats += tmp
            edge_padded_sufficient_stats += padded_tmp
        return edge_sufficient_stats, edge_padded_sufficient_stats
