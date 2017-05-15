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
import operator

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

class SelectiveStructuredPriorityGraft():
    """
    Structured priority grafting class
    """
    def __init__(self, variables, num_states, max_num_states, data, list_order, method='structured', ss_test=dict(), pq_dict = None):
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
        self.sufficient_stats, self.padded_sufficient_stats = self.mn.get_unary_sufficient_stats(self.data , self.max_num_states)
        if pq_dict == None:
            self.pq = initialize_priority_queue(search_space=self.search_space)
        else:
            self.pq = pq_dict
        self.search_space = set(self.search_space)
        self.edges_list = list()
        if method == 'queue':
            while len(self.pq) > 0:
                item = self.pq.popitem()
                self.edges_list = [item] + self.edges_list
        self.method = method
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
        self.not_connected = dict()
        for var in self.variables:
            self.graph.add_node(var)
            self.frozen[var] = list()
            self.not_connected[var] = True
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
        self.select_unit = 5
        self.added_edge_index = None
        self.is_grow_k = False
        self.m = 5
        self.total_reservoir_weight = 0
        self.is_reservoir_full = False
        self.max_update_step = 50
        self.current_update_step = 0
        self.is_shrink_k = False
        self.alpha = .99
        self.snapshot_count = 0

        self.treated_hub = dict()

        self.updated_nodes = dict()

    def on_structured(self):
        self.structured = True

    def set_alpha(self, alpha=.9):
        self.alpha = alpha

    def set_max_update_step(self, max_update_step=50):
        self.max_update_step = max_update_step

    def set_reassigned_nodes(self, m=10):
        self.m = m

    def on_grow_k(self):
        self.is_grow_k = True

    def on_shrink_k(self):
        self.is_shrink_k = True

    def set_top_relvant(self, k=5):
        self.k = k
        self.select_unit = min(self.k, self.select_unit)

    def set_select_unit(self, select_unit=5):
        self.select_unit = select_unit
        self.select_unit = min(self.k, self.select_unit)

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
        MAX_SNAPSHOTS = 200
        learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
        learned_mn.load_factors_from_matrices()
        if exec_time != 0:
            self.snapshot_count += 1
        # hacked to do reservoir sampling
        if exec_time==0 or len(self.mn_snapshots) < MAX_SNAPSHOTS:
            self.mn_snapshots[exec_time] = learned_mn
        else:
            # print "Snapshot reservior full."
            if random.randint(0, self.snapshot_count) <= MAX_SNAPSHOTS:
                to_replace = random.choice(self.mn_snapshots.keys())
                while to_replace == 0:
                    to_replace = random.choice(self.mn_snapshots.keys())
                # print "Replacing snapshot from time " + repr(to_replace)
                # print "\t with snapshot from time " + repr(exec_time)
                del(self.mn_snapshots[to_replace])
                self.mn_snapshots[exec_time] = learned_mn

                # print "Current snapshots: "
                # print sorted(self.mn_snapshots.keys())


    def save_graph(self, exec_time=0):
        graph = copy.deepcopy(self.graph)
        self.graph_snapshots[exec_time] = graph

    def set_regularization_indices(self, unary_indices, pairwise_indices):
        for node in self.variables:
            self.node_regularizers[node] = (unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]])
        for edge in self.active_set:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]

    def reinit_weight_vec(self, unary_indices, pairwise_indices, weights_opt, vector_length_per_edge, old_node_regularizers, old_edge_regularizers ,  num_added_edges):

        tmp_weights_opt = np.zeros(len(weights_opt) + num_added_edges * vector_length_per_edge)
        for edge in self.active_set:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]
            try:
                tmp_weights_opt[list(self.edge_regularizers[edge].flatten())] = weights_opt[list(old_edge_regularizers[edge].flatten())]
            except:
                self.added_edge_index = list(self.edge_regularizers[edge].flatten())
        if len(old_node_regularizers) > 0:
            for node in self.variables:
                self.node_regularizers[node] = unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]]
                try:
                    tmp_weights_opt[list(self.node_regularizers[node])] = weights_opt[list(old_node_regularizers[node])]
                except:
                    pass
        old_edge_regularizers = copy.deepcopy(self.edge_regularizers)
        old_node_regularizers = copy.deepcopy(self.node_regularizers)

        return tmp_weights_opt, old_node_regularizers, old_edge_regularizers

    def continue_graft(self, break_grafting, is_activated_edge, num_edges):
        return (not break_grafting) and (len(self.pq) > 0 or is_activated_edge or len(self.k_relevant)>0 ) and \
                len(self.active_set) < num_edges and self.is_limit_sufficient_stats_reached()

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
        weights_opt, tmp_metric_exec_time = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, self.data, verbose=False, loss=objec)
        metric_exec_time += tmp_metric_exec_time

        objec.extend(objec)

        if self.is_monitor_mn:
            exec_time =  time.time() - exec_time_origin - metric_exec_time
            self.save_mn()
            self.save_mn(exec_time=exec_time)
            # self.save_graph(exec_time=exec_time)

        if self.is_plot_queue:
            columns, rows, values = list(), list(), list()
            loop_num = 0
            pq_history = dict()

        is_activated_edge, activated_edges_list, iter_number = self.activation_test()

        old_edge_regularizers = list()

        break_grafting = False

        while self.continue_graft(break_grafting, is_activated_edge, num_edges):

            
            if len(self.pq) == 0 and not is_activated_edge:
                if num_edges < float('inf'):
                    activated_edges_list = [x for x in sorted(self.k_relevant, key=self.k_relevant.get, reverse=True)[ : num_edges - len(self.active_set)]]
                if num_edges == float('inf'):
                    print('damping reservoir')
                    print(len(self.frozen_list))
                    activated_edges_list = [x for x in sorted(self.k_relevant, key=self.k_relevant.get, reverse=True)]

                break_grafting = True

            for activated_edge in activated_edges_list:
                if self.is_grow_k:
                    self.k += 1
                if self.is_shrink_k:
                    self.k -= 1
                self.active_set.append(activated_edge)
                self.graph.add_edge(activated_edge[0], activated_edge[1])
                self.not_connected[activated_edge[0]] = False
                self.not_connected[activated_edge[1]] = False

            print(is_activated_edge)
            print('active edges')
            print(len(self.active_set))

            if self.structured and len(self.active_set) > 15:
                self.update_pq()

            if self.is_verbose:
                self.print_update(activated_edges_list)

            if self.is_synthetic and precision[-1] < self.precison_threshold and len(self.active_set) > self.start_num:
                learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
                learned_mn.load_factors_from_matrices()
                return learned_mn, None, suff_stats_list, recall, precision, f1_score, objec, True

            self.aml_optimize = self.setup_grafting_learner(len(self.data))
            unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
            self.set_regularization_indices(unary_indices, pairwise_indices)
            tmp_weights_opt, old_node_regularizers, old_edge_regularizers= self.reinit_weight_vec(unary_indices, pairwise_indices, weights_opt, vector_length_per_edge, old_node_regularizers, old_edge_regularizers, len(activated_edges_list))
            weights_opt, tmp_metric_exec_time = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, self.data, verbose=False, loss=objec)
            metric_exec_time += tmp_metric_exec_time

            if self.is_monitor_mn:
                exec_time = time.time() - exec_time_origin - metric_exec_time
                self.save_mn(exec_time=exec_time)
                # self.save_graph(exec_time=exec_time)
            self.total_iter_num.append(iter_number)

            if self.is_show_metrics:
                self.update_metrics(edges, recall, precision, f1_score, suff_stats_list)
                if f1_score[-1] >= .7:
                    self.ss_at_70 = (len(self.sufficient_stats) - len(self.variables)) / len(self.initial_search_space)
                    is_ss_at_70_regeistered = True
            
            is_activated_edge, activated_edges_list, iter_number = self.activation_test()


        self.sufficient_stats.clear()
        self.padded_sufficient_stats.clear()
        self.edge_regularizers.clear()
        self.node_regularizers.clear()

        if is_activated_edge == False:
                self.is_converged = True

        final_active_set = self.active_set

        if self.is_remove_zero_edges:
            # REMOVE NON RELEVANT EDGES
            final_active_set = self.remove_zero_edges()
            self.active_set = final_active_set

        if self.is_plot_queue:
            self.make_queue_plot_synthetic_truth(pq_history, final_active_set, loop_num, len_search_space)

        learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
        learned_mn.load_factors_from_matrices()

        if self.is_show_metrics:
            self.print_metrics(recall, precision, f1_score)
            return learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, False

        if self.is_verbose:
            print('Final Active Set')
            print(final_active_set)

        return learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, False

    def update_k_relevant(self, bp):
        #Update relevant edges
        dropped = 0
        update = False
        # print('updated')
        # print(self.updated_nodes)
        for rel_edge in self.k_relevant.keys():
            for node in self.updated_nodes.keys():
                if nx.has_path(self.graph, node, rel_edge[0]) or nx.has_path(self.graph, node, rel_edge[0]):
                    update = True
                    break
            if not update:       
                continue
            else:
                sufficient_stat_edge = self.sufficient_stats[rel_edge]
                unormalized_belief = bp.unormalized_var_beliefs[rel_edge[0]] + np.matrix(bp.unormalized_var_beliefs[rel_edge[1]]).T
                belief = unormalized_belief - logsumexp(unormalized_belief)
                gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(sufficient_stat_edge) / len(self.data)).squeeze()
                gradient_norm = np.sqrt(gradient.dot(gradient))
                length_normalizer = np.sqrt(len(gradient))
                passed = (gradient_norm / length_normalizer) > self.edge_l1
                if not passed:
                    dropped += 1
                    direct_penalty = 1 - gradient_norm/(length_normalizer * self.edge_l1)
                    self.frozen_list.append((rel_edge, direct_penalty))
                    self.k_relevant.pop(rel_edge, None)
                else:
                    self.k_relevant[rel_edge] = (gradient_norm / length_normalizer)
        # print('reservoir')
        # print(self.k_relevant)

    # def update_pq(self):
    #     if self.m > 0:
    #         self.centrality = nx.degree_centrality(self.graph)
    #         # self.centrality = nx.betweenness_centrality(self.graph)
    #         self.m_central_nodes = list()
    #         for i in range(self.m):
    #             node = max(self.centrality.iteritems(), key=operator.itemgetter(1))[0]
    #             self.centrality[node] = -1
    #             self.m_central_nodes.append(node)
            
    #         for node_1 in self.m_central_nodes:
    #             for node_2 in self.m_central_nodes:
    #                 if (node_1,node_2) in self.pq:
                        # self.pq.updateitem((node_1,node_2), self.pq[(node_1,node_2)] - 1)


    def update_pq(self):
        if self.m > 0:
            self.centrality = nx.degree_centrality(self.graph)
            # self.centrality = self.graph.degree

            # sorted_centrality= sorted(self.centrality.items(), key=operator.itemgetter(1))

            max_score = max(self.centrality.iteritems(), key=operator.itemgetter(1))[1]
            mean_score = np.mean(self.centrality.values())
            alpha = 0
            threshold = (1 - alpha) * mean_score + alpha * max_score

            # for i in range(1, len(sorted_centrality)):
            for i in range(1, len(self.variables)):

                (node, node_score) = max(self.centrality.iteritems(), key=operator.itemgetter(1))
                # node, node_score = sorted_centrality[-i]
                if node_score > threshold:
                    self.centrality.pop(node, None)
                    hub = node
                    hub_score = node_score
                    if hub not in self.treated_hub:
                        self.treated_hub[hub] = True
                        for node in self.variables: # prioritize edges for hubs
                            # node_score = self.centrality[node]
                            # if self.graph.degree(node) < 2:
                            edge = (min(hub,node),max(hub,node))
                            # print('prioritize')
                            # print(edge in self.edges)
                            if edge in self.pq:
                                self.pq.updateitem(edge, self.pq[edge] -1)
                else: break

                # elif node_score < .01:
                #     leaf = node
                #     leaf_score = node_score
                #     # if hub not in self.treated_hub:
                #         # self.treated_hub[hub] = True
                #     for node in self.variables: # prioritize edges for hubs
                #         node_score = self.centrality[node]
                #         if node_score<.01:
                #             edge = (min(leaf,node),max(leaf,node))
                #             print('deprioritize')
                #             print(edge in self.edges)
                #             if edge in self.pq:
                #                 # self.pq.updateitem(edge, max(self.pq[edge], 1 - leaf_score))
                #                 self.pq.updateitem(edge, max(self.pq[edge], +1))


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

        if len(self.pq) == 0 and len(self.frozen_list) > 1:
                self.is_added = False
                print('refilling')
                for frozen_items in self.frozen_list:
                    self.pq.additem(frozen_items[0], frozen_items[1])
                refill = True
                self.frozen_list = list()

        while len(self.pq)>0:
            while len(self.pq)>0:
                # print('pq')
                # print(len(self.pq))
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
                unormalized_belief = bp.unormalized_var_beliefs[edge[0]] + np.matrix(bp.unormalized_var_beliefs[edge[1]]).T
                belief = unormalized_belief - logsumexp(unormalized_belief)
                gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(sufficient_stat_edge) / len_data).squeeze()
                grad = np.abs(gradient) - self.l1_coeff
                if self.l1_coeff > 0:
                    grad = np.abs(gradient) - self.l1_coeff
                    grad[grad<0] = 0
                    grad *= np.sign(gradient) 
                    gradient = grad
                gradient_norm = np.sqrt(gradient.dot(gradient))
                length_normalizer = np.sqrt(len(gradient))
                passed = (gradient_norm / length_normalizer) > self.edge_l1
                if passed:
                    if not self.is_added:
                            self.is_added = True 
                    self.k_relevant[edge] = gradient_norm / length_normalizer
                    if len(self.k_relevant) == self.k + 1:
                        dropped_edge = min(self.k_relevant.iteritems(), key=operator.itemgetter(1))[0] #DROP THE ITEM
                        direct_penalty = 1 - self.k_relevant[dropped_edge] / self.edge_l1
                        self.frozen_list.append((dropped_edge, direct_penalty))
                        self.k_relevant.pop(dropped_edge, None)
                        self.is_reservoir_full = True
                else:
                    direct_penalty = 1 - gradient_norm/(length_normalizer * self.edge_l1)
                    self.frozen_list.append((edge, direct_penalty))

                if iteration_activation == self.max_update_step:
                    if len(self.k_relevant) > 0:
                        self.current_update_step = 0
                        selected_edges_list = list()
                        self.updated_nodes = dict()
                        max_weight = max(self.k_relevant.iteritems(), key=operator.itemgetter(1))[1]
                        mean_weight = np.mean(self.k_relevant.values())
                        threshold = (1 - self.alpha) * mean_weight + self.alpha * max_weight
                        is_threshold_reached = False
                        while not is_threshold_reached and len(self.k_relevant) > 0:
                            (selected_edge, selected_weight) = max(self.k_relevant.iteritems(), key=operator.itemgetter(1))
                            if selected_weight >= threshold and selected_edge[0] not in self.updated_nodes and selected_edge[1] not in self.updated_nodes:
                                self.updated_nodes[selected_edge[0]] = True
                                self.updated_nodes[selected_edge[1]] = True
                                self.k_relevant.pop(selected_edge, None)
                                self.search_space.remove(selected_edge)
                                selected_edges_list.append(selected_edge)
                            else:
                                is_threshold_reached = True
                                # self.k -= 1
                        # print('activate')
                        # print(len(self.pq))
                        # print(len(self.frozen_list))
                        return True, selected_edges_list, iteration_activation
                    else:
                        iteration_activation = 0
            print('Done')
            print(self.is_added)
            if self.is_added:
                self.is_added = False
                print('refilling')
                for frozen_items in self.frozen_list:
                    self.pq.additem(frozen_items[0], frozen_items[1])
                refill = True
                self.frozen_list = list()
        return False, [(0, 0)], iteration_activation


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


    def make_queue_plot_ground_truth(self, values, rows, columns, loop_num, len_search_space):
        """
        Plot queue reorganization using ground truth edges
        """
        view_queue_tmp = sps.csr_matrix((np.array(values), (np.array(rows), np.array(columns))) , (loop_num + 1, len_search_space))
        view_queue_tmp = view_queue_tmp.todense()
        view_queue_tmp[view_queue_tmp==0] = .2
        view_queue = np.zeros((loop_num+2, len_search_space+2))
        view_queue[1:loop_num+1, 1:len_search_space+1] = view_queue_tmp
        plt.imshow(view_queue,interpolation='none',cmap='binary', aspect='auto')
        plt.title(self.method + ' Ground truth')
        # plt.colorbar()
        plt.axis('off')
        file_name = 'truth_' + self.method + str(len(self.variables)) +'_Nodes.png'
        plt.savefig(os.path.join(self.plot_path, file_name))
        plt.close()

    def make_queue_plot_synthetic_truth(self, pq_history, final_active_set, loop_num, len_search_space):
        """
        Plot queue reorganization using learned truth edges
        """
        plt.close()
        columns = list()
        rows = list()
        values = list()
        for t in range(len(pq_history)):
            copy_pq = pq_history[t+1]
            for t1 in range(t):
                columns.append(t1)
                rows.append(t)
                values.append(.5)
            len_pq = len(copy_pq)
            for c in range(len_pq):
                if self.method == 'structured':
                    test_edge = copy_pq.popitem()[0]
                if self.method == 'queue':
                    test_edge = copy_pq.pop()[0]
                if test_edge in final_active_set:
                    columns.append(c + t)
                    rows.append(t )
                    values.append(1)
        view_queue_tmp = sps.csr_matrix((np.array(values), (np.array(rows), np.array(columns))) , (loop_num, len_search_space))
        view_queue_tmp = view_queue_tmp.todense()
        view_queue_tmp[view_queue_tmp==0] = .2
        view_queue = np.zeros((loop_num+2, len_search_space+2))
        view_queue[1:loop_num+1, 1:len_search_space+1] = view_queue_tmp
        plt.imshow(view_queue, interpolation='none',cmap='binary', aspect='auto')
        plt.title(self.method + ' Synthetic truth')
        # plt.colorbar()
        plt.axis('off')
        file_name = 'synth_' + self.method + str(len(self.variables)) +'_Nodes.png'
        plt.savefig(os.path.join(self.plot_path, file_name))
        plt.close()

    def update_plot_info(self, loop_num, columns, rows, values, pq_history, edges):
        """
        update queue plot realted information
        """
        loop_num += 1
        for t in range(len(self.active_set)):
            columns.append(t)
            rows.append(len(self.active_set))
            values.append(.5)
        if self.method == 'queue':
            copy_pq = copy.deepcopy(self.edges_list)
        else:
            copy_pq = copy.deepcopy(self.pq)
        pq_history[loop_num] = copy.deepcopy(copy_pq)
        len_pq = len(copy_pq)
        for c in range(len_pq):
            if self.method == 'queue':
                test_edge = copy_pq.pop()[0]
            else:
                test_edge = copy_pq.popitem()[0]
            if test_edge in edges:
                columns.append(c + len(self.active_set) )
                rows.append(len(self.active_set))
                values.append(1)
        return loop_num

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
            curr_recall = float(len([x for x in set(self.active_set) if x in edges]))/len(edges)
        except:
            curr_recall = 0
        recall.append(curr_recall)
        try:
            curr_precision = float(len([x for x in edges if x in set(self.active_set)]))/len(set(self.active_set))
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
        for edge in self.active_set:
            i = self.aml_optimize.belief_propagators[0].mn.edge_index[edge]
            edge_weights = self.aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:self.aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :self.aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
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
