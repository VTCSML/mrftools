from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
# from grafting_util import initialize_priority_queue, reset_unary_factors, reset_edge_factors
from graph_mining_util import make_graph, select_edge_to_inject
import copy
import operator
from graph_mining_util import draw_graph, make_graph
import scipy.sparse as sps
import itertools
import networkx as nx
import os
import random
from scipy.misc import logsumexp
from grafting_util import compute_likelihood

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# + Functionality:
#   Pairwise MRF stracture learning using classical grafting
#
# + TODO : add logs and reference
# 
# + Author: (walidch)
# + Reference : 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Graft():
    """
    Grafting class
    """
    def __init__(self, variables, num_states, max_num_states, data, list_order):
        """
        Initialize Graft class
        """
        self.variables = variables
        self.num_states = num_states
        self.max_num_states = max_num_states
        self.mn = MarkovNet()
        self.mn.initialize_unary_factors(variables, num_states)
        # self.search_space = self.mn.search_space
        self.search_space = [self.mn.search_space[i] for i in list_order]
        self.data = data
        self.sufficient_stats, self.padded_sufficient_stats = self.mn.get_unary_sufficient_stats(self.data , self.max_num_states)
        self.l1_coeff = 0
        self.l2_coeff = 0
        self.node_l1 = 0
        self.edge_l1 = .1
        self.max_iter_graft = 500
        self.active_set = []
        self.edge_regularizers, self.node_regularizers = dict(), dict()
        self.graph_snapshots, self.mn_snapshots = dict(), dict()
        self.is_show_metrics = False
        self.is_verbose = False
        self.is_monitor_mn = False
        self.is_converged = False
        self.is_limit_suffstats = False
        self.is_remove_zero_edges = False
        self.zero_threshold = 1e-2
        self.is_synthetic = False
        self.graph = nx.Graph()
        for var in self.variables:
            self.graph.add_node(var)

    def on_limit_sufficient_stats(self, max_sufficient_stats_ratio):
        """
        Reduce search space by selecting a random subset of edges
        """
        self.is_limit_suffstats = True
        self.search_space = [self.mn.search_space[i] for i in sorted(random.sample(xrange(len(self.mn.search_space)), int(max_sufficient_stats_ratio * len(self.mn.search_space))))]

    def on_zero_treshold(self, zero_threshold=1e-2):
        """
        Set a limit for the maximum amount of sufficient statistics to be computed
        """
        self.is_remove_zero_edges = True
        self.zero_threshold = zero_threshold

    def setup_learning_parameters(self, edge_l1, node_l1=0, l1_coeff=0, l2_coeff=0, max_iter_graft=500):
        """
        Set grafting parameters
        """
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.node_l1 = node_l1
        self.edge_l1 = edge_l1
        self.max_iter_graft = max_iter_graft

    def on_synthetic(self, precison_threshold = .7):
        self.is_synthetic = True
        self.precison_threshold = precison_threshold


    def on_monitor_mn(self):
        """
        Enable monitoring Markrov net
        """
        self.is_monitor_mn = True
        self.mn_snapshots = dict()

    def on_show_metrics(self):
        self.is_show_metrics = True


    def on_verbose(self):
        self.is_verbose = True

    def save_mn(self, exec_time=0):
        learned_mn = copy.deepcopy(self.aml_optimize.belief_propagators[0].mn)
        learned_mn.load_factors_from_matrices()
        self.mn_snapshots[exec_time] = learned_mn

    def save_graph(self, exec_time=0):
        graph = copy.deepcopy(self.graph)
        self.graph_snapshots[exec_time] = graph

    def set_regularization_indices(self, unary_indices, pairwise_indices):
        for node in self.variables:
            self.node_regularizers[node] = (unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]])
        for edge in self.active_set:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]

    def reinit_weight_vec(self, unary_indices, pairwise_indices, weights_opt, vector_length_per_edge):
        tmp_weights_opt = np.random.randn(len(weights_opt) + vector_length_per_edge)
        for edge in self.active_set:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]
            try:
                tmp_weights_opt[self.edge_regularizers[edge]] = weights_opt[old_edge_regularizers[edge]]
            except:
                pass
        # self.node_regularizers=[]
        for node in self.variables:
            # self.node_regularizers.extend(unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]])
            self.node_regularizers[node] = unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]]
        try:
            tmp_weights_opt[self.node_regularizers] = weights_opt[old_node_regularizers]
        except:
            pass
        old_edge_regularizers = copy.deepcopy(self.edge_regularizers)
        old_node_regularizers = copy.deepcopy(self.node_regularizers)

        return tmp_weights_opt, old_node_regularizers, old_edge_regularizers

    def learn_structure(self, num_edges, edges=list()):
        """
        Main function for grafting
        """
        # INITIALIZE VARIABLES
        self.edges = edges
        data_len = len(self.data)
        np.random.seed(0)
        if self.is_monitor_mn:
            exec_time_origin = time.time()
        self.aml_optimize = self.setup_grafting_learner(len(self.data))

        for edge in self.search_space:
            self.sufficient_stats[edge], self.padded_sufficient_stats[edge] = self.get_sufficient_stats_per_edge(self.mn, edge)

        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        self.set_regularization_indices(unary_indices, pairwise_indices)

        old_node_regularizers = self.node_regularizers
        old_edge_regularizers = self.edge_regularizers


        vector_length_per_var = self.max_num_states
        vector_length_per_edge = self.max_num_states ** 2
        len_search_space = len(self.search_space)

        if self.is_show_metrics:
            recall, precision, suff_stats_list, f1_score, objec  = [0,0], [0,0], [0,0], [0,0], []

        tmp_weights_opt = np.random.randn(self.aml_optimize.weight_dim)
        weights_opt = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, data_len, verbose=False, loss=objec)
        self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)

        objec.extend(objec)

        if self.is_monitor_mn:
            exec_time =  time.time() - exec_time_origin
            self.save_mn()
            self.save_mn(exec_time=exec_time)
            self.save_graph(exec_time=exec_time)


        is_activated_edge, activated_edge = self.activation_test()

        while (len(self.search_space) > 0 and is_activated_edge) and len(self.active_set) < num_edges: # Stop if all edges are added or no edge is added at the previous iteration
            
            
            self.active_set.append(activated_edge)
            if self.is_verbose:
                self.print_update(activated_edge)

            # draw_graph(active_set, variables)
            # self.mn.set_edge_factor(activated_edge, np.ones((len(self.mn.unary_potentials[activated_edge[0]]), len(self.mn.unary_potentials[activated_edge[1]]))))
            
            self.graph.add_edge(activated_edge[0], activated_edge[1])

            self.aml_optimize = self.setup_grafting_learner(len(self.data))

            unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
            self.set_regularization_indices(unary_indices, pairwise_indices)
            tmp_weights_opt, old_node_regularizers, old_edge_regularizers= self.reinit_weight_vec(unary_indices, pairwise_indices, weights_opt, vector_length_per_edge)

            weights_opt = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, data_len, verbose=False, loss=objec)
            self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)

            if self.is_monitor_mn:
                exec_time = time.time() - exec_time_origin
                self.save_mn(exec_time=exec_time)
                self.save_graph(exec_time=exec_time)

            if self.is_show_metrics and is_activated_edge:
                self.update_metrics(edges, recall, precision, f1_score, suff_stats_list)

            if self.is_synthetic and precision[-1] < self.precison_threshold and len(self.active_set) > 3:
                return learned_mn, None, suff_stats_list, recall, precision, f1_score, objec, True

            is_activated_edge, activated_edge = self.activation_test()


        if is_activated_edge == False:
                self.is_converged = True


        final_active_set = self.active_set

        if self.is_remove_zero_edges:
            # REMOVE NON RELEVANT EDGES
            final_active_set = self.remove_zero_edges()
            self.active_set = final_active_set
            self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)
            if self.is_monitor_mn:
                exec_time = time.time() - exec_time_origin
                self.save_mn(exec_time=exec_time)
            if self.is_show_metrics:
                self.update_metrics(edges, recall, precision, f1_score, suff_stats_list)


        # draw_graph(active_set, variables)

        learned_mn = self.aml_optimize.belief_propagators[0].mn
        self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)
        learned_mn.load_factors_from_matrices()
        if self.is_show_metrics:
            self.print_metrics(recall, precision, f1_score)
            return learned_mn, final_active_set, suff_stats_list, recall, precision, f1_score, objec, False
        if self.is_verbose:
            print('Final Active Set')
            print(final_active_set)

        return learned_mn, final_active_set, None, None, None, None, objec, False


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


    def activation_test(self):
        """
        Fonctionality:
        1 - Compute the gradient for the current weight vector 
        """
        edge_mean_weights = []
        gradient_vec = []
        map_vec = []
        bp = self.aml_optimize.belief_propagators[0]
        bp.load_beliefs()
        is_activated = False
        activated_edge = None
        for edge in self.search_space:
            belief = bp.var_beliefs[edge[0]] + np.matrix(bp.var_beliefs[edge[1]]).T
            belief = belief - logsumexp(belief)
            gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(self.sufficient_stats[edge]) / len(self.data)).squeeze()
            length_normalizer = np.sqrt(len(gradient))
            gradient_norm = np.sqrt(gradient.dot(gradient)) / length_normalizer
            # if edge in self.edges:
                # print(edge)
                # print(gradient_norm)
            gradient_vec.append(gradient_norm)
            map_vec.append(edge)
            # print(gradient_vec)
            # print(np.array(gradient_vec))
        if len(self.search_space) > 1:
            max_ind = np.array(gradient_vec).argmax(axis=0)
            max_grad = max(gradient_vec)
        else:
            max_ind = 0
            max_grad = gradient_vec[0]
        if max_grad > self.edge_l1:
            is_activated = True
            activated_edge = map_vec[max_ind]
            self.search_space.remove(activated_edge)
            # print('+')
            # print(activated_edge)
            # print(max_grad)
        return is_activated, activated_edge


    def print_update(self, activated_edge):
        """
        print update
        """
        print('ACTIVATED EDGE')
        print(activated_edge)
        print('CURRENT ACTIVE SPACE')
        print(self.active_set)

    def update_metrics(self, edges, recall, precision, f1_score, suff_stats_list):
        """
        UPDATE METRICS
        """
        curr_recall = float(len([x for x in self.active_set if x in edges]))/len(edges)
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
        
        suff_stats_list.append((len(self.sufficient_stats) - len(self.variables)))

    def remove_zero_edges(self):
        """
        Filter out near zero edges
        """
        bp = self.aml_optimize.belief_propagators[0]
        self.aml_optimize.belief_propagators[0].mn.update_edge_tensor()
        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        final_active_set = list()
        for edge in self.active_set:
            # length_normalizer = float(1) / (len(bp.mn.unary_potentials[edge[0]]) * len(bp.mn.unary_potentials[edge[1]]))
            i = self.aml_optimize.belief_propagators[0].mn.edge_index[edge]
            edge_weights = self.aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:self.aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :self.aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
            if np.sqrt(edge_weights.dot(edge_weights)) / np.sqrt(len(edge_weights))  > self.zero_threshold:
                final_active_set.append(edge)
        return final_active_set

    def print_metrics(self, recall, precision, f1_score):
        """
        Print metrics
        """
        print('SUFFICIENT STATS')
        print(len(self.sufficient_stats) - len(self.variables))
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
