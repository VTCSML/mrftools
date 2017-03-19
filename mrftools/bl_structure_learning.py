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
import random

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


class L1_learner():
    """
    L1_learner class
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
        self.search_space = [self.mn.search_space[i] for i in list_order]
        self.data = data
        self.sufficient_stats, self.padded_sufficient_stats = self.mn.get_unary_sufficient_stats(self.data , self.max_num_states)
        self.l1_coeff = 0
        self.l2_coeff = 0
        self.node_l1 = 0
        self.edge_l1 = .1
        self.zero_threshold = 1e-2
        self.max_iter_graft = 500
        self.active_set = []
        self.edge_regularizers, self.node_regularizers = dict(), dict()
        self.is_show_metrics = False
        self.is_verbose = False
        self.is_monitor_mn = False
        self.is_converged = False
        self.is_limit_suffstats = False

    def on_limit_sufficient_stats(self, max_sufficient_stats_ratio):
        """
        Reduce search space by selecting a random subset of edges
        """
        self.is_limit_suffstats = True
        # print('Initial search space')
        # print(self.search_space)
        self.search_space = [self.mn.search_space[i] for i in sorted(random.sample(xrange(len(self.mn.search_space)), int(max_sufficient_stats_ratio * len(self.mn.search_space))))]

    def setup_learning_parameters(self, edge_l1, node_l1=0, l1_coeff=0, l2_coeff=0, max_iter_graft=500, zero_threshold=.05):
        """
        Set grafting parameters
        """
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.node_l1 = node_l1
        self.edge_l1 = edge_l1
        self.max_iter_graft = max_iter_graft
        self.zero_threshold = zero_threshold

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

    def learn_structure(self, edges=list()):
        """
        Main function for grafting
        """
        # INITIALIZE VARIABLES

        if self.is_monitor_mn:
            exec_time_origin = time.time()

        self.aml_optimize = self.setup_learner()

        if self.is_monitor_mn:
            learned_mn = self.aml_optimize.belief_propagators[0].mn
            learned_mn.load_factors_from_matrices()
            exec_time = time.time() - exec_time_origin
            self.mn_snapshots[exec_time] = learned_mn

        for edge in self.search_space:
            self.sufficient_stats[edge], self.padded_sufficient_stats[edge] = self.get_sufficient_stats_per_edge(self.mn, edge)
            self.mn.set_edge_factor(edge, np.zeros((len(self.mn.unary_potentials[edge[0]]), len(self.mn.unary_potentials[edge[1]]))))

        if self.is_show_metrics:
            recall, precision, suff_stats_list = list(), list(), list()

        np.random.seed(0)
        vector_length_per_var = self.max_num_states
        vector_length_per_edge = self.max_num_states ** 2
        len_search_space = len(self.search_space)
        weights_opt = self.aml_optimize.learn(np.random.randn(self.aml_optimize.weight_dim), self.max_iter_graft, self.edge_regularizers, self.node_regularizers)

        # REMOVE NON RELEVANT EDGES
        final_active_set = self.remove_zero_edges()


        learned_mn = self.aml_optimize.belief_propagators[0].mn
        learned_mn.load_factors_from_matrices()


        if self.is_show_metrics:
            recall, precision = self.get_metrics(final_active_set)
            return learned_mn, final_active_set, suff_stats_list, recall, precision

        if self.is_verbose:
            print('Final Active Set')
            print(final_active_set)

        return learned_mn, final_active_set, None, None, None


    def setup_learner(self):
        """
        Initialize learner with training data
        """
        aml_optimize = ApproxMaxLikelihood(self.mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(self.l1_coeff, self.l2_coeff, self.node_l1, self.edge_l1)
        aml_optimize.init_grafting()
        unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        tau_q = np.zeros(aml_optimize.weight_dim)
        for var in self.mn.variables:
            i = aml_optimize.belief_propagators[0].mn.var_index[var]
            inds = unary_indices[:, i]
            tau_q[inds] = self.padded_sufficient_stats[var] / len(self.data)
        for edge in self.active_set:
            i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
            inds = pairwise_indices[:, :, i]
            tau_q[inds] = self.padded_sufficient_stats[edge] / len(self.data)
        aml_optimize.set_sufficient_stats(tau_q)
        return aml_optimize

    def get_metrics(self, edges):
        """
        UPDATE METRICS
        """

        try:
            recall = float(len([x for x in self.active_set if x in edges]))/len(edges)
        except:
            recall = 0
        try:
            precision = float(len([x for x in edges if x in self.active_set]))/len(self.active_set)
        except:
            precision = 0

        return recall, precision

    def remove_zero_edges(self):
        """
        Filter out near zero edges
        """
        bp = self.aml_optimize.belief_propagators[0]
        self.aml_optimize.belief_propagators[0].mn.update_edge_tensor()
        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        final_active_set = list()
        for edge in self.active_set:
            length_normalizer = float(1) / (len(bp.mn.unary_potentials[edge[0]]) * len(bp.mn.unary_potentials[edge[1]]))
            i = self.aml_optimize.belief_propagators[0].mn.edge_index[edge]
            edge_weights = self.aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:self.aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :self.aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
            if length_normalizer * np.sqrt(edge_weights.dot(edge_weights))  > self.zero_threshold:
                final_active_set.append(edge)
        return final_active_set


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
