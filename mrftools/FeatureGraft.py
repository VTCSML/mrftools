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


class FeatureGraft():
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
        self.is_show_metrics = False
        self.is_verbose = False
        self.is_monitor_mn = False
        self.is_converged = False
        self.is_limit_suffstats = False
        self.is_remove_zero_edges = False
        self.zero_threshold = 1e-2
        self.zero_feature_indices = list()
        self.active_features = list()
        self.relevant_features = list()
        self.is_synthetic = False
        self.is_full_l1 = False

    def on_full_l1(self):
        self.is_full_l1 = True

    def on_synthetic(self, precison_threshold = .7):
        self.is_synthetic = True
        self.precison_threshold = precison_threshold

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

    def setup_learning_parameters(self, edge_l1=0, node_l1=0, l1_coeff=0, l2_coeff=0, max_iter_graft=500):
        """
        Set grafting parameters
        """
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.node_l1 = node_l1
        self.edge_l1 = edge_l1
        self.max_iter_graft = max_iter_graft


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

    def set_regularization_indices(self, unary_indices, pairwise_indices):
        for node in self.variables:
            self.node_regularizers[node] = (unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]])
        for edge in self.active_set:
            self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]


    def get_edge_indices(self, pairwise_indices):
        self.feature2edge = dict()
        for edge in self.search_space:
            edge_indices = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]].tolist()
            edge_indices = [index for indices in edge_indices for index in indices]
            self.feature2edge.update({key: edge for key in edge_indices})
            self.zero_feature_indices.extend(edge_indices)



    def learn_structure(self, num_features, edges=list()):
        """
        Main function for grafting
        """
        # INITIALIZE VARIABLES
        data_len = len(self.data)
        self.edges = edges
        np.random.seed(0)
        if self.is_monitor_mn:
            exec_time_origin = time.time()

        for activated_edge in self.search_space:
            self.mn.set_edge_factor(activated_edge, np.zeros((len(self.mn.unary_potentials[activated_edge[0]]), len(self.mn.unary_potentials[activated_edge[1]]))))
            self.active_set.append(activated_edge)


        for edge in self.search_space:
            self.sufficient_stats[edge], self.padded_sufficient_stats[edge] = self.get_sufficient_stats_per_edge(self.mn, edge)
        self.aml_optimize = self.setup_grafting_learner()

        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        self.get_edge_indices(pairwise_indices)

        for feature in self.feature2edge.keys():
            if self.feature2edge[feature] in self.edges:
                self.relevant_features.append(feature)

        if self.is_monitor_mn:
            self.save_mn()

        weights_opt = np.random.randn(self.aml_optimize.weight_dim)

        if self.is_full_l1:
            self.zero_feature_indices = []

        weights_opt[self.zero_feature_indices] = 0

        weights_opt, activated_feature = self.aml_optimize.learn(weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, data_len, verbose=True, is_feature_graft=True, zero_feature_indices = self.zero_feature_indices)
        self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)

        if self.is_monitor_mn:
            exec_time =  time.time() - exec_time_origin
            self.save_mn()
            self.save_mn(exec_time=exec_time)

        if self.is_show_metrics:
            recall, precision, f1_score = [0,0], [0,0], [0,0]


        while activated_feature and len(self.active_features) < num_features:

            self.active_features.append(activated_feature)
            
            self.active_set.append(self.feature2edge[activated_feature])

            self.active_set = list(set(self.active_set))

            if self.is_show_metrics and activated_feature:
                    self.update_metrics(edges, recall, precision, f1_score)

            if self.is_verbose:
                self.print_update(activated_feature, precision[-1])

            if self.is_synthetic and precision[-1] < self.precison_threshold and len(self.active_features) > 5:
                return learned_mn, self.active_set, recall, precision, f1_score, True

            weights_opt[self.zero_feature_indices] = 0
            weights_opt[activated_feature] = 0
            weights_opt, activated_feature = self.aml_optimize.learn(weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers, data_len, verbose=True, is_feature_graft=True, zero_feature_indices = self.zero_feature_indices)
            self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)


            self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)

            if self.is_monitor_mn:
                exec_time = time.time() - exec_time_origin
                self.save_mn(exec_time=exec_time)

        self.aml_optimize.belief_propagators[0].mn.set_weights(weights_opt)
        learned_mn = self.aml_optimize.belief_propagators[0].mn
        learned_mn.load_factors_from_matrices()
        if self.is_show_metrics:
            self.print_metrics(recall, precision, f1_score)
            return learned_mn, self.active_set, recall, precision, f1_score, False

        return learned_mn, self.active_set, None, None, None, False


    def setup_grafting_learner(self):
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


    def print_update(self,activated_feature, precision):
        """
        print update
        """
        print('ACTIVATED EDGE')
        print(self.feature2edge[activated_feature])
        print('CURRENT ACTIVE SPACE')
        print(self.active_features)
        print('PRECISION')
        print(precision)


    def update_metrics(self, edges, recall, precision, f1_score):
        """
        UPDATE METRICS
        """
        curr_recall = float(len([x for x in self.active_features if x in self.relevant_features]))/len(self.relevant_features)
        recall.append(curr_recall)

        try:
            curr_precision = float(len([x for x in self.relevant_features if x in self.active_features]))/len(self.active_features)
        except:
            curr_precision = 0

        precision.append(curr_precision)

        if curr_precision==0 or curr_recall==0:
            curr_f1_score = 0
        else:
            curr_f1_score = (2 * curr_precision * curr_recall) / (curr_precision + curr_recall)

        f1_score.append(curr_f1_score)

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
