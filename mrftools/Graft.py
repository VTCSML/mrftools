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
    Structured priority grafting class
    """

    def __init__(self, variables, num_states, max_num_states, data, list_order):
        """
        Initialize StructuredPriorityGraft class
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
        for edge in self.search_space:
            self.sufficient_stats[edge], self.padded_sufficient_stats[edge] = self.get_sufficient_stats_per_edge(self.mn, edge)
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

    def on_show_metrics(self):
        self.is_show_metrics = True


    def on_verbose(self):
        self.is_verbose = True

    def learn_structure(self, num_edges, edges=list()):
        """
        Main function for grafting
        """
        # INITIALIZE VARIABLES
        if self.is_show_metrics:
            recall, precision, suff_stats_list = list(), list(), list()

        np.random.seed(0)

        self.aml_optimize = self.setup_grafting_learner()
        vector_length_per_var = self.max_num_states
        vector_length_per_edge = self.max_num_states ** 2
        len_search_space = len(self.search_space)
        weights_opt = self.aml_optimize.learn(np.random.randn(self.aml_optimize.weight_dim), self.max_iter_graft, self.edge_regularizers, self.node_regularizers)
        ## GRADIENT TEST

        is_activated_edge, activated_edge = self.activation_test()

        while is_activated_edge and len(self.active_set) < num_edges:
            while (self.search_space and is_activated_edge) and len(self.active_set) < num_edges: # Stop if all edges are added or no edge is added at the previous iteration

                self.active_set.append(activated_edge)
                if self.is_verbose:
                    self.print_update(activated_edge)
                # draw_graph(active_set, variables)
                if self.is_show_metrics:
                    self.update_metrics(edges, recall, precision, suff_stats_list)
                self.mn.set_edge_factor(activated_edge, np.zeros((len(self.mn.unary_potentials[activated_edge[0]]), len(self.mn.unary_potentials[activated_edge[1]]))))
                self.aml_optimize = self.setup_grafting_learner()
                
                self.aml_optimize.belief_propagators[0].mn.update_edge_tensor()
                for edge in self.active_set:
                    unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
                    self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]
                #OPTIMIZE
                tmp_weights_opt = np.concatenate((weights_opt, np.random.randn(vector_length_per_edge)))
                weights_opt = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers)
                ## GRADIENT TEST
                is_activated_edge, activated_edge = self.activation_test()

            #Outerloop
            self.aml_optimize = self.setup_grafting_learner()
            weights_opt = self.aml_optimize.learn(np.zeros(self.aml_optimize.weight_dim), 2500, self.edge_regularizers, self.node_regularizers)
            is_activated_edge, activated_edge = self.activation_test()

        if self.is_show_metrics and is_activated_edge:
            self.update_metrics(edges, recall, precision, suff_stats_list)
        # REMOVE NON RELEVANT EDGES
        final_active_set = self.remove_zero_edges()

        # draw_graph(active_set, variables)
        # print('Cleaned Active set')
        # print(final_active_set)
        # # LEARN FINAL MRF
        # mn_old = mn
        # mn = MarkovNet()
        # reset_unary_factors(mn, mn_old)
        # reset_edge_factors(mn, mn_old, final_active_set)
        # # aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), final_active_set)
        # # for edge in final_active_set:
        # #     i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        # #     edge_regularizers[edge] = pairwise_indices[:, :, i]
        # # weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)

        learned_mn = self.aml_optimize.belief_propagators[0].mn
        learned_mn.load_factors_from_matrices()

        if self.is_show_metrics:
            self.print_metrics(recall, precision)
            return learned_mn, final_active_set, suff_stats_list, recall, precision

        if self.is_verbose:
            print('Final Active Set')
            print(final_active_set)

        return learned_mn, final_active_set, None, None, None


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
            belief = bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
            gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(self.sufficient_stats[edge]) / len(self.data)).squeeze()
            gradient_norm = np.sqrt(gradient.dot(gradient))
            gradient_vec.append(gradient_norm)
            map_vec.append(edge)
        max_ind = np.array(gradient_vec).argmax(axis=0)
        max_grad = max(gradient_vec)
        if max_grad >= self.edge_l1:
            is_activated = True
            activated_edge = map_vec[max_ind]
            self.search_space.remove(activated_edge)
        return is_activated, activated_edge


    def print_update(self, activated_edge):
        """
        print update
        """
        print('ACTIVATED EDGE')
        print(activated_edge)
        print('CURRENT ACTIVE SPACE')
        print(self.active_set)

    def update_metrics(self, edges, recall, precision, suff_stats_list):
        """
        UPDATE METRICS
        """
        recall.append(float(len([x for x in self.active_set if x in edges]))/len(edges))
        precision.append(float(len([x for x in edges if x in self.active_set]))/len(self.active_set))
        suff_stats_list.append(100 * (len(self.sufficient_stats) - len(self.variables))/(len(self.variables) ** 2 - len(self.variables)) / 2)

    def remove_zero_edges(self):
        """
        Filter out near zero edges
        """
        self.aml_optimize.belief_propagators[0].mn.update_edge_tensor()
        unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        final_active_set = list()
        for edge in self.active_set:
            i = self.aml_optimize.belief_propagators[0].mn.edge_index[edge]
            edge_weights = self.aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:self.aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :self.aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
            if np.sqrt(edge_weights.dot(edge_weights))  > self.zero_threshold:
                final_active_set.append(edge)
        return final_active_set

    def print_metrics(self, recall, precision):
        """
        Print metrics
        """
        print('SUFFICIENT STATS')
        print(len(self.sufficient_stats)- len(self.variables))
        print('Recall')
        print(recall[-1])
        print('Precision')
        print(precision[-1])

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
