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
#   Pairwise MRF stracture learning using structured priority grafting
#
# + TODO : add logs and reference
# 
# + Author: (walidch)
# + Reference : 
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StructuredPriorityGraft():
    """
    Structured priority grafting class
    """
    def __init__(self, variables, num_states, max_num_states, data, list_order, method='structured'):
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
        self.pq = initialize_priority_queue(self.search_space)
        self.method = method
        self.l1_coeff = 0
        self.l2_coeff = 0
        self.node_l1 = 0
        self.edge_l1 = .1
        self.zero_threshold = 1e-2
        self.max_iter_graft = 500
        self.active_set = []
        self.edge_regularizers, self.node_regularizers = dict(), dict()
        self.is_show_metrics = False
        self.is_plot_queue = False
        self.is_verbose = False
        self.priority_increase_decay_factor = .8
        self.priority_decrease_decay_factor = .95
        self.plot_path = '.'
        self.is_monitor_mn = False
        self.mn_snapshots = dict()
        self.is_converged = False

    def set_priority_factors(priority_increase_decay_factor, priority_decrease_decay_factor):
        """
        Set priority factors
        """
        self.priority_increase_decay_factor = priority_increase_decay_factor
        self.priority_decrease_decay_factor = priority_decrease_decay_factor


    def on_monitor_mn(self):
        """
        Enable monitoring Markrov net
        """
        self.is_monitor_mn = True

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

    def on_plot_queue(self, plot_path='.'):
        self.is_plot_queue = True
        self.plot_path = plot_path

    def on_verbose(self):
        self.is_verbose = True

    def learn_structure(self, num_edges, edges=list()):
        """
        Main function for grafting
        """
        # INITIALIZE VARIABLES
        if self.is_monitor_mn:
            exec_time_origin = time.time()

        self.aml_optimize = self.setup_grafting_learner(len(self.data))

        if self.is_monitor_mn:
            learned_mn = self.aml_optimize.belief_propagators[0].mn
            learned_mn.load_factors_from_matrices()
            exec_time = time.time() - exec_time_origin
            self.mn_snapshots[exec_time] = learned_mn

        if self.is_plot_queue:
            columns, rows, values = list(), list(), list()
            loop_num = 0
            pq_history = dict()
        if self.is_show_metrics:
            recall, precision, suff_stats_list, iterations = list(), list(), list(), list()

        np.random.seed(0)

        vector_length_per_var = self.max_num_states
        vector_length_per_edge = self.max_num_states ** 2
        len_search_space = len(self.search_space)

        weights_opt = self.aml_optimize.learn(np.random.randn(self.aml_optimize.weight_dim), self.max_iter_graft, self.edge_regularizers, self.node_regularizers)
        ## GRADIENT TEST

        is_activated_edge, activated_edge, iter_number = self.activation_test()

        while is_activated_edge and len(self.active_set) < num_edges:
            while ((len(self.pq) > 0) and is_activated_edge) and len(self.active_set) < num_edges: # Stop if all edges are added or no edge is added at the previous iteration
                
                if self.is_monitor_mn:
                    learned_mn = self.aml_optimize.belief_propagators[0].mn
                    learned_mn.load_factors_from_matrices()
                    exec_time = time.time() - exec_time_origin
                    self.mn_snapshots[exec_time] = learned_mn

                if self.is_plot_queue:
                    loop_num = self.update_plot_info(loop_num, columns, rows, values, pq_history, edges)

                self.active_set.append(activated_edge)
                if self.is_verbose:
                    self.print_update(activated_edge)
                # draw_graph(active_set, variables)
                if self.is_show_metrics:
                    self.update_metrics(edges, recall, precision, suff_stats_list, iterations, iter_number)
                
                self.mn.set_edge_factor(activated_edge, np.zeros((len(self.mn.unary_potentials[activated_edge[0]]), len(self.mn.unary_potentials[activated_edge[1]]))))
                self.aml_optimize = self.setup_grafting_learner(len(self.data))
                self.aml_optimize.belief_propagators[0].mn.update_edge_tensor()

                #REINITIALIZE NEW VECTOR
                tmp_weights_opt = np.random.randn(len(weights_opt) + vector_length_per_edge)
                unary_indices, pairwise_indices = self.aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
                for edge in self.active_set:
                    self.edge_regularizers[edge] = pairwise_indices[:, :, self.aml_optimize.belief_propagators[0].mn.edge_index[edge]]
                    try:
                        tmp_weights_opt[self.edge_regularizers[edge]] = weights_opt[old_edge_regularizers[edge]]
                    except:
                        pass
                for node in self.variables:
                    self.node_regularizers[node] = unary_indices[:, self.aml_optimize.belief_propagators[0].mn.var_index[node]]
                    try:
                        tmp_weights_opt[self.node_regularizers[node]] = weights_opt[old_node_regularizers[node]]
                    except:
                        pass
                old_edge_regularizers = copy.deepcopy(self.edge_regularizers)
                old_node_regularizers = copy.deepcopy(self.node_regularizers)


                #OPTIMIZE
                # tmp_weights_opt = np.concatenate((weights_opt, np.random.randn(vector_length_per_edge))) # NEED To REORGANIZE THE WEIGHT VECTOR 
                
                weights_opt = self.aml_optimize.learn(tmp_weights_opt, self.max_iter_graft, self.edge_regularizers, self.node_regularizers)
                
                ## GRADIENT TEST
                is_activated_edge, activated_edge, iter_number = self.activation_test()


            #Outerloop
            weights_opt = self.aml_optimize.learn(weights_opt, 2500, self.edge_regularizers, self.node_regularizers)
            is_activated_edge, activated_edge, iter_number = self.activation_test()


            if self.is_monitor_mn:
                learned_mn = self.aml_optimize.belief_propagators[0].mn
                learned_mn.load_factors_from_matrices()
                exec_time = time.time() - exec_time_origin
                self.mn_snapshots[exec_time] = learned_mn

        if is_activated_edge == False:
                self.is_converged = True

        if self.is_show_metrics and is_activated_edge:
            self.update_metrics(edges, recall, precision, suff_stats_list, iterations, iter_number)
        # REMOVE NON RELEVANT EDGES
        final_active_set = self.remove_zero_edges()

        if self.is_plot_queue:
            self.make_queue_plot_ground_truth(values, rows, columns, loop_num, len_search_space)
            self.make_queue_plot_synthetic_truth(pq_history, final_active_set, loop_num, len_search_space)

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
            self.print_metrics(iterations, recall, precision)
            return learned_mn, final_active_set, suff_stats_list, recall, precision, iterations

        if self.is_verbose:
            print('Final Active Set')
            print(final_active_set)

        return learned_mn, final_active_set, None, None, None, None


    def setup_grafting_learner(self, len_data):
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
            tau_q[inds] = self.padded_sufficient_stats[var] / len_data
        for edge in self.active_set:
            i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
            inds = pairwise_indices[:, :, i]
            tau_q[inds] = self.padded_sufficient_stats[edge] / len_data
        aml_optimize.set_sufficient_stats(tau_q)
        return aml_optimize


    def activation_test(self):
        """
        Test edges for activation
        """
        iteration_activation = 0
        tmp_list = []
        bp = self.aml_optimize.belief_propagators[0]
        bp.load_beliefs()
        copy_search_space = copy.deepcopy(self.search_space)
        original_pq = copy.deepcopy(self.pq)
        while len(self.pq)>0:
            item = self.pq.popitem()# Get edges by order of priority
            edge = item[0]
            if edge in copy_search_space:
                iteration_activation += 1
                copy_search_space.remove(edge)
                if edge in self.sufficient_stats:
                    sufficient_stat_edge = self.sufficient_stats[edge]
                else:
                    sufficient_stat_edge, padded_sufficient_stat_edge =  self.get_sufficient_stats_per_edge(self.aml_optimize.belief_propagators[0].mn,edge)
                    self.sufficient_stats[edge] = sufficient_stat_edge
                    self.padded_sufficient_stats[edge] = padded_sufficient_stat_edge
                belief = bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
                bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
                gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(sufficient_stat_edge) / len(self.data)).squeeze()
                gradient_norm = np.sqrt(gradient.dot(gradient))
                length_normalizer = float(1)  / ( len(bp.mn.unary_potentials[edge[0]])  * len(bp.mn.unary_potentials[edge[1]] ))
                activate = gradient_norm > length_normalizer * self.edge_l1
                if activate:
                    self.search_space.remove(item[0])
                    if self.method == 'queue':
                        _ = original_pq.pop(edge)
                        self.pq = original_pq
                    else:
                        [self.pq.additem(items[0], items[1]) for items in tmp_list]# If an edge is activated, return the previously poped edges with reduced priority
                        edge = item[0]
                        if self.method == 'structured':
                            reward1 = self.priority_increase_decay_factor ** 1 * (1 - gradient_norm/(length_normalizer * self.edge_l1))
                            reward2 = self.priority_increase_decay_factor ** 2 * (1 - gradient_norm/(length_normalizer * self.edge_l1))
                            neighbors_1 = list(bp.mn.get_neighbors(edge[0]))
                            neighbors_2 = list(bp.mn.get_neighbors(edge[1]))
                            curr_resulting_edges_1 = list(set([(x, y) for (x, y) in
                                          list(itertools.product([edge[0]], neighbors_2)) +
                                          list(itertools.product([edge[1]], neighbors_1)) if
                                          x < y and (x, y) in self.search_space]))
                            curr_resulting_edges_2 = list(set([(x, y) for (x, y) in
                                          list(itertools.product(neighbors_1, neighbors_2)) +
                                          list(itertools.product(neighbors_2, neighbors_1)) if
                                          x < y and (x, y) in self.search_space]))
                            for res_edge in curr_resulting_edges_1:
                                try:
                                    self.pq.updateitem(res_edge, self.pq[res_edge] + reward1)
                                except:
                                    pass
                            for res_edge in curr_resulting_edges_2:
                                try:
                                    self.pq.updateitem(res_edge, self.pq[res_edge] + reward2)
                                except:
                                    pass
                    return True, edge, iteration_activation
                else:
                    if self.method == 'queue':
                        pass
                    direct_penalty = 1 - gradient_norm/(length_normalizer * self.edge_l1)
                    tmp_list.append( (item[0], item[1] + direct_penalty) )# Store not activated edges in a temporary list
                    edge = item[0]
                    if self.method == 'structured':
                        penalty1 = self.priority_decrease_decay_factor ** 1 * direct_penalty
                        penalty2 = self.priority_decrease_decay_factor ** 2 * direct_penalty
                        neighbors_1 = list(bp.mn.get_neighbors(edge[0]))
                        neighbors_2 = list(bp.mn.get_neighbors(edge[1]))
                        curr_resulting_edges_1 = list(set([(x, y) for (x, y) in
                                      list(itertools.product([edge[0]], neighbors_2)) +
                                      list(itertools.product([edge[1]], neighbors_1)) if
                                      x < y and (x, y) in self.search_space]))
                        curr_resulting_edges_2 = list(set([(x, y) for (x, y) in
                                      list(itertools.product(neighbors_1, neighbors_2)) +
                                      list(itertools.product(neighbors_2, neighbors_1)) if
                                      x < y and (x, y) in self.search_space]))
                        for res_edge in curr_resulting_edges_1:
                            try:
                                self.pq.updateitem(res_edge, self.pq[res_edge] +  penalty1)
                                # copy_search_space.remove(res_edge)
                            except:
                                pass
                        for res_edge in curr_resulting_edges_2:
                            try:
                                self.pq.updateitem(res_edge, self.pq[res_edge] + penalty2)
                                # copy_search_space.remove(res_edge)
                            except:
                                pass
        return False, (0, 0), iteration_activation



    def make_queue_plot_ground_truth(self, values, rows, columns, loop_num, len_search_space):
        """
        Plot queue reorganization using ground truth edges
        """
        view_queue_tmp = sps.csr_matrix((np.array(values), (np.array(rows), np.array(columns))) , (loop_num, len_search_space))
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
                test_edge = copy_pq.popitem()[0]
                if test_edge in final_active_set:
                    columns.append(c + t)
                    rows.append(t )
                    values.append(1)
        view_queue_tmp = sps.csr_matrix((np.array(values), (np.array(rows), np.array(columns))) , (loop_num, len_search_space))
        view_queue_tmp = view_queue_tmp.todense()
        view_queue_tmp[view_queue_tmp==0] = .2
        view_queue = np.zeros((loop_num+2, len_search_space+2))
        view_queue[1:loop_num+1, 1:len_search_space+1] = view_queue_tmp
        # plt.imshow(view_queue,interpolation='none',cmap='binary')
        plt.imshow(view_queue,interpolation='none',cmap='binary', aspect='auto')
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
        copy_pq = copy.deepcopy(self.pq)
        pq_history[loop_num] = copy.deepcopy(copy_pq)
        len_pq = len(copy_pq)
        for c in range(len_pq):
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
        print('CURRENT ACTIVE SPACE')
        print(self.active_set)

    def update_metrics(self, edges, recall, precision, suff_stats_list, iterations, iter_number):
        """
        UPDATE METRICS
        """
        recall.append(float(len([x for x in self.active_set if x in edges]))/len(edges))
        precision.append(float(len([x for x in edges if x in self.active_set]))/len(self.active_set))
        suff_stats_list.append((len(self.sufficient_stats) - len(self.variables)))
        iterations.append(iter_number)

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

    def print_metrics(self, iterations, recall, precision):
        """
        Print metrics
        """
        print('Average Selection iterations')
        print(float(sum(iterations))/len(iterations))
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
