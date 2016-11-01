
from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
from grafting_util import priority_reassignment, initialize_priority_queue, setup_learner, reset_unary_factors, reset_edge_factors, queue_mean_gradient_test
from graph_mining_util import make_graph, select_edge_to_inject
import copy
import operator
from graph_mining_util import draw_graph, make_graph
import scipy.sparse as sps


def queue_graft( variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, max_iter_graft, max_num_states, verbose, num_edges, edges, list_order, zero_threshold, plot_queue):
    """
    Main Script for priority graft algorithm.
    Reference: To be added.
    """
    # INITIALIZE VARIABLES
    sel_time = list()
    recall, precision = list(), list()
    suff_stats_list = list()
    iterations = list()
    timing = list()
    priority_reassignements, num_injection, num_success, num_edges_reassigned = 0, 0, 0, 0
    naive_edges_reassigned, graph_edges_reassigned,edges_reassigned, map_weights_to_variables, map_weights_to_edges, active_set = [], [], [], [], [], []
    edge_regularizers, var_regularizers = dict(), dict()
    np.random.seed(0)
    vector_length_per_var = max_num_states
    vector_length_per_edge = max_num_states ** 2
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    search_space = mn.search_space
    len_search_space = len(search_space)
    search_space = [ search_space[i] for i in list_order]
    # GET DATA EXPECTATION
    sufficient_stats, padded_sufficient_stats = mn.get_unary_sufficient_stats(data, max_num_states)
    # INITIALIZE PRIORITY QUEUE
    pq = initialize_priority_queue(search_space)
    aml_optimize = setup_learner(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats,  len(data), active_set)
    # START GRAFTING
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), max_iter_graft, edge_regularizers, var_regularizers)
    ## GRADIENT TEST
    is_activated_edge, activated_edge, curr_edges_reassigned, sel_iter = queue_mean_gradient_test(aml_optimize.belief_propagators, search_space, pq, data, edge_reg, len(active_set)+2, sufficient_stats, padded_sufficient_stats, max_num_states)
    if sel_iter:
        iterations.append(sel_iter)
    edges_reassigned.extend(curr_edges_reassigned)
    naive_edges_reassigned.extend(curr_edges_reassigned)
    t = time.time()
    if plot_queue:
        loop_num = 0
        columns = list()
        rows = list()
        values = list()

    while is_activated_edge and len(active_set) < num_edges:
        while ((len(pq) > 0) and is_activated_edge) and len(active_set) < num_edges: # Stop if all edges are added or no edge is added at the previous iteration
            if plot_queue:
                loop_num += 1
                for t in range(len(active_set)):
                    columns.append(t)
                    rows.append(len(active_set))
                    values.append(.5)
                copy_pq = copy.deepcopy(pq)

                len_pq = len(copy_pq)
                for c in range(len_pq):
                    test_edge = copy_pq.popitem()[0]
                    if test_edge in edges:
                        columns.append(c + len(active_set))
                        rows.append(len(active_set))
                        values.append(1)

            active_set.append(activated_edge)
            if verbose:
                print('ACTIVATED EDGE')
                print(activated_edge)
                print('CURRENT ACTIVE SPACE')
                print(active_set)
            # draw_graph(active_set, variables)

            recall.append(float(len([x for x in active_set if x in edges]))/len(edges))
            precision.append(float(len([x for x in edges if x in active_set]))/len(active_set))
            suff_stats_list.append(100 * (len(sufficient_stats) - len(variables))/(len(variables) ** 2 - len(variables)) / 2)
            timing.append(time.time() - t)

            mn.set_edge_factor(activated_edge, np.zeros((len(mn.unary_potentials[activated_edge[0]]), len(mn.unary_potentials[activated_edge[1]]))))
            t11 = time.time()
            aml_optimize = setup_learner(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), active_set)
            

            aml_optimize.belief_propagators[0].mn.update_edge_tensor()
            for edge in active_set:
                unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
                edge_regularizers[edge] = pairwise_indices[:, :, aml_optimize.belief_propagators[0].mn.edge_index[edge]]

            #OPTIMIZE
            tmp_weights_opt = np.concatenate((weights_opt, np.random.randn(vector_length_per_edge)))
            weights_opt = aml_optimize.learn(tmp_weights_opt, max_iter_graft, edge_regularizers, var_regularizers)
            ## GRADIENT TEST
            t11 = time.time()
            is_activated_edge, activated_edge, curr_edges_reassigned,sel_iter = queue_mean_gradient_test(aml_optimize.belief_propagators, search_space, pq, data, edge_reg, len(active_set)+2, sufficient_stats, padded_sufficient_stats, max_num_states)
            t11 = time.time() - t11
            sel_time.append(t11)
            if sel_iter:
                iterations.append(sel_iter)
            if curr_edges_reassigned:
                naive_edges_reassigned.extend(curr_edges_reassigned)


        aml_optimize = setup_learner(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), active_set)
        weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)
        is_activated_edge, activated_edge, curr_edges_reassigned, sel_iter = queue_mean_gradient_test(aml_optimize.belief_propagators, search_space, pq, data, edge_reg, len(active_set)+2, sufficient_stats, padded_sufficient_stats, max_num_states)

    recall.append(float(len([x for x in active_set if x in edges]))/len(edges))
    precision.append(float(len([x for x in edges if x in active_set]))/len(active_set))
    suff_stats_list.append(100 * (len(sufficient_stats) - len(variables))/(len(variables) ** 2 - len(variables)) / 2)
    timing.append(time.time() - t)
    # REMOVE NON RELEVANT EDGES
    aml_optimize.belief_propagators[0].mn.update_edge_tensor()
    unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()

    final_active_set = active_set
    final_active_set = []
    edge_mean_weights = list()
    for edge in active_set:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_weights = aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
        edge_mean_weights.append((edge,edge_weights.dot(edge_weights) / len(edge_weights)))
        if np.sqrt(edge_weights.dot(edge_weights))  > zero_threshold:
            final_active_set.append(edge)


    if plot_queue:
        view_queue_tmp = sps.csr_matrix((np.array(values), (np.array(rows), np.array(columns))) , (loop_num, len_search_space))
        view_queue_tmp = view_queue_tmp.todense()
        view_queue_tmp[view_queue_tmp==0] = .2
        view_queue = np.zeros((loop_num+2, len_search_space+2))
        view_queue[1:loop_num+1, 1:len_search_space+1] = view_queue_tmp

        # plt.imshow(view_queue,interpolation='none',cmap='binary')
        plt.imshow(view_queue,interpolation='none',cmap='binary', aspect='auto')
        plt.title('Queue Ground truth')
        # plt.colorbar()
        plt.axis('off')
        plt.savefig('pq_plot/Q_Synth_' + str(len(variables)) +'Nodes_' + str(len(edges)) + 'Edges.png')
        plt.close()



    learned_mn = aml_optimize.belief_propagators[0].mn
    learned_mn.load_factors_from_matrices()


    naive_edges_reassigned = list(set(naive_edges_reassigned))
    print('Naive edge reassignments')
    print(len(naive_edges_reassigned))
    if naive_edges_reassigned:
        naive_reassignment_success_rate = float(len([x for x in naive_edges_reassigned if x not in active_set])) / float(len(naive_edges_reassigned))
        print('Naive reassignment success rate')
        print(naive_reassignment_success_rate)
    print('Average Selection iterations')
    print(float(sum(iterations))/len(iterations))


    print('SUFFICIENT STATS')
    print(len(sufficient_stats)- len(variables))

    print(iterations)

    return learned_mn, final_active_set, suff_stats_list, recall, precision, timing, iterations
