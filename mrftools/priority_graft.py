from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
from grafting_util import priority_reassignment, priority_mean_gradient_test, initialize_priority_queue, setup_learner_1, update_grafting_metrics, reset_unary_factors, reset_edge_factors
from graph_mining_util import make_graph, select_edge_to_inject
import copy
import operator

def priority_graft( variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, prune_threshold, prune_firing_threshold, max_iter_graft, max_num_states, verbose):
    """
    Main Script for priority graft algorithm.
    Reference: To be added.
    """
    # INITIALIZE VARIABLES
    priority_reassignements, num_injection, num_success, num_edges_reassigned, num_edges = 0, 0, 0, 0, 0
    edges_reassigned, map_weights_to_variables, map_weights_to_edges, active_set = [], [], [], []
    np.random.seed(0)
    vector_length_per_var = max_num_states
    vector_length_per_edge = max_num_states ** 2
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    search_space = mn.search_space
    num_possible_edges = len(search_space)

    edge_regularizers, var_regularizers = dict(), dict()

    # INITIALIZE PRIORITY QUEUE
    pq = initialize_priority_queue(mn)

    # GET DATA EXPECTATION
    sufficient_stats, padded_sufficient_stats = mn.get_sufficient_stats(data, max_num_states)

    aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), active_set)

    # # ADD DATA
    # setup_learner(aml_optimize, data)

    # START GRAFTING
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), max_iter_graft, edge_regularizers, var_regularizers)
    ## GRADIENT TEST
    is_activated_edge, activated_edge, sel_gradient = priority_mean_gradient_test(aml_optimize.belief_propagators, search_space, pq, sufficient_stats, data, l1_coeff)
    # outer_loop = 0
    added_edges = 0
    prune_time = 0
    while is_activated_edge:
        while ((len(pq) > 0) and is_activated_edge): # Stop if all edges are added or no edge is added at the previous iteration
            num_edges += 1
            added_edges += 1
            prune_time += 1
            active_set.append(activated_edge)
            # print('Selected gradient')
            # print(sel_gradient)
            if verbose:
                print('ACTIVATED EDGE')
                print(activated_edge)
                print('CURRENT ACTIVE SPACE')
                print(active_set)


            


            ################>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            mn.set_edge_factor(activated_edge, np.zeros((len(mn.unary_potentials[activated_edge[0]]), len(mn.unary_potentials[activated_edge[1]]))))
            
            aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), active_set)

            #OPTIMIZE
            tmp_weights_opt = np.concatenate((weights_opt, np.random.randn(added_edges * vector_length_per_edge)))
            weights_opt = aml_optimize.learn(tmp_weights_opt, max_iter_graft, edge_regularizers, var_regularizers)

            added_edges = 0
            ## PRUNING TEST
            if num_edges > prune_firing_threshold * num_possible_edges and prune_time >= 6: # Test for priority reassignment if graph density is above 'prune_firing_threshol'
                prune_time = 0
                pq , active_set, search_space, injection, success, is_added_edge, resulting_edges  = priority_reassignment(variables, active_set, aml_optimize, prune_threshold, data, search_space, pq, l1_coeff, sufficient_stats, mn)
                if is_added_edge:
                    added_edges += 1
                num_success, num_injection, priority_reassignements, num_edges_reassigned = update_grafting_metrics(injection, success, resulting_edges, edges_reassigned, num_success, num_injection, num_edges_reassigned, priority_reassignements)
            ## GRADIENT TEST
            is_activated_edge, activated_edge, sel_gradient = priority_mean_gradient_test(aml_optimize.belief_propagators, search_space, pq, sufficient_stats, data, l1_coeff)

        aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), active_set)
        weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)
        is_activated_edge, activated_edge, sel_gradient = priority_mean_gradient_test(aml_optimize.belief_propagators, search_space, pq, sufficient_stats, data, l1_coeff)
        # outer_loop += 1

    # REMOVE NON RELEVANT EDGES
    aml_optimize.belief_propagators[0].mn.update_edge_tensor()
    unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()

    final_active_set = []
    edge_mean_weights = list()
    for edge in active_set:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_weights = aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
        # if not all(val < .5 for val in list(np.abs(edge_weights))):
        edge_mean_weights.append((edge,edge_weights.dot(edge_weights) / len(edge_weights)))
        if edge_weights.dot(edge_weights) / len(edge_weights) > .05:
            final_active_set.append(edge)
    edge_mean_weights.sort(key=operator.itemgetter(1), reverse=True)
    sorted_mean_edges = [x[0] for x in edge_mean_weights]
    # print('Cleaned Active set')
    # print(final_active_set)

    # LEARN FINAL MRF
    mn_old = mn
    mn = MarkovNet()
    reset_unary_factors(mn, mn_old)
    reset_edge_factors(mn, mn_old, final_active_set)
    aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len(data), final_active_set)
    # for edge in final_active_set:
    #     i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
    #     edge_regularizers[edge] = pairwise_indices[:, :, i]
    weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)



    learned_mn = aml_optimize.belief_propagators[0].mn
    learned_mn.load_factors_from_matrices()

    edges_reassigned = list(set(edges_reassigned))
    print('Graph priority reassignments')
    print(len(edges_reassigned))
    if edges_reassigned:
        reassignment_success_rate = float(len([x for x in edges_reassigned if x not in active_set])) / float(len(edges_reassigned))
        print('Graph reassignment success rate')
        print(reassignment_success_rate)
    # print('Outer loop')
    # print(outer_loop)

    return learned_mn, final_active_set


