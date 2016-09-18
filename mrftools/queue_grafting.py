from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
from grafting_util import priority_reassignment, move_to_tail, priority_gradient_test, reduce_priority, edge_gradient_test, initialize_priority_queue, setup_learner
from graph_mining_util import make_graph, select_edge_to_inject
from pqdict import pqdict
import copy


def queue_graft( variables, num_states, data, l1_coeff, max_iter_graft, max_num_states, verbose):
    """
    Main Script for priority graft algorithm.
    Reference: To be added.
    """
    # INITIALIZE VARIABLES
    priority_reassignements, num_injection, num_success, num_edges_reassigned, num_edges = 0, 0, 0, 0, 0
    edges_reassigned, map_weights_to_variables, map_weights_to_edges, active_set = [], [], [], []
    np.random.seed(0)
    vector_length_per_var = max_num_states
    vector_length_per_edge = 2 * max_num_states ** 2
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    search_space = mn.search_space
    num_possible_edges = len(search_space)
    # INITIALIZE PRIORITY QUEUE
    pq = initialize_priority_queue(mn)
    # GET DATA EXPECTATION
    edges_data_sum = mn.get_edges_data_sum(data)
    # ADD DATA
    setup_learner(aml_optimize, data)
    # START GRAFTING
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), max_iter_graft)
    ## GRADIENT TEST
    is_activated_edge, activated_edge = priority_gradient_test(aml_optimize.belief_propagators, search_space, pq, edges_data_sum, data, l1_coeff)

    while ((len(pq) > 0) and is_activated_edge): # Stop if all edges are added or no edge is added at the previous iteration
        num_edges += 1
        active_set.append(activated_edge)
        if verbose:
            print('ACTIVATED EDGE')
            print(activated_edge)
            print('CURRENT ACTIVE SPACE')
            print(active_set)
        map_weights_to_variables.append(activated_edge)
        map_weights_to_edges.append(activated_edge)
        ## UPDATE MN
        mn.set_edge_factor(activated_edge, np.zeros((len(mn.unary_potentials[activated_edge[0]]), len(mn.unary_potentials[activated_edge[1]]))))
        ## CREATE A NEW 'ApproxMaxLikelihood'
        aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(l1_coeff, 0)
        setup_learner(aml_optimize, data)
        weights_opt = np.concatenate((weights_opt, np.random.randn(vector_length_per_edge)))
        weights_opt = aml_optimize.learn(weights_opt, max_iter_graft)
        ## GRADIENT TEST
        is_activated_edge, activated_edge = priority_gradient_test(aml_optimize.belief_propagators, search_space, pq, edges_data_sum, data, l1_coeff)
    # OPTIMIZE UNTILL CONVERGENCE TO GET OPTIMAL WEIGHTS
    weights_opt = aml_optimize.learn(weights_opt, 1500)

    # REMOVE NON RELEVANT EDGES
    active_set = []
    num_weights = 0
    k = 0
    for var in map_weights_to_edges:
        curr_weights = weights_opt[k: k + vector_length_per_edge]
        if not all(i < .0001 for i in curr_weights):
            active_set.append(var)
        k += vector_length_per_edge
    print('Cleaned Active set')
    print(active_set)

    # LEARN FINAL MRF
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    mn.initialize_edge_factors(active_set, map_weights_to_variables)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    setup_learner(aml_optimize, data)
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), 1500)

    # MAKE WEIGHTS DICT
    weights_dict = dict()
    j = 0
    for var in map_weights_to_variables:
        if isinstance(var, tuple):
            size = vector_length_per_edge
            current_weight = weights_opt[j : j + size]
            j += size
            weights_dict[var] = current_weight
        else:
            size = vector_length_per_var
            current_weight = weights_opt[j : j + size]
            j += size
            weights_dict[var] = current_weight

    learned_mn = aml_optimize.belief_propagators[0].mn
    learned_mn.load_factors_from_matrices()

    if edges_reassigned:
        reassignment_success_rate = float(len([x for x in edges_reassigned if x not in active_set])) / float(len(edges_reassigned))
        print('reassignment_success_rate')
        print(reassignment_success_rate)

    return learned_mn, weights_opt, weights_dict, active_set