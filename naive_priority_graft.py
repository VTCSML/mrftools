from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
from grafting_util import naive_priority_gradient_test
from graph_mining_util import make_graph, select_edge_to_inject
from pqdict import pqdict
import copy

MAX_ITER_GRAFT = 1

def naive_priority_graft( variables, num_states, data, l1_coeff):
    """
    Main Script for naive priority graft algorithm.
    Reference: To be added.
    """
    priority_reassignements, num_injection, num_success, num_edges_reassigned, max_num_states, num_edges = 0, 0, 0, 0, 0, 0
    edges_reassigned, map_weights_to_variables, map_weights_to_edges, active_set = [], [], [], []
    np.random.seed(0)
    mn = MarkovNet()
    for var in variables:
        mn.set_unary_factor(var, np.zeros(num_states[var]))
        if max_num_states < num_states[var]:
            max_num_states = num_states[var]
        map_weights_to_variables.append(var)
    vector_length_per_var = max_num_states
    vector_length_per_edge = max_num_states ** 2
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    mn.init_search_space()
    search_space = mn.search_space

    # INITIALIZE PRIORITY QUEUE
    pq = pqdict()
    for edge in search_space:
        pq.additem(edge, 0)
    edges_data_sum = mn.get_edges_data_sum(data)

    # ADD DATA
    for instance in data:
        aml_optimize.add_data(instance)

    # START GRAFTING
    num_possible_edges = len(search_space)
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), MAX_ITER_GRAFT)
    added_edge, selected_var, pq, search_space = naive_priority_gradient_test(aml_optimize.belief_propagators, search_space, pq, edges_data_sum, data, l1_coeff, 1)
    while ((len(pq) > 0) and added_edge): # Stop if all edges are added or no edge is added at the previous iteration
        num_edges += 1
        active_set.append(selected_var)
        print('ACTIVATED EDGE')
        print(selected_var)
        print('CURRENT ACTIVE SPACE')
        print(active_set)
        new_weights_num = vector_length_per_edge
        map_weights_to_variables.append(selected_var)
        map_weights_to_edges.append(selected_var)
        mn.set_edge_factor(selected_var, np.zeros((len(mn.unary_potentials[selected_var[0]]), len(mn.unary_potentials[selected_var[1]]))))
        aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(l1_coeff, 0)
        # ADD DATA
        for instance in data:
            aml_optimize.add_data(instance)
        tmp_weights_opt = .3 * np.ones(aml_optimize.weight_dim)
        weights_opt = aml_optimize.learn(tmp_weights_opt, MAX_ITER_GRAFT)

        added_edge, selected_var, pq, search_space = naive_priority_gradient_test(aml_optimize.belief_propagators, search_space, pq, edges_data_sum, data, l1_coeff, 1)

    # OPTIMIZE UNTILL CONVERGENCE TO GET OPTIMAL WEIGHTS
    weights_opt = aml_optimize.learn(weights_opt, 1500)

    # REMOVE NON RELEVANT EDGES
    active_set = []
    num_weights = 0
    k = 0
    for var in map_weights_to_edges:
        curr_weights = weights_opt[k : k + vector_length_per_edge]
        if not all(i < .0001 for i in curr_weights):
            active_set.append(var)
        k += vector_length_per_edge
    print('Cleaned Active set')
    print(active_set)

    # LEARN FINAL MRF
    mn = MarkovNet()
    for var in variables:
        mn.set_unary_factor(var, np.zeros(num_states[var]))
    for selected_var in active_set:
        mn.set_edge_factor(selected_var, np.zeros(
            (len(mn.unary_potentials[selected_var[0]]), len(mn.unary_potentials[selected_var[1]]))))
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    # ADD DATA
    for instance in data:
        aml_optimize.add_data(instance)
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

    return learned_mn, weights_opt, weights_dict, active_set

