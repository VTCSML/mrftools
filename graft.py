from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import time
from grafting_util import get_max_gradient

MAX_ITER_GRAFT = 1500

def graft(variables, num_states, data, l1_coeff):
    """
    Main Script for graft algorithm.
    Reference: To be added.
    """
    priority_reassignements, num_injection, num_success, num_edges_reassigned, num_weights_opt, max_num_states, num_edges = 0, 0, 0, 0, 0, 0, 0
    edges_reassigned, map_weights_to_variables, map_weights_to_edges, active_set, sel_time_vec = [], [], [], [], []
    vector_length_per_var = max_num_states
    vector_length_per_edge = max_num_states ** 2
    np.random.seed(0)
    mn = MarkovNet()
    for var in variables:
        mn.set_unary_factor(var, np.zeros(num_states[var]))
        num_weights_opt += num_states[var]
        if max_num_states < num_states[var]:
            max_num_states = num_states[var]
        map_weights_to_variables.append(var)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    mn.init_search_space()
    search_space = mn.search_space
    edges_data_sum = mn.get_edges_data_sum(data)

    # ADD DATA
    for instance in data:
        aml_optimize.add_data(instance)

    # START GRAFTING
    num_possible_edges = len(search_space)
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), MAX_ITER_GRAFT)
    selected_var, max_grad = get_max_gradient(aml_optimize.belief_propagators, len(data), search_space, edges_data_sum)

    while (np.abs(max_grad) > l1_coeff) and (len(search_space) > 0):
        print('ACTIVATED EDGE')
        print(selected_var)
        print('ACTIVE SPACE')
        print(active_set)
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

        selected_var, max_grad = get_max_gradient(aml_optimize.belief_propagators, len(data), search_space, edges_data_sum)

    print('WEIGHTS')
    print(weights_opt)

    # REMOVE NON RELEVANT EDGES
    j = 0
    active_set = []
    num_weights = 0
    k = 0
    for var in map_weights_to_edges:
        curr_weights = weights_opt[k : k + vector_length_per_edge]
        print('curr_weights')
        print(curr_weights)
        if not all(abs(i) < .0001 for i in curr_weights):
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

    #MAKE WEIGHTS DICT
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

    return mn, weights_opt, weights_dict, active_set

