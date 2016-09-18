from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import time
from grafting_util import get_max_gradient, setup_learner

MAX_ITER_GRAFT = 2500

def graft(variables, num_states, data, l1_coeff, max_iter_graft, max_num_states, verbose):
    """
    Main Script for graft algorithm.
    Reference: To be added.
    """
    max_num_states, num_edges = 0, 0
    map_weights_to_variables, map_weights_to_edges, active_set = [], [], []
    np.random.seed(0)
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    search_space = mn.search_space
    num_possible_edges = len(search_space)
    vector_length_per_var = max_num_states
    vector_length_per_edge = 2 * max_num_states ** 2
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, 0)
    search_space = mn.search_space
    # GET DATA EXPECTATION
    edges_data_sum = mn.get_edges_data_sum(data)
    # ADD DATA
    setup_learner(aml_optimize, data)

    # START GRAFTING
    num_possible_edges = len(search_space)
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), max_iter_graft)
    activated_edge, max_grad, search_space = get_max_gradient(aml_optimize.belief_propagators, len(data), search_space, edges_data_sum)

    while (np.abs(max_grad) > l1_coeff) and (len(search_space) > 0):
        num_edges += 1
        active_set.append(activated_edge)
        if verbose:
            print('ACTIVATED EDGE')
            print(activated_edge)
            print('CURRENT ACTIVE SPACE')
            print(active_set)
        map_weights_to_variables.append(activated_edge)
        map_weights_to_edges.append(activated_edge)
        mn.set_edge_factor(activated_edge, np.zeros((len(mn.unary_potentials[activated_edge[0]]), len(mn.unary_potentials[activated_edge[1]]))))
        aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(l1_coeff, 0)
        # ADD DATA
        setup_learner(aml_optimize, data)
        tmp_weights_opt = np.random.randn(aml_optimize.weight_dim)
        weights_opt = aml_optimize.learn(tmp_weights_opt, max_iter_graft)

        activated_edge, max_grad, search_space = get_max_gradient(aml_optimize.belief_propagators, len(data), search_space, edges_data_sum)

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

    return learned_mn, weights_opt, weights_dict, active_set

