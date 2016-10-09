from MarkovNet import MarkovNet
import numpy as np
import operator
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import time
from grafting_util import get_max_mean_gradient, get_max_gradient, setup_learner


def bl_structure_learning(variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, max_iter_graft, max_num_states, verbose):
    """
    Main Script for graft algorithm.
    Reference: To be added.
    """
    var_regularizers = dict()
    map_weights_to_variables, map_weights_to_edges, active_set = [], [], []
    vector_length_per_var = max_num_states
    num_node_weights = max_num_states * len(variables)
    vector_length_per_edge = 2 * max_num_states ** 2
    np.random.seed(0)
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, l2_coeff, edge_reg, 0)
    search_space = mn.search_space
    edge_regularizers = dict()
    num_possible_edges = len(search_space)
    # ADD DATA
    setup_learner(aml_optimize, data)
    num_weights_nodes = aml_optimize.weight_dim

    # GET DATA EXPECTATION
    edges_data_sum = mn.get_edges_data_sum(data)
    for edge in search_space:
        map_weights_to_variables.append(edge)
        map_weights_to_edges.append(edge)
        mn.set_edge_factor(edge, np.zeros((len(mn.unary_potentials[edge[0]]), len(mn.unary_potentials[edge[1]]))))
    aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
    aml_optimize.set_regularization(l1_coeff, l2_coeff,  var_reg, edge_reg)

    # ADD DATA
    setup_learner(aml_optimize, data)
    unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
    for edge in search_space:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_regularizers[edge] = pairwise_indices[:, :, i]

    for var in variables:
        var_regularizers[var] = unary_indices[:, aml_optimize.belief_propagators[0].mn.var_index[var]]

    # Learn weights
    weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), max_iter_graft, edge_regularizers, var_regularizers)
    active_set = []
    # REMOVE NON RELEVANT EDGES
    aml_optimize.belief_propagators[0].mn.update_edge_tensor()
    unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
    edge_mean_weights = list()
    for edge in search_space:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_weights = aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
        # if not all(val < .5 for val in list(np.abs(edge_weights))):
        edge_mean_weights.append((edge,edge_weights.dot(edge_weights) / len(edge_weights)))
        if edge_weights.dot(edge_weights) / len(edge_weights) > .05:
            active_set.append(edge)
        print('////////////')
        print(edge)
        # print(edge_weights)
        print(edge_weights.dot(edge_weights) / len(edge_weights))
    edge_mean_weights.sort(key=operator.itemgetter(1), reverse=True)
    sorted_mean_edges = [x[0] for x in edge_mean_weights]
    print('Cleaned Active set')
    print(active_set)
    print('Sorted edges per mean')
    print(sorted_mean_edges)
    # LEARN FINAL MRF
    edge_regularizers = dict()
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    mn.initialize_edge_factors(active_set, map_weights_to_variables)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, l2_coeff,  var_reg, edge_reg)
    setup_learner(aml_optimize, data)
    for edge in active_set:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_regularizers[edge] = pairwise_indices[:, :, i]
    weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)

    # MAKE WEIGHTS DICT
    weights_dict = dict()
    # j = 0
    # for var in map_weights_to_variables:
    #     if isinstance(var, tuple):
    #         size = vector_length_per_edge
    #         current_weight = weights_opt[j : j + size]
    #         j += size
    #         weights_dict[var] = current_weight
    #     else:
    #         size = vector_length_per_var
    #         current_weight = weights_opt[j : j + size]
    #         j += size
    #         weights_dict[var] = current_weight

    learned_mn = aml_optimize.belief_propagators[0].mn
    learned_mn.load_factors_from_matrices()

    return learned_mn, weights_opt, weights_dict, active_set

