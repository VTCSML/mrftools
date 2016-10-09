from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import time
from grafting_util import get_sorted_mean_gradient_gl, get_max_gradient, setup_learner
import operator


def graft_light(variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, max_num_states, verbose, edges_to_graft_number):
    """
    Main Script for graft algorithm.
    Reference: To be added.
    """
    max_iter_graft = 1
    active_set = []
    np.random.seed(0)
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, l2_coeff, edge_reg, var_reg)
    search_space = mn.search_space
    vector_length_per_edge = max_num_states ** 2
    # GET DATA EXPECTATION
    sufficient_stats, padded_sufficient_stats = mn.get_sufficient_stats(data, max_num_states)
    # ADD DATA
    setup_learner(aml_optimize, data)
    edge_regularizers, var_regularizers = dict(), dict()
    # START GRAFTING

    # unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
    # for var in variables:
    #     var_regularizers[var] = unary_indices[:, aml_optimize.belief_propagators[0].mn.var_index[var]]

    weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), max_iter_graft, edge_regularizers, var_regularizers)
    activated_edges, search_space = get_sorted_mean_gradient_gl(aml_optimize.belief_propagators, len(data), search_space, sufficient_stats, l1_coeff, edges_to_graft_number)
    t = time.time()
    while activated_edges and (len(search_space) > 0):
        t1 = time.time()
        active_set.extend(activated_edges)
        if verbose:
            print('ACTIVATED EDGE')
            print(activated_edges)
            print('CURRENT ACTIVE SPACE')
            print(active_set)

        ################>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        mn.set_edge_factor(activated_edge, np.zeros((len(mn.unary_potentials[activated_edge[0]]), len(mn.unary_potentials[activated_edge[1]]))))
        aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(l1_coeff, l2_coeff, var_reg, edge_reg)
        aml_optimize.init_grafting()
        unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        tau_q = np.zeros(aml_optimize.weight_dim)
        for var in mn.variables:
            i = aml_optimize.belief_propagators[0].mn.var_index[var]
            inds = unary_indices[:, i]
            tau_q[inds] = padded_sufficient_stats[var] / len(data)
        for edge in active_set:
            i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
            inds = pairwise_indices[:, :, i]
            tau_q[inds] = padded_sufficient_stats[edge] / len(data)
        aml_optimize.set_sufficient_stats(tau_q)
        ################>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        ##########################################################################################
        # for edge in activated_edges:
        #     mn.set_edge_factor(edge, np.zeros((len(mn.unary_potentials[edge[0]]), len(mn.unary_potentials[edge[1]]))))
        # aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
        aml_optimize.set_regularization(l1_coeff, l2_coeff, var_reg, edge_reg)
        # ADD DATA
        # t5 = time.time()
        # setup_learner(aml_optimize, data)
        # print('time setup learner')
        # print(time.time() - t5)
        # tmp_weights_opt = np.random.randn(aml_optimize.weight_dim)
        # ##########################################################################################
        # unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        # for edge in activated_edges:
        #     i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        #     edge_regularizers[edge] = pairwise_indices[:, :, i]


        tmp_weights_opt = np.concatenate((weights_opt, np.random.randn(len(activated_edges) * vector_length_per_edge)))
        weights_opt = aml_optimize.learn(tmp_weights_opt, max_iter_graft, edge_regularizers, var_regularizers)
        activated_edges, search_space = get_sorted_mean_gradient_gl(aml_optimize.belief_propagators, len(data), search_space, edges_data_sum, l1_coeff, edges_to_graft_number)
    
    print('time loop')
    print(time.time() - t)
    print 'removing non relevant edges'
    # REMOVE NON RELEVANT EDGES
    aml_optimize.learn(tmp_weights_opt, 5000, edge_regularizers, var_regularizers)
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
        # print(edge)
        # print(edge_weights)
        # print(edge_weights.dot(edge_weights) / len(edge_weights))
    edge_mean_weights.sort(key=operator.itemgetter(1), reverse=True)
    sorted_mean_edges = [x[0] for x in edge_mean_weights]
    print('Sorted edges per mean')
    print(sorted_mean_edges)
    print('Cleaned Active set')
    print(final_active_set)

    # LEARN FINAL MRF
    edge_regularizers = dict()
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    mn.initialize_edge_factors(active_set, map_weights_to_variables)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, l2_coeff, var_reg, edge_reg)
    setup_learner(aml_optimize, data)
    for edge in final_active_set:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_regularizers[edge] = pairwise_indices[:, :, i]
    weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)

    for edge in final_active_set:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        edge_weights = aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()
        # print 'Final weights'
        # print edge
        # print edge_weights

    # # MAKE WEIGHTS DICT
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

    return learned_mn, weights_opt, weights_dict, final_active_set

