from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import time
from grafting_util import get_max_mean_gradient, get_max_gradient, setup_learner, setup_learner_1
import operator


def graft(variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, max_iter_graft, max_num_states, verbose):
    """
    Main Script for graft algorithm.
    Reference: To be added.
    """
    active_set = []
    np.random.seed(0)
    vector_length_per_var = max_num_states
    vector_length_per_edge = max_num_states ** 2
    mn = MarkovNet()
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    search_space = mn.search_space
    # GET DATA EXPECTATION
    sufficient_stats, padded_sufficient_stats = mn.get_sufficient_stats(data, max_num_states)

    aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats,  len(data), active_set)
    edge_regularizers, var_regularizers = dict(), dict()
    # START GRAFTING

    # unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
    # for var in variables:
    #     var_regularizers[var] = unary_indices[:, aml_optimize.belief_propagators[0].mn.var_index[var]]

    weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), max_iter_graft, edge_regularizers, var_regularizers)
    activated_edge, max_grad, search_space = get_max_mean_gradient(aml_optimize.belief_propagators, len(data), search_space, sufficient_stats)
    # print('max_grad')
    # print(max_grad)
    t = time.time()
    while (np.abs(max_grad) > l1_coeff) and (len(search_space) > 0):
        t1 = time.time()
        # print('max_grad')
        # print(max_grad)
        active_set.append(activated_edge)
        if verbose:
            print('ACTIVATED EDGE')
            print(activated_edge)
            print('CURRENT ACTIVE SPACE')
            print(active_set)
        
        mn.set_edge_factor(activated_edge, np.zeros((len(mn.unary_potentials[activated_edge[0]]), len(mn.unary_potentials[activated_edge[1]]))))
        aml_optimize = setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats,  len(data), active_set)


        # unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
        # i = aml_optimize.belief_propagators[0].mn.edge_index[activated_edge]
        # edge_regularizers[activated_edge] = pairwise_indices[:, :, i]
        t2 = time.time()
        tmp_weights_opt = np.concatenate((weights_opt, np.random.randn(vector_length_per_edge)))
        weights_opt = aml_optimize.learn(tmp_weights_opt, max_iter_graft, edge_regularizers, var_regularizers)
        # print('time opt')
        # print(time.time() - t2)
        activated_edge, max_grad, search_space = get_max_mean_gradient(aml_optimize.belief_propagators, len(data), search_space, sufficient_stats)
    #     print('time iter')
    #     print(time.time() - t1)
    # print('time loop')
    # print(time.time() - t)
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

    # # LEARN FINAL MRF
    # edge_regularizers = dict()
    # mn = MarkovNet()
    # map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    # mn.initialize_edge_factors(active_set, map_weights_to_variables)
    # aml_optimize = ApproxMaxLikelihood(mn)
    # aml_optimize.set_regularization(l1_coeff, l2_coeff, var_reg, edge_reg)
    # setup_learner(aml_optimize, data)
    # for edge in final_active_set:
    #     i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
    #     edge_regularizers[edge] = pairwise_indices[:, :, i]
    # weights_opt = aml_optimize.learn(np.zeros(aml_optimize.weight_dim), 2500, edge_regularizers, var_regularizers)

    # for edge in final_active_set:
    #     i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
    #     edge_weights = aml_optimize.belief_propagators[0].mn.edge_pot_tensor[:aml_optimize.belief_propagators[0].mn.num_states[edge[1]], :aml_optimize.belief_propagators[0].mn.num_states[edge[0]], i].flatten()

    learned_mn = aml_optimize.belief_propagators[0].mn
    learned_mn.load_factors_from_matrices()

    return learned_mn, final_active_set

