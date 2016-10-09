from MarkovNet import MarkovNet
import numpy as np
from ApproxMaxLikelihood import ApproxMaxLikelihood
from scipy.optimize import minimize, check_grad
import matplotlib.pyplot as plt
import time
from grafting_util import priority_reassignment, priority_mean_gradient_test, initialize_priority_queue, setup_learner, update_grafting_metrics, edge_gradient_test
from graph_mining_util import make_graph, select_edge_to_inject
import copy
import operator
from grafting_util import compute_likelihood
import numpy as np
from generate_two_clusters import generate_two_clusters_data
from BeliefPropagator import BeliefPropagator


def method_sanity_check( variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, max_iter_graft, max_num_states, basic_edges):
    """
    Main Script for priority graft algorithm.
    Reference: To be added.
    """
    edge_regularizers, var_regularizers = dict(), dict()
    # INITIALIZE VARIABLES
    np.random.seed(0)
    vector_length_per_var = max_num_states
    vector_length_per_edge = max_num_states ** 2
    mn = MarkovNet()
    search_space = mn.search_space
    map_weights_to_variables = mn.initialize_unary_factors(variables, num_states)
    edges = basic_edges
    for edge in edges:
         mn.set_edge_factor(edge, np.zeros((len(mn.unary_potentials[edge[0]]), len(mn.unary_potentials[edge[1]]))))
    sufficient_stats, padded_sufficient_stats = mn.get_sufficient_stats(data, max_num_states)
    aml_optimize = ApproxMaxLikelihood(mn)
    aml_optimize.set_regularization(l1_coeff, l2_coeff, var_reg, edge_reg)
    setup_learner(aml_optimize, data)
    num_weights_nodes = aml_optimize.weight_dim
    weights_opt = aml_optimize.learn(np.random.randn(aml_optimize.weight_dim), max_iter_graft, edge_regularizers, var_regularizers)
    edge = (0,12)
    added_edge = edge_gradient_test(aml_optimize.belief_propagators[0], edge, sufficient_stats, data, l1_coeff)

    return added_edge

def main():
    l1_coeff = 0
    l2_coeff = 0
    edge_reg = 0
    var_reg = 0
    l1_coeffs = .01
    max_grafting_iter = 2500
    recalls, precisions, likelihoods = list(), list(), list()
    print('Simulating data...')

    basic_edges = [(0,1), (0,2), (0,3), (0,4), (4,5), (6,7), (7,13), (8,13), (9,13), (10,13), (11,13), (12,13)]

    # Connect clusters

    basic_edges.append((5,6))


    model, variables, data, max_num_states, num_states, edges = generate_two_clusters_data(5000,10, basic_edges)
    num_attributes = len(variables)
    train_data = data
    print('nodes')
    print(variables)
    print('Number of Data Points')
    print(len(data))
    print('Number of Attributes')
    print(num_attributes)
    print('Number of States per Attribute')
    print(num_states)
    print('Edges')
    print(edges)
    print(' START')
    print('CONNECT CLUSTERS : ')
    print(not method_sanity_check( variables, num_states, data, l1_coeff, l2_coeff, var_reg, edge_reg, max_grafting_iter, max_num_states, basic_edges))

if  __name__ =='__main__':
    main()