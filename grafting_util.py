import numpy as np
from scipy.misc import logsumexp
import copy
from graph_mining_util import *

def priority_reassignment(variables, activeSet, aml_optimize , pruneThreshold, data, searchSpace, pq, l1coeff, edgesDataSum):
    currGraph = make_graph(variables, activeSet)
    injection = False
    success = False
    found, selectedEdge, resultingEdges = select_edge_to_inject(currGraph, searchSpace, pruneThreshold)
    if found:
        injection = True
        print('Testing priority Reassignment possibility using edge:')
        print(selectedEdge)
        addedEdge = edge_gradient_test(aml_optimize.belief_propagators, selectedEdge, edgesDataSum, data, l1coeff)
        if addedEdge:
            print('priority Reassignment NOT Authorized')
            pq.pop(selectedEdge)
            searchSpace.remove(selectedEdge)
            activeSet.append(selectedEdge)
            print('ACTIVATED EDGE')
            print(selectedEdge)
            print('ACTIVE SPACE')
            print(activeSet)
        else:
            success = True
            print('priority Reassignment Authorized')
            print('priority Reassigned Edges:')
            print(resultingEdges)
            pq.pop(selectedEdge)
            searchSpace.remove(selectedEdge)
            pq = reduce_priority(pq, resultingEdges)
    return pq, activeSet, searchSpace, injection, success, resultingEdges

# def gradient_test(bp, searchSpace, dataSum, data, l1Coeff):
#     """Compute the gradient w.r.t. the current weight"""
#     for e in range(len(searchSpace)):
#         edge = searchSpace[e]
#         belief = bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
#                  bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
#         gradient = (np.exp(belief.reshape((-1, 1)).tolist()) - np.asarray(dataSum[edge]) / len(data)).squeeze()
#         test = all(i < l1Coeff for i in np.abs(gradient))
#         if not test:
#             return True, edge
#     return False, (0,0)

def edge_gradient_test(bps, edge, data_sum, data, l1_coeff):
    """
    Functionality :
    1 - Compute the gradient of the current edge. 
    2 - Activate edge if gradient has at least one component bigger than l1_coeff.
    """
    n = 0
    belief = 0
    for bp in bps:
        bp.load_beliefs()
        belief += bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
        bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
        n += 1
    belief = belief / n
    gradient = (np.exp(belief.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
    g = np.abs(gradient)
    remove = all(i < l1_coeff for i in g)
    if remove:
        return False
    return True

def priority_gradient_test(bps, searchSpace, pq, dataSum, data, l1Coeff):
    """
    Functionality :
    1 - Parse the priority queue 'pq' and compute the gradient of the current edge. 
    2 - Activate edge if gradient has at least one component bigger than l1_coeff.
    """
    tmp_list = []
    while len(pq)>0:
        item = pq.popitem()# Get edges by order of priority
        edge = item[0]
        n = 0
        belief = 0
        for bp in bps:
            bp.load_beliefs()
            belief += bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
            bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
            n += 1
        belief = belief / n
        gradient = (np.exp(belief.reshape((-1, 1)).tolist()) - np.asarray(dataSum[edge]) / len(data)).squeeze()
        activate = not all(i < l1Coeff for i in np.abs(gradient))
        if activate:
            searchSpace.remove(item[0])
            for item in tmp_list:# If an edge is activated, return the previously poped edges
                pq.additem(item[0],item[1])
            return True, edge, pq, searchSpace
        else: tmp_list.append(item)# Store not activated edges in a temporary list
    return False, (0, 0), pq, searchSpace

def naive_priority_gradient_test(bps, search_space, pq, data_sum, data, l1_coeff, reassignmet):
    """
    Functionality :
    1 - Parse the priority queue 'pq' and compute the gradient of the current edge. 
    2 - Activate edge if gradient has at least one component bigger than l1_coeff.
    3 - Reduce the priority of not activated edges.
    """
    tmp_list = []
    while len(pq)>0:
        item = pq.popitem()# Get edges by order of priority
        edge = item[0]
        n = 0
        belief = 0
        for bp in bps:
            bp.load_beliefs()
            belief += bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
            bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
            n += 1
        belief = belief / n
        gradient = (np.exp(belief.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
        activate = not all(i < l1_coeff for i in np.abs(gradient))
        if activate:
            search_space.remove(item[0])
            for item in tmp_list:
                pq.additem(item[0], item[1]+reassignmet)# If an edge is activated, return the previously poped edges with reduced priority
            return True, edge, pq, search_space
        else:
            tmp_list.append(item)# Store not activated edges in a temporary list
    return False, (0, 0), pq, search_space

def get_max_gradient(bps, data_length, curr_search_space, edges_data_sum):
    """
    Fonctionality:
    1 - Compute the gradient for the current weight vector 
    """
    gradient_vec = []
    map_vec = []
    for edge in curr_search_space:
        n = 0
        belief = 0
        for bp in bps:
            bp.load_beliefs()
            belief += bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
            bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
            n += 1
        belief = belief / n
    gradient = (np.exp(belief.reshape((-1, 1)).tolist()) - np.asarray(edges_data_sum[edge]) / data_length).squeeze()
    gradient_vec.extend(gradient)
    for i in range(len(gradient)):
        map_vec.append(edge)
    selected_feature = np.abs(gradient_vec).argmax(axis=0)
    activated_edge = map_vec[selected_feature]
    max_grad = np.abs(gradient_vec).max(axis=0)
    return activated_edge, max_grad

def compute_likelihood(mn, num_nodes, data):
    """
    Functionality:
    1 - Compute the likelihood for the learned MRF
    """
    likelihood = 0
    unary_potentials_copy = copy.deepcopy(mn.unary_potentials)
    for instance in data:
        likelihood_instance = 0
        for curr_node in range(num_nodes):
            inner_exp = copy.deepcopy(unary_potentials_copy[curr_node])
            curr_node_potential = copy.deepcopy(unary_potentials_copy[curr_node])
            likelihood_instance_node = curr_node_potential[instance[curr_node]]
            curr_neighbors = list(mn.get_neighbors(curr_node))
            for t in range(len(curr_neighbors)):
                pair_pots = copy.deepcopy(mn.get_potential((curr_node, curr_neighbors[t])))
                likelihood_instance_node += pair_pots[instance[curr_node], instance[curr_neighbors[t]]]
                inner_exp += pair_pots[:, instance[curr_neighbors[t]]]
            logZ = logsumexp(inner_exp)
            likelihood_instance += likelihood_instance_node - logZ
        likelihood += likelihood_instance
    return - (float(likelihood) / float(len(data)))

def move_to_tail(search_space, edges):
    """
    Functionality:
    1 - Update the search space
    """
    for edge in edges:
        search_space.remove(edge)
        search_space.append(edge)

def reduce_priority(pq, edges):
    """
    Functionality:
    1 - Reduce priority of edges in the priority queue 'pq'
    """
    for edge in edges:
        pq.updateitem(edge, pq[edge]+1)
    return pq