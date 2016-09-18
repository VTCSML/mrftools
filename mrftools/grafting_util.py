import numpy as np
from scipy.misc import logsumexp
import copy
from graph_mining_util import *
from pqdict import pqdict

def priority_reassignment(variables, active_set, aml_optimize , prune_threshold, data, search_space, pq, l1_coeff, edges_data_sum):
    """
    Functionality :
    1 - Select edge to be tested. 
    2 - Activate edge if if it passes the pruning test.
    3 - Decrease the priority of the edge and resulting edges if it does not pass the pruning test. 
    """
    curr_graph = make_graph(variables, active_set)
    injection = False
    success = False
    found, selected_edge, resulting_edges = select_edge_to_inject(curr_graph, search_space, prune_threshold)
    if found:
        injection = True
        print('Testing priority Reassignment possibility using edge:')
        print(selected_edge)
        added_edge = edge_gradient_test(aml_optimize.belief_propagators, selected_edge, edges_data_sum, data, l1_coeff)
        if addedEdge:
            print('priority Reassignment NOT Authorized')
            pq.pop(selected_edge)
            search_Space.remove(selected_edge)
            active_set.append(selected_edge)
            # print('ACTIVATED EDGE')
            # print(selected_edge)
            # print('ACTIVE SPACE')
            # print(active_Set)
        else:
            success = True
            print('priority Reassignment Authorized')
            # print('priority Reassigned Edges:')
            # print(resulting_edges)
            pq.pop(selected_edge)
            search_space.remove(selected_edge)
            pq = reduce_priority(pq, resulting_edges)
    return pq, active_set, search_space, injection, success, resulting_edges

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

def priority_gradient_test(bps, search_space, pq, data_sum, data, l1_coeff):
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
        gradient = (np.exp(belief.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
        activate = not all(i < l1_coeff for i in np.abs(gradient))
        if activate:
            search_space.remove(item[0])
            for item in tmp_list:# If an edge is activated, return the previously poped edges
                pq.additem(item[0],item[1])
            return True, edge
        else: tmp_list.append(item)# Store not activated edges in a temporary list
    return False, (0, 0)

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
            return True, edge, tmp_list
        else:
            tmp_list.append(item)# Store not activated edges in a temporary list
    return False, (0, 0), tmp_list

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
    curr_search_space.remove(activated_edge)
    return activated_edge, max_grad, curr_search_space

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
            for neighbor in curr_neighbors:
                pair_pots = copy.deepcopy(mn.get_potential((curr_node, neighbor)))
                likelihood_instance_node += pair_pots[instance[curr_node], instance[neighbor]]
                inner_exp += pair_pots[:, instance[neighbor]]
            logZ = logsumexp(inner_exp)
            likelihood_instance += likelihood_instance_node - logZ
        likelihood += likelihood_instance
    return - (float(likelihood) / float(len(data)))


def initialize_priority_queue(mn):
    """
    Initialize priority queue for grafting
    """
    pq = pqdict()
    for edge in mn.search_space:
        pq.additem(edge, 0)
    return pq

def setup_learner(aml_optimize, data):
    for instance in data:
        aml_optimize.add_data(instance)

def update_grafting_metrics(injection, success, resulting_edges, edges_reassigned, num_success, num_injection, num_edges_reassigned, priority_reassignements):
    if injection:
        num_injection += 1
        if success:
            num_success += 1
            priority_reassignements += len(resulting_edges)
            new_edges_reassigned = [x for x in resulting_edges if x not in edges_reassigned]
            num_edges_reassigned += len(new_edges_reassigned)
            edges_reassigned.extend(new_edges_reassigned)
    return num_success, num_injection, priority_reassignements, num_edges_reassigned


def update_mod_grafting_metrics(injection, success, resulting_edges, edges_reassigned, graph_edges_reassigned, num_success, num_injection, num_edges_reassigned, priority_reassignements):
    if injection:
        num_injection += 1
        if success:
            num_success += 1
            priority_reassignements += len(resulting_edges)
            new_edges_reassigned = [x for x in resulting_edges if x not in edges_reassigned]
            num_edges_reassigned += len(new_edges_reassigned)
            edges_reassigned.extend(new_edges_reassigned)
            graph_edges_reassigned.extend(new_edges_reassigned)
    return num_success, num_injection, priority_reassignements, num_edges_reassigned

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