import numpy as np
from scipy.misc import logsumexp
import copy
from graph_mining_util import *
from pqdict import pqdict
from ApproxMaxLikelihood import ApproxMaxLikelihood

def priority_reassignment(variables, active_set, aml_optimize , prune_threshold, data, search_space, pq, l1_coeff, edges_data_sum, mn):
    """
    Functionality :
    1 - Select edge to be tested. 
    2 - Activate edge if if it passes the pruning test.
    3 - Decrease the priority of the edge and resulting edges if it does not pass the pruning test. 
    """
    curr_graph = make_graph(variables, active_set)
    injection = False
    success = False
    added_edge = False
    found, selected_edge, resulting_edges = select_edge_to_inject(curr_graph, search_space, prune_threshold)
    if found:
        injection = True
        # print('Testing priority Reassignment possibility using edge:')
        # print(selected_edge)
        added_edge = edge_gradient_test(aml_optimize.belief_propagators[0], selected_edge, edges_data_sum, data, l1_coeff)
        if added_edge:
            # print('priority Reassignment NOT Authorized')
            pq.pop(selected_edge)
            search_space.remove(selected_edge)
            active_set.append(selected_edge)
            mn.set_edge_factor(selected_edge, np.zeros((len(mn.unary_potentials[selected_edge[0]]), len(mn.unary_potentials[selected_edge[1]]))))
            # print('ACTIVATED EDGE')
            # print(selected_edge)
            # print('ACTIVE SPACE')
            # print(active_Set)
        else:
            success = True
            # print('priority Reassignment Authorized')
            # print('priority Reassigned Edges:')
            # print(resulting_edges)
            pq.pop(selected_edge)
            search_space.remove(selected_edge)
            pq = reduce_priority(pq, resulting_edges)
    return pq, active_set, search_space, injection, success, added_edge, resulting_edges



def naive_priority_mean_gradient_test(bps, search_space, pq, data_sum, data, l1_coeff, reassignmet):
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
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
        mean_gradient = np.sqrt(gradient.dot(gradient) / len(gradient))
        activate = mean_gradient > l1_coeff
        if activate:
            search_space.remove(item[0])
            for item in tmp_list:
                pq.additem(item[0], item[1]+reassignmet)# If an edge is activated, return the previously poped edges with reduced priority
            return True, edge, tmp_list
        else:
            tmp_list.append(item)# Store not activated edges in a temporary list
    return False, (0, 0), tmp_list

def edge_gradient_test(bp, edge, data_sum, data, l1_coeff):
    """
    Functionality :
    1 - Compute the gradient of the current edge. 
    2 - Activate edge if gradient has at least one component bigger than l1_coeff.
    """
    bp.load_beliefs()
    belief = bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(
    bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
    gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
    mean_gradient = np.sqrt(gradient.dot(gradient) / len(gradient))
    # print('GRADIENT')
    # print(mean_gradient)
    remove = mean_gradient < l1_coeff
    if remove:
        return False
    return True


def priority_mean_gradient_test(bps, search_space, pq, data_sum, data, l1_coeff):
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
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
        mean_gradient = np.sqrt(gradient.dot(gradient) / len(gradient))
        activate = mean_gradient > l1_coeff
        if activate:
            search_space.remove(item[0])
            for item in tmp_list:# If an edge is activated, return the previously poped edges
                pq.additem(item[0],item[1])
            return True, edge, mean_gradient
        else: tmp_list.append(item)# Store not activated edges in a temporary list
    return False, (0, 0), 0


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
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
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
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(data_sum[edge]) / len(data)).squeeze()
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
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(edges_data_sum[edge]) / data_length).squeeze()
        gradient_vec.extend(gradient)
        for i in range(len(gradient)):
            map_vec.append(edge)
    selected_feature = np.abs(gradient_vec).argmax(axis=0)
    activated_edge = map_vec[selected_feature]
    max_grad = np.abs(gradient_vec).max(axis=0)
    curr_search_space.remove(activated_edge)
    return activated_edge, max_grad, curr_search_space


def get_sorted_mean_gradient_gl(bps, data_length, curr_search_space, edges_data_sum, l1_coeff, edges_to_graft_number):
    """
    Fonctionality:
    1 - Compute the gradient for the current weight vector for grafting light 
    """
    edge_mean_weights = []
    activated_edges = list()
    for edge in curr_search_space:
        bp = bps[0]
        bp.load_beliefs()
        belief = bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(edges_data_sum[edge]) / data_length).squeeze()
        # print(edge)
        mean_gradient = np.sqrt(gradient.dot(gradient) / len(gradient))
        # mean_gradient = gradient.dot(gradient) / len(gradient)
        # print(mean_gradient)
        edge_mean_weights.append((edge, mean_gradient))
    edge_mean_weights.sort(key=operator.itemgetter(1), reverse=True)
    sorted_mean_edges = [x[0] for x in edge_mean_weights]
    sorted_mean_gradients = [x[1] for x in edge_mean_weights]
    # print(sorted_mean_edges)
    [activated_edges.append(sorted_mean_edges[i]) for i in range(min(edges_to_graft_number, len(sorted_mean_gradients))) if sorted_mean_gradients[i] > l1_coeff]
    for edge in activated_edges:
        curr_search_space.remove(edge)
    return activated_edges, curr_search_space


def get_max_mean_gradient(bps, data_length, curr_search_space, edges_data_sum):
    """
    Fonctionality:
    1 - Compute the gradient for the current weight vector 
    """
    edge_mean_weights = []
    gradient_vec = []
    map_vec = []
    for edge in curr_search_space:
        bp = bps[0]
        bp.load_beliefs()
        belief = bp.var_beliefs[edge[0]] - bp.mn.unary_potentials[edge[0]] + np.matrix(bp.var_beliefs[edge[1]] - bp.mn.unary_potentials[edge[1]]).T
        gradient = (np.exp(belief.T.reshape((-1, 1)).tolist()) - np.asarray(edges_data_sum[edge]) / data_length).squeeze()
        # print(edge)
        mean_gradient = np.sqrt(gradient.dot(gradient) / len(gradient))
        # mean_gradient = gradient.dot(gradient) / len(gradient)
        # print(mean_gradient)
        edge_mean_weights.append((edge, mean_gradient))
        gradient_vec.append(mean_gradient)
        map_vec.append(edge)
    edge_mean_weights.sort(key=operator.itemgetter(1), reverse=True)
    sorted_mean_edges = [x[0] for x in edge_mean_weights]
    # print(sorted_mean_edges)
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


def setup_learner_1(mn, l1_coeff, l2_coeff, var_reg, edge_reg, padded_sufficient_stats, len_data, active_set):
    aml_optimize = ApproxMaxLikelihood(mn) #Create a new 'ApproxMaxLikelihood' object at each iteration using the updated markov network
    aml_optimize.set_regularization(l1_coeff, l2_coeff, var_reg, edge_reg)
    aml_optimize.init_grafting()
    unary_indices, pairwise_indices = aml_optimize.belief_propagators[0].mn.get_weight_factor_index()
    tau_q = np.zeros(aml_optimize.weight_dim)
    for var in mn.variables:
        i = aml_optimize.belief_propagators[0].mn.var_index[var]
        inds = unary_indices[:, i]
        tau_q[inds] = padded_sufficient_stats[var] / len_data
    for edge in active_set:
        i = aml_optimize.belief_propagators[0].mn.edge_index[edge]
        inds = pairwise_indices[:, :, i]
        tau_q[inds] = padded_sufficient_stats[edge] / len_data
    aml_optimize.set_sufficient_stats(tau_q)
    return aml_optimize


def reset_unary_factors(mn, mn_old):
    for var in mn_old.variables:
        mn.set_unary_factor(var, mn_old.unary_potentials[var])

def reset_edge_factors(mn, mn_old, active_set):
    for edge in active_set:
        mn.set_edge_factor(edge, mn_old.edge_potentials[edge])
