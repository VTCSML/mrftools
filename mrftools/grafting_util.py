import numpy as np
# from scipy.misc import logsumexp
import copy
from graph_mining_util import *
from pqdict import pqdict
from ApproxMaxLikelihood import ApproxMaxLikelihood
import time
from random import shuffle, uniform
from sklearn.metrics import accuracy_score




def compute_likelihood_1(mn, num_nodes, data):
    """
    Functionality:
    1 - Compute the likelihood for the learned MRF
    """
    data_copy = copy.deepcopy(data)
    new_active_set = list(mn.edge_potentials.keys()) 
    unary_potentials_copy = copy.deepcopy(mn.unary_potentials)
    likelihood_1 = 0
    for instance in data_copy:
        likelihood_instance_1 = 0
        eliminated = []
        for curr_node in range(num_nodes):
            curr_node_potential = copy.deepcopy(unary_potentials_copy[curr_node])
            inner_exp_1 = np.zeros(len(curr_node_potential[:]))
            likelihood_instance_node_1 = 0
            curr_neighbors = list(mn.get_neighbors(curr_node)) 
            has_neighbor = len(curr_neighbors) > 0
            # print(has_neighbor)
            if has_neighbor:
                for neighbor in curr_neighbors:
                    pair_pots = copy.deepcopy(mn.get_potential((curr_node, neighbor)))
                    likelihood_instance_node_1 += copy.deepcopy(pair_pots[instance[curr_node], instance[neighbor]])
                    inner_exp_1 += copy.deepcopy(pair_pots[:, instance[neighbor]])
            else:
                inner_exp_1 = copy.deepcopy(curr_node_potential[:])
                likelihood_instance_node_1 = copy.deepcopy(curr_node_potential[instance[curr_node]])
            logZ_1 = logsumexp(inner_exp_1)[0]
            # print(logZ_1)
            likelihood_instance_1 += (likelihood_instance_node_1 - logZ_1)

        likelihood_1 += likelihood_instance_1
    nll = - (float(likelihood_1) / float(len(data)))
    print(nll)
    return nll



def compute_likelihood(mn, num_nodes, data, variables = None):
    """
    Functionality:
    1 - Compute the likelihood for the learned MRF
    """
    if variables == None:
        variables = range(num_nodes)
    # print(len( mn.edge_potentials))
    # print(mn.neighbors)
    data_copy = copy.deepcopy(data)
    new_active_set = list(mn.edge_potentials.keys()) 
    # new_edge = [x for x in new_active_set if x not in old_active_set]
    unary_potentials_copy = copy.deepcopy(mn.unary_potentials)
    likelihood = 0
    likelihood_1 = 0
    total_diff = 0
    for instance in data_copy:
        # try:
        #     old_node_likelihoods = history[i]
        # except:
        #     old_node_likelihoods = {}
        likelihood_instance = 0
        likelihood_instance_1 = 0
        eliminated = []
        for curr_node in range(num_nodes):
            # eliminated.append(curr_node)
            curr_node_potential = copy.deepcopy(unary_potentials_copy[variables[curr_node]])
            # print(curr_node_potential[:])
            inner_exp = np.exp(curr_node_potential[:])
            # print(inner_exp)
            inner_exp_1 = curr_node_potential[:]
            ####################
            likelihood_instance_node = np.exp(curr_node_potential[instance[variables[curr_node]]])
            likelihood_instance_node_1 = copy.deepcopy(curr_node_potential[instance[variables[curr_node]]])
            ####################
            # curr_neighbors = [x for x in list(mn.get_neighbors(curr_node)) if x not in eliminated]
            curr_neighbors = list(mn.get_neighbors(variables[curr_node])) 
            has_neighbor = len(curr_neighbors) > 0
            # if has_neighbor:
            #     test_instances = copy.deepcopy(data_copy)
            for neighbor in curr_neighbors:
                # new_test_instances = [x for x in test_instances if x[neighbor] == instance[neighbor]]
                # test_instances = new_test_instances
                pair_pots = copy.deepcopy(mn.get_potential((variables[curr_node], neighbor)))
                ####################
                likelihood_instance_node = likelihood_instance_node * np.exp(copy.deepcopy(pair_pots[instance[variables[curr_node]], instance[neighbor]]))
                likelihood_instance_node_1 = likelihood_instance_node_1 + copy.deepcopy(pair_pots[instance[variables[curr_node]], instance[neighbor]])
                # print('////')
                # print(inner_exp)
                # print(np.exp(copy.deepcopy(pair_pots[:, instance[neighbor]])))
                inner_exp = inner_exp * np.exp(copy.deepcopy(pair_pots[:, instance[neighbor]]))
                # print(inner_exp)
                inner_exp_1 = inner_exp_1 + copy.deepcopy(pair_pots[:, instance[neighbor]])


            # if has_neighbor:
            #     conditioned_instances = [x for x in test_instances if x[curr_node] == instance[curr_node]]

            logZ_1 = logsumexp(inner_exp_1)
            # try:
            #     # print(old_node_likelihoods[curr_node])
            #     diff = likelihood_instance_node / np.sum(inner_exp) - old_node_likelihoods[curr_node]
            #     total_diff += diff
            #     # if np.abs(diff)>1e-3:
            #         # print(new_edge)
            #         # print((curr_node, instance[curr_node]))
            #         # print(diff)
            # except:
            #     pass
            #     # print('Nan')

            # print(curr_node)
            # print(has_neighbor)
            # print(np.log(likelihood_instance_node / np.sum(inner_exp)))

            likelihood_instance = likelihood_instance + np.log(likelihood_instance_node / np.sum(inner_exp))
            likelihood_instance_1 = likelihood_instance_1 + likelihood_instance_node_1 - logZ_1
        # history[i] = old_node_likelihoods
        # i += 1
        likelihood = likelihood + likelihood_instance
        likelihood_1 = likelihood_1 + likelihood_instance_1
        # print('..')
        # print(likelihood)
        # print(likelihood_1)
    # nll = - (float(likelihood) / float(len(data)))
    nll = - (float(likelihood_1[0]) / float(len(data)))
    print(nll)
    return nll


def sanity_check_likelihood(mn, num_nodes, data):
    """
    Functionality:
    1 - Compute the likelihood for the learned MRF
    """
    unary_potentials_copy = copy.deepcopy(mn.unary_potentials)
    first = True
    instance = data[0]
    for curr_node in range(num_nodes):
        likelihood_node = 0
        inner_exp = np.exp(curr_node_potential[:])
        curr_node_potential = copy.deepcopy(unary_potentials_copy[curr_node])
        for i in range(len(unary_potentials_copy[curr_node])):
            likelihood_instance_node = curr_node_potential[i]
            curr_neighbors = list(mn.get_neighbors(curr_node))
            for neighbor in curr_neighbors:
                pair_pots = copy.deepcopy(mn.get_potential((curr_node, neighbor)))
                likelihood_instance_node += pair_pots[i, instance[neighbor]]
                inner_exp += pair_pots[:, instance[neighbor]]
            logZ = logsumexp(inner_exp)
            likelihood_node += likelihood_instance_node - logZ
        print(np.exp(likelihood_node))
    return 0



def compute_accuracy_synthetic(mn, variables, data, variable, t):
    """
    Functionality:
    1 - Compute the likelihood for the learned MRF
    """
    data_copy = copy.deepcopy(data)
    likelihood = 0
    unary_potentials_copy = copy.deepcopy(mn.unary_potentials)
    results = []
    true_states, predicted_states = list(), list()
    for instance in data_copy:
        likelihoods = dict()
        real_state = instance[variable]
        for i in range(len(mn.unary_potentials[variable])):
            # print(i)
            instance[variable] = i
            likelihood_instance = 0
            for curr_node in variables:
                curr_node_potential = copy.deepcopy(unary_potentials_copy[curr_node])
                inner_exp = np.exp(curr_node_potential[:])
                likelihood_instance_node = np.exp(curr_node_potential[instance[curr_node]])
                curr_neighbors = list(mn.get_neighbors(curr_node))
                # if curr_node == 0 and t==0:
                #     print(curr_node_potential[:])
                for neighbor in curr_neighbors:
                    pair_pots = copy.deepcopy(mn.get_potential((curr_node, neighbor)))
                    likelihood_instance_node = likelihood_instance_node * np.exp(pair_pots[instance[curr_node], instance[neighbor]])
                    inner_exp = inner_exp * np.exp(copy.deepcopy(pair_pots[:, instance[neighbor]]))
                logZ = np.log(np.sum(inner_exp))
                # if curr_node == 0:
                    # print(np.log(likelihood_instance_node / np.sum(inner_exp)))
                likelihood_instance += np.log(likelihood_instance_node / np.sum(inner_exp))
            likelihoods[i] = likelihood_instance
        likely_state = max(likelihoods.iteritems(), key=operator.itemgetter(1))[0]
        predicted_states.append(likely_state)
        true_states.append(real_state)
    #     print(likelihoods)
    # print(predicted_states)
    # print(true_states)
    return accuracy_score(true_states, predicted_states) , true_states, predicted_states



def compute_accuracy(mn, variables, data, variable, unobserved_state):
    """
    Functionality:
    1 - Compute the likelihood for the learned MRF
    """
    likelihood = 0
    unary_potentials_copy = copy.deepcopy(mn.unary_potentials)
    results = []
    true_states, predicted_states = list(), list()
    for instance in data:
        if instance[variable] != unobserved_state:
            likelihoods = dict()
            real_state = instance[variable]
            for i in range(len(mn.unary_potentials[variable])):
                instance[variable] = i
                likelihood_instance = 0
                for curr_node in variables:
                    inner_exp = copy.deepcopy(unary_potentials_copy[curr_node])
                    curr_node_potential = copy.deepcopy(unary_potentials_copy[curr_node])
                    likelihood_instance_node = curr_node_potential[instance[curr_node]]
                    curr_neighbors = list(mn.get_neighbors(curr_node))
                    for neighbor in curr_neighbors:
                        pair_pots = copy.deepcopy(mn.get_potential((curr_node, neighbor)))
                        likelihood_instance_node += pair_pots[instance[curr_node], instance[neighbor]]
                        inner_exp = inner_exp * np.exp(copy.deepcopy(pair_pots[:, instance[neighbor]]))
                    logZ = logsumexp(inner_exp)
                    likelihood_instance += likelihood_instance_node - logZ
                likelihoods[i] = likelihood_instance
            # print('likelihoods')
            # print(likelihoods)
            likely_state = max(likelihoods.iteritems(), key=operator.itemgetter(1))[0]
            if likely_state == real_state:
                results.append(1)
            else:
                results.append(1)
            predicted_states.append(likely_state)
            true_states.append(real_state)
    return accuracy_score(true_states, predicted_states) , true_states, predicted_states


def get_all_ss(variables, num_states, data):
    ss_test = dict()
    for var1 in variables:
        for var2 in variables:
            if var1 < var2:
                edge = (var1, var2)
                edge_sufficient_stats = np.asarray(np.zeros((num_states[edge[0]], num_states[edge[1]])).reshape((-1, 1)))
                for states in data:
                    table = np.zeros((num_states[edge[0]], num_states[edge[1]]))
                    table[states[edge[0]], states[edge[1]]] = 1
                    tmp = np.asarray(table.reshape((-1, 1)))
                    edge_sufficient_stats += tmp
                    ss_test[edge] = edge_sufficient_stats
    return ss_test


def initialize_priority_queue(search_space=None, variables=list()):
    """
    Initialize priority queue for grafting
    """
    pq = pqdict()
    if search_space == None:
        search_space = list()
        for var1 in variables:
            for var2 in variables:
                if var1<var2:
                    edge = (var1, var2)
                    pq.additem(edge, uniform(0,1e-5))
    else:
        for edge in search_space:
            pq.additem(edge, 0)
    return pq

def reset_unary_factors(mn, mn_old):
    for var in mn_old.variables:
        mn.set_unary_factor(var, mn_old.unary_potentials[var])

def reset_edge_factors(mn, mn_old, active_set):
    for edge in active_set:
        mn.set_edge_factor(edge, mn_old.edge_potentials[edge])

def get_sufficient_stats_per_edge(mn, data, max_states, edge):
        """Compute joint states reoccurrences in the data"""
        edge_padded_sufficient_stats = np.asarray(np.zeros((max_states, max_states)).reshape((-1, 1)))
        edge_sufficient_stats = np.asarray(np.zeros((len(mn.unary_potentials[edge[0]]), (len(mn.unary_potentials[edge[1]])))).reshape((-1, 1)))
        for states in data:
            padded_table = np.zeros((max_states, (max_states)))
            padded_table[states[edge[0]], states[edge[1]]] = 1
            padded_tmp = np.asarray(padded_table.reshape((-1, 1)))
            table = np.zeros((len(mn.unary_potentials[edge[0]]), (len(mn.unary_potentials[edge[1]]))))
            table[states[edge[0]], states[edge[1]]] = 1
            tmp = np.asarray(table.reshape((-1, 1)))
            edge_sufficient_stats += tmp
            edge_padded_sufficient_stats += padded_tmp
        return edge_sufficient_stats, edge_padded_sufficient_stats


def logsumexp(matrix, dim = None):
    """Compute log(sum(exp(matrix), dim)) in a numerically stable way."""
    try:
        with np.errstate(over='raise', under='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val


