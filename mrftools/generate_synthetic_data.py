import numpy as np
from MarkovNet import MarkovNet
from GibbsSampler import GibbsSampler
from GibbsSamplerMod import GibbsSamplerMod
import itertools
import random
from scipy.stats import rv_continuous
import networkx as nx
import matplotlib.pyplot as plt

def create_custom_random_model(nodes_num, state_min, max_states, edge_std, node_std, mrf_density=.1):
    np.random.seed(0)
    model = MarkovNet()
    edges, variables = list(), list()
    num_states_dict = dict()
    variables = range(nodes_num)
    
    num_states = [np.random.randint(state_min, max_states + 1) for r in range(nodes_num)]
    max_num_states = max(num_states)

    node = 0
    for num in num_states:
        num_states_dict[node] = num
        node += 1

    for node in num_states_dict.keys():
        ###########################
        # model.set_unary_factor(node, np.random.multivariate_normal(mean=np.zeros(num_states_dict[node]), cov=node_std * np.eye(num_states_dict[node], dtype=int), size=1).squeeze())
        ###########################


        # print(np.random.multivariate_normal(mean=np.zeros(num_states_dict[node]), cov=node_std * np.eye(num_states_dict[node], dtype=int), size=1))

        # print(np.random.normal(0, node_std, num_states_dict[node]))

        # model.set_unary_factor(node, np.zeros(num_states_dict[node]))
        model.set_unary_factor(node, np.random.normal(100, node_std, num_states_dict[node]))

        # model.set_unary_factor(node, np.random.randint(30, size=num_states_dict[node]))

    ws = nx.barabasi_albert_graph(nodes_num, 2)
    edges = list()

    for node in ws.nodes():
        for neighbor in ws[node]:
            if node < neighbor:
                edges.append((node, neighbor))
    
    # all_possible_edges = [x for x in list(itertools.product(variables, variables)) if x[0] < x[1]]

    # edges = [all_possible_edges[i] for i in sorted(random.sample(xrange(len(all_possible_edges)), max(int(mrf_density * len(all_possible_edges)), 1)))]
    
    for edge in edges:
        #######################
        # size = num_states_dict[edge[0]] * num_states_dict[edge[1]]
        # random_weights = np.random.multivariate_normal(mean=10*np.ones(size), cov=edge_std * np.eye(size,dtype=int), size=1)
        # resized_random_weights = random_weights.reshape(num_states_dict[edge[0]], num_states_dict[edge[1]])
        # model.set_edge_factor(edge, resized_random_weights)
        #######################

        # factor = random.sample(range(-100,100,10),  (num_states_dict[edge[0]], num_states_dict[edge[1]]))
        factor = np.random.normal(100, edge_std, (num_states_dict[edge[0]], num_states_dict[edge[1]]))
        # print(factor)
        x = []
        [x.append(factor[i,j]) for i in range(num_states_dict[edge[0]]) for j in range(num_states_dict[edge[1]])]
        # print(x)
        model.set_edge_factor(edge, np.random.normal(factor))
        # model.set_edge_factor(edge,np.log(np.random.randint(30, size=(num_states_dict[edge[0]], num_states_dict[edge[1]]))))
        # print(np.random.randn(num_states_dict[num_states_dict[edge[0]], num_states_dict[edge[1]]))
        # print(np.random.randint(5, size=(num_states_dict[edge[0]], num_states_dict[edge[1]])))

        # plt.plot(x, linewidth=1)
        # plt.savefig('../../../factor.png')

    return model, max_num_states, num_states_dict, variables, edges




def generate_synthetic_data(data_points, clusters, nodes_per_cluster, max_states):
    model, max_num_states, num_states_dict, variables, edges = create_model(clusters, nodes_per_cluster, max_states)
    data = sample_data(model, data_points)
    return model, variables, data, max_num_states, num_states_dict, edges


def generate_random_synthetic_data(data_points, nodes_num, edge_std = 1, node_std = 1, mrf_density=.1, state_min=2, state_max=4):
    print('> Creating model')
    model, max_num_states, num_states_dict, variables, edges = create_custom_random_model(nodes_num, state_min, state_min, edge_std, node_std, mrf_density=mrf_density)
    print('> Generating data')
    data = sample_data(model, data_points)
    return model, variables, data, max_num_states, num_states_dict, edges

def create_random_model(nodes_num, state_min, max_states, edge_std, node_std, mrf_density=.1):
    np.random.seed(0)
    model = MarkovNet()
    edges, variables = list(), list()
    num_states_dict = dict()
    
    variables = range(nodes_num)
    
    num_states = [np.random.randint(state_min, max_states + 1) for r in range(nodes_num)]
    max_num_states = max(num_states)

    node = 0
    for num in num_states:
        num_states_dict[node] = num
        node += 1

    for node in num_states_dict.keys():
        model.set_unary_factor(node, np.random.normal(0, node_std, num_states_dict[node]))
        # model.set_unary_factor(node, np.random.randint(30, size=num_states_dict[node]))
    
    all_possible_edges = [x for x in list(itertools.product(variables, variables)) if x[0] < x[1]]

    edges = [all_possible_edges[i] for i in sorted(random.sample(xrange(len(all_possible_edges)), int(mrf_density * len(all_possible_edges))))]
    
    for edge in edges:
        # model.set_edge_factor(edge, .1 * np.random.randn(num_states_dict[edge[0]], num_states_dict[edge[1]]) + 0)
        model.set_edge_factor(edge, np.random.normal(0, edge_std, (num_states_dict[edge[0]], num_states_dict[edge[1]])))
        # model.set_edge_factor(edge,np.log(np.random.randint(30, size=(num_states_dict[edge[0]], num_states_dict[edge[1]]))))
        # print(np.random.randn(num_states_dict[num_states_dict[edge[0]], num_states_dict[edge[1]]))
        # print(np.random.randint(5, size=(num_states_dict[edge[0]], num_states_dict[edge[1]])))

    return model, max_num_states, num_states_dict, variables, edges


def create_model(clusters, nodes_per_cluster, max_states):
    np.random.seed(0)
    model = MarkovNet()
    edges, variables = [], []
    k = 0
    num_states_dict = dict()
    old_outer_node = None
    clusters = clusters
    for cluster in range(clusters):
        for i in range(k + 1, k + nodes_per_cluster):
            edge = (min(k, i), max(k,i))
            edges.append(edge)
        outer_node = k + nodes_per_cluster
        outer_edge = (min(outer_node, i), max(outer_node, i))
        edges.append(outer_edge)
        if old_outer_node:
            edges.append((min(outer_node, old_outer_node), max(outer_node, old_outer_node)))
        k += nodes_per_cluster + 1
        old_outer_node = outer_node
    
    num_states = [np.random.randint(2,max_states) for r in xrange(k)]
    # print('num_states')
    # print(num_states)
    
    node = 0
    for num in num_states:
        num_states_dict[node] =  num
        variables.append(node)
        node += 1

    max_num_states = max(num_states)

    for node in num_states_dict.keys():
        model.set_unary_factor(node, np.random.randn(num_states_dict[node]))

    for edge in edges:
        model.set_edge_factor(edge, np.random.randn(num_states_dict[edge[0]], num_states_dict[edge[1]]))

    return model, max_num_states, num_states_dict, variables, edges

def sample_data(model, data_points):
    sampler = GibbsSamplerMod(model)
    sampler.init_states(0)

    # mix = data_points

    mix = data_points
    num_samples = data_points

    sampler.gibbs_sampling(mix, num_samples)

    samples = sampler.samples

    return samples

def convert_to_binary_data(variables, num_states_dict, data):
    print('binarizing data')
    binary_vars = list()
    binary_to_orginial_hash = dict()
    i = 0
    binary_data = list()
    for var in variables:
        for j in range(i, i + num_states_dict[var]):
            binary_vars.append(j)
            binary_to_orginial_hash[j] = var
        i = i + num_states_dict[var]
    for instance in data:
        binary_instance = {key: 0 for key in binary_vars}
        for var in variables:
            binary_instance[instance[var]] = 1
        binary_data.append(binary_instance)
    max_num_states = 2
    num_states_dict = {key: 2 for key in binary_vars}
    return binary_data, binary_vars, binary_to_orginial_hash, max_num_states, num_states_dict

