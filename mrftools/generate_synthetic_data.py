import numpy as np
from MarkovNet import MarkovNet
from GibbsSampler import GibbsSampler

def generate_synthetic_data(data_points, clusters, nodes_per_cluster, max_states):
    model, max_num_states, num_states_dict, variables, edges = create_model(clusters, nodes_per_cluster, max_states)
    data = sample_data(model, data_points)
    return model, variables, data, max_num_states, num_states_dict, edges

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

    # edges = [(0,1), (1,2), (3,5), (3,4), (4,8), (5,8), (2,6)]
    # k = 10
    # num_states = [4, 3, 2, 2, 3, 5, 2, 3, 2, 4]
    
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
    sampler = GibbsSampler(model)
    sampler.init_states(0)

    mix = data_points
    num_samples = data_points

    sampler.gibbs_sampling(mix, num_samples)

    return sampler.samples
