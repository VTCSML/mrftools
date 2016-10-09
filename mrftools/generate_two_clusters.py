import numpy as np
from MarkovNet import MarkovNet
from GibbsSampler import GibbsSampler

def generate_two_clusters_data(data_points, max_states, basic_edges):
    model, max_num_states, num_states_dict, variables, edges = create_model(max_states, basic_edges)
    data = sample_data(model, data_points)
    return model, variables, data, max_num_states, num_states_dict, edges

def create_model(max_states, basic_edges):
    np.random.seed(0)
    model = MarkovNet()
    edges, variables = [], []
    k = 0
    num_states_dict = dict()
    edges = basic_edges

    # Connect one central node to neighbors of the other

    # edges.extend([(0,7), (0,8), (0,9), (0,10), (0,11)])
    # edges.extend([(12,1), (12,2), (12,3), (12,4)])

    inter_cluster_edges = []
    # Connect neighbors
    # inter_cluster_edges.extend([(1,7), (1,8), (1,9), (1,10), (1,11), (1,12)])
    # inter_cluster_edges.extend([(2,7), (2,8), (2,9), (2,10), (2,11), (2,12)])
    # inter_cluster_edges.extend([(3,7), (3,8), (3,9), (3,10), (3,11), (3,12)])
    # inter_cluster_edges.extend([(4,7), (4,8), (4,9), (4,10), (4,11), (4,12)])
    # inter_cluster_edges.extend([(5,7), (5,8), (5,9), (5,10), (5,11), (5,12)])

    edges.extend(inter_cluster_edges)

    k = 14
    num_states = [np.random.randint(2,max_states) for r in xrange(k)]
    
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