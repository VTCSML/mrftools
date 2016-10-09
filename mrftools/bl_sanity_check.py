import time
from DataCleaning import load_instances, load_attribute_values
import matplotlib.pyplot as plt
from priority_graft import priority_graft
from grafting_util import compute_likelihood
from queue_grafting import queue_graft
from naive_priority_graft import naive_priority_graft
from mod_priority_graft import mod_priority_graft
from bl_structure_learning import bl_structure_learning
import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic_data import generate_synthetic_data
from BeliefPropagator import BeliefPropagator
import matplotlib.pyplot as plt


MAX_ITER_GRAFTING = 50000
verbose = True


def main():
    print('Simulating data...')
    model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(10000, 2, 3, 10)#(data_points, clusters, nodes_per_cluster, max_states)
    num_attributes = len(variables)
    train_data = data[: len(data) - 2001]
    test_data = data[len(data) - 2000 : len(data) - 1]
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

    l1_coeff = 0
    l2_coeff = 0
    edge_reg = 0
    var_reg = 0
    likelihood, precision, recall = list(), list(), list()

    # l1_coeffs = list(np.arange(0, .02, .005))
    l1_coeffs = [.01]
    for l1_coeff in l1_coeffs:
        print('l1_coeffs: ' + str(l1_coeff))
        t = time.time()
        mn_graft, weights_graft, weights_dict_graft, active_space_Graft = bl_structure_learning(variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, MAX_ITER_GRAFTING, max_num_states, verbose)
        print(edges)
        print('====================================')
        elapsed_graft = time.time() - t
        print('Baseline ENDED AFTER')
        print(elapsed_graft)
        print('recall')
        curr_recall = float(len([x for x in active_space_Graft if x in edges]))/len(edges)
        print(curr_recall)
        print('precision')
        curr_precision = float(len([x for x in edges if x in active_space_Graft]))/len(active_space_Graft)
        print(curr_precision)
        curr_likelihood = compute_likelihood(mn_graft, num_attributes, test_data)
        print('Likelihood Graft')
        print(curr_likelihood)

    # plt.plot(l1_coeffs, likelihood)
    # plt.title('likelihood')
    # plt.show()
    # plt.plot(l1_coeffs, precision)
    # plt.title('precision')
    # plt.show()
    # plt.plot(l1_coeffs, recall)
    # plt.title('recall')
    # plt.show()



    # bp = BeliefPropagator(model)
    # bp.infer()
    # learned_bp = BeliefPropagator(mn_graft)
    # learned_bp.infer()
    # for var in model.variables:
    #     learned_marg = np.exp(learned_bp.var_beliefs[var])
    #     true_marg = np.exp(bp.var_beliefs[var])
    #     print "Learned vs true marginals for %d:" % var
    #     print learned_marg
    #     print true_marg

if  __name__ =='__main__':
    main()