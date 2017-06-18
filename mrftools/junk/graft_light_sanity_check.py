import time
from DataCleaning import load_instances, load_attribute_values
import matplotlib.pyplot as plt
from priority_graft import priority_graft
from grafting_util import compute_likelihood
from queue_grafting import queue_graft
from naive_priority_graft import naive_priority_graft
from mod_priority_graft import mod_priority_graft
from graft_light import graft_light
import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic_data import generate_synthetic_data
from BeliefPropagator import BeliefPropagator


def main():
    number_edges_to_graft = 4
    verbose = True
    # verbose = False
    l1_coeff = 0
    l2_coeff = 0
    edge_reg = 0
    var_reg = 0
    # edge_regs = list(np.arange(.1, 2, .1))
    # l1_coeffs = list(np.arange(.15, .275, .025))
    var_regs = [0]
    edge_regs = [0]
    # var_regs = [0, 1e-2, 1e-1, 1]
    # edge_regs = [0, 1e-2, 1e-1, 1]
    l1_coeffs = [.01]
    recalls, precisions, likelihoods = list(), list(), list()
    print('Simulating data...')
    model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(2000, 4,3,10)
    num_attributes = len(variables)
    train_data = data[: len(data) - 200]
    test_data = data[len(data) - 200 : len(data) - 1]
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
    for var_reg in var_regs:
        for l1_coeff in l1_coeffs:
            for edge_reg in edge_regs:

                print('+++++++++++ edge coeff: ' + str(edge_reg) + '  &&  var coeff: '+ str(var_reg) + '  &&  l1 coeff: '+ str(l1_coeff) + ' ++++++++++++++++++')
                mn_graft, weights_graft, weights_dict_graft, active_space_Graft = graft_light(variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_num_states, verbose, number_edges_to_graft)
                print('recall')
                recall = float(len([x for x in active_space_Graft if x in edges]))/len(edges)
                print(recall)
                print('precision')
                precision = float(len([x for x in edges if x in active_space_Graft]))/len(active_space_Graft)
                print(precision)
                likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                print('Likelihood Graft')
                print(likelihood_graft)
                recalls.append(recall)
                precisions.append(precision)
                likelihoods.append(likelihood_graft)


    # plt.plot(edge_regs, likelihoods)
    # plt.title('likelihood')
    # plt.show()
    # plt.plot(edge_regs, precisions)
    # plt.title('precision')
    # plt.show()
    # plt.plot(edge_regs, recalls)
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