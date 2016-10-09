import time
import matplotlib.pyplot as plt
from mod_priority_graft import mod_priority_graft
from graft import graft
from bl_structure_learning import bl_structure_learning
from grafting_util import compute_likelihood
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from BeliefPropagator import BeliefPropagator
from priority_graft import priority_graft
from naive_priority_graft import naive_priority_graft


def main():
    verbose = False
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
    max_grafting_iter = 2500
    max_priority_grafting_iter = 50
    max_baseline_iter = 2500

    graft_recall_vec, graft_precision_vec, graft_likelihood_vec = list(), list(), list()
    naive_graft_recall_vec, naive_graft_precision_vec, naive_graft_likelihood_vec = list(), list(), list()
    priority_graft_recall_vec, priority_graft_precision_vec, priority_graft_likelihood_vec = list(), list(), list()
    mod_graft_recall_vec, mod_graft_precision_vec, mod_graft_likelihood_vec = list(), list(), list()
    bl_recall_vec, bl_precision_vec, bl_likelihood_vec = list(), list(), list()
    t_priority_graft_vec, t_naive_graft_vec, t_mod_graft_vec, t_graft_vec = list(), list(), list(),list()

    num_cluster_range = range(1, 10, 1)
    num_nodes_vec = []
    for num_cluster in num_cluster_range:
        model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(10000, num_cluster,5,10)
        num_attributes = len(variables)
        print('==================================================== '+ str(num_attributes) +' NODES ==================================================== ')
        print('Simulating data...')
        train_data = data[: len(data) - 201]
        test_data = data[len(data) - 200 : len(data) - 1]
        print('Number of Data Points')
        print(len(data))
        print('Number of States per Attribute')
        print(num_states)
        print('Edges')
        print(edges)
        num_nodes_vec.append(len(variables))
        for var_reg in var_regs:
            for l1_coeff in l1_coeffs:
                for edge_reg in edge_regs:


                    # print('baseline +++++++++++ edge coeff: ' + str(edge_reg) + '  &&  var coeff: '+ str(var_reg) + '  &&  l1 coeff: '+ str(l1_coeff) + ' ++++++++++++++++++')
                    # mn_graft, weights_graft, weights_dict_graft, active_space_bl = bl_structure_learning(variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_baseline_iter, max_num_states, verbose)
                    # print(edges)
                    # print('recall')
                    # curr_recall = float(len([x for x in active_space_bl if x in edges]))/len(edges)
                    # print(curr_recall)
                    # print('precision')
                    # curr_precision = float(len([x for x in edges if x in active_space_bl]))/len(active_space_bl)
                    # print(curr_precision)
                    # curr_likelihood = compute_likelihood(mn_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(curr_likelihood)

                    print(' ----------------------------------- PRIORITY GRAFT (STRUCTURE) -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_priority_graft = time.time()
                    mn_graft, active_space_mod_Graft = priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg,  2, .5, max_priority_grafting_iter, max_num_states, verbose)
                    t_priority_graft = time.time() - t_priority_graft
                    print('recall')
                    recall = float(len([x for x in active_space_mod_Graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_mod_Graft]))/len(active_space_mod_Graft)
                    print(precision)
                    likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_graft)
                    priority_graft_recall_vec.append(recall)
                    priority_graft_precision_vec.append(precision)
                    priority_graft_likelihood_vec.append(likelihood_graft)
                    t_priority_graft_vec.append(t_priority_graft)


                    

                    print(' ----------------------------------- GENERAL PRIORITY GRAFT -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_mod_graft = time.time()
                    mn_graft, active_space_mod_Graft = mod_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg,  2, .5, max_priority_grafting_iter, max_num_states, verbose)
                    t_mod_graft = time.time() - t_mod_graft
                    print('recall')
                    recall = float(len([x for x in active_space_mod_Graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_mod_Graft]))/len(active_space_mod_Graft)
                    print(precision)
                    likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_graft)
                    mod_graft_recall_vec.append(recall)
                    mod_graft_precision_vec.append(precision)
                    mod_graft_likelihood_vec.append(likelihood_graft)
                    t_mod_graft_vec.append(t_mod_graft)


                    print(' ----------------------------------- PRIORITY GRAFT (NAIVE) -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_naive_graft = time.time()
                    mn_graft, active_space_naive_Graft = naive_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose)
                    t_naive_graft = time.time() - t_naive_graft
                    print('recall')
                    recall = float(len([x for x in active_space_naive_Graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_naive_Graft]))/len(active_space_naive_Graft)
                    print(precision)
                    likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_graft)
                    naive_graft_recall_vec.append(recall)
                    naive_graft_precision_vec.append(precision)
                    naive_graft_likelihood_vec.append(likelihood_graft)
                    t_naive_graft_vec.append(t_naive_graft)

                    print(' ----------------------------------- GRAFT -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_graft = time.time()
                    mn_graft, active_space_Graft = graft(variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_grafting_iter, max_num_states, verbose)
                    t_graft = time.time() - t_graft
                    print('recall')
                    recall = float(len([x for x in active_space_Graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_Graft]))/len(active_space_Graft)
                    print(precision)
                    likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_graft)
                    graft_recall_vec.append(recall)
                    graft_precision_vec.append(precision)
                    graft_likelihood_vec.append(likelihood_graft)
                    t_graft_vec.append(t_graft)

                    print('~~~~~~~~~ EXEC TIME ~~~~~~~~~')
                    print('PRIORITY GRAFT (STRUCTURE) time')
                    print(t_priority_graft)

                    print('PRIORITY GRAFT (NAIVE) time')
                    print(t_naive_graft)

                    print('GENERAL PRIORITY GRAFT time')
                    print(t_mod_graft)

                    print('GRAFT time')
                    print(t_graft)

    print('+++++++++++ GRAFT +++++++++++')
    print('Exec time')
    print(t_graft_vec)
    print('Number of nodes')
    print(num_nodes_vec)
    print('Likelihoods')
    print(graft_likelihood_vec)
    print('recall')
    print(graft_recall_vec)
    print('precision')
    print(graft_precision_vec)

    print('+++++++++++ PRIORITY GRAFT (NAIVE)+++++++++++')
    print('Exec time')
    print(t_naive_graft_vec)
    print('Number of nodes')
    print(num_nodes_vec)
    print('Likelihoods')
    print(naive_graft_likelihood_vec)
    print('recall')
    print(naive_graft_recall_vec)
    print('precision')
    print(naive_graft_precision_vec)

    print('+++++++++++ PRIORITY GRAFT (STRUCTURE) +++++++++++')
    print('Exec time')
    print(t_priority_graft_vec)
    print('Likelihoods')
    print('Number of nodes')
    print(num_nodes_vec)
    print(priority_graft_likelihood_vec)
    print('recall')
    print(priority_graft_recall_vec)
    print('precision')
    print(priority_graft_precision_vec)

    print('+++++++++++ GENERAL PRIORITY GRAFT +++++++++++')
    print('Exec time')
    print(t_mod_graft_vec)
    print('Likelihoods')
    print('Number of nodes')
    print(num_nodes_vec)
    print(mod_graft_likelihood_vec)
    print('recall')
    print(mod_graft_recall_vec)
    print('precision')
    print(mod_graft_precision_vec)




if  __name__ =='__main__':
    main()