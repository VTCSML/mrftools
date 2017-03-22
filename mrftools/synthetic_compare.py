import time
import matplotlib.pyplot as plt
# from graft import graft
# from bl_structure_learning import bl_structure_learning
from grafting_util import compute_likelihood
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from BeliefPropagator import BeliefPropagator
# from priority_graft import priority_graft
# from naive_priority_graft import naive_priority_graft
# from strcutured_priority_graft import strcutured_priority_graft
# from new_naive_priority_graft import new_naive_priority_graft
# from new_strcutured_priority_graft import new_strcutured_priority_graft
from random import shuffle
from graph_mining_util import draw_graph, make_graph
from scipy import signal as sg 
from new_naive_priority_gradient_graft import new_naive_priority_gradient_graft
from queue_graft import queue_graft
from new_strcutured_priority_gradient_graft import new_strcutured_priority_gradient_graft



def main():
    plot_queue = True
    zero_threshold = .05
    verbose = False
    # verbose = True
    l2_coeff = 0
    # edge_regs = list(np.arange(.1, 2, .1))
    # l1_coeffs = list(np.arange(.15, .275, .025))
    var_regs = [0]
    edge_regs = [.075]
    # var_regs = [0, 1e-2, 1e-1, 1]
    # edge_regs = [0, 1e-2, 1e-1, 1]

    # l1_coeffs = [.01]
    l1_coeffs = [0]
    max_grafting_iter = 2500
    max_priority_grafting_iter = 500
    max_baseline_iter = 2500

    strcutured_priority_graft_recall_vec, strcutured_priority_graft_precision_vec, strcutured_priority_graft_likelihood_vec, t_strcutured_priority_graft_vec = list(), list(), list(), list()
    graft_recall_vec, graft_precision_vec, graft_likelihood_vec = list(), list(), list()
    naive_graft_recall_vec, naive_graft_precision_vec, naive_graft_likelihood_vec = list(), list(), list()
    priority_graft_recall_vec, priority_graft_precision_vec, priority_graft_likelihood_vec = list(), list(), list()
    mod_graft_recall_vec, mod_graft_precision_vec, mod_graft_likelihood_vec = list(), list(), list()
    bl_recall_vec, bl_precision_vec, bl_likelihood_vec = list(), list(), list()
    t_priority_graft_vec, t_naive_graft_vec, t_mod_graft_vec, t_graft_vec = list(), list(), list(),list()
    new_strcutured_priority_graft_recall_vec, new_strcutured_priority_graft_precision_vec, new_strcutured_priority_graft_likelihood_vec, t_new_strcutured_priority_graft_vec = list(), list(), list(), list()
    new_naive_graft_recall_vec, new_naive_graft_precision_vec, new_naive_graft_likelihood_vec, t_new_naive_graft_vec = list(), list(), list(), list()

    num_cluster_range = range(3, 4, 1)
    edge_num = 20
    num_nodes_vec = []
    print('================================= ///////////////////START//////////////// ========================================= ')
    for num_cluster in num_cluster_range:
        print('Simulating data...')
        model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(5000, num_cluster, 6, 10)
        print(edges)
        # graph = make_graph(variables, edges)
        # draw_graph(edges, variables)
        edge_num = len(edges) + 7
        # edge_num = float('inf')
        num_attributes = len(variables)
        train_data = data[: len(data) - 201]
        test_data = data[len(data) - 200 : len(data) - 1]
        print('==================================================== '+ str(num_attributes) +' NODES ==================================================== ')
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

                    list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
                    shuffle(list_order)

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


                    print(' ----------------------------------- NEW PRIORITY Gradient GRAFT (NAIVE) -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_new_naive_gradient_graft = time.time()
                    mn_new_gradient_graft, active_space_new_naive_gradient_Graft, naive_suff_stats_list, naive_recall, naive_precision, naive_time, naive_iterations = new_naive_priority_gradient_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose, edge_num, edges, list_order, zero_threshold, plot_queue)
                    t_new_naive_gradient_graft = time.time() - t_new_naive_gradient_graft
                    print('recall')
                    recall = float(len([x for x in active_space_new_naive_gradient_Graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_new_naive_gradient_Graft]))/len(active_space_new_naive_gradient_Graft)
                    print(precision)
                    likelihood_new_gradient_graft = compute_likelihood(mn_new_gradient_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_new_gradient_graft)
                    # new_naive_graft_recall_vec.append(recall)
                    # new_naive_graft_precision_vec.append(precision)
                    # new_naive_graft_likelihood_vec.append(likelihood_new_graft)
                    # t_new_naive_graft_vec.append(t_new_naive_graft)
                    print('time')
                    print(t_new_naive_gradient_graft)


                    # print(' ----------------------------------- NEW PRIORITY GRAFT (NAIVE) -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_new_naive_graft = time.time()
                    # mn_new_graft, active_space_new_naive_Graft, naive_suff_stats_list, naive_recall, naive_precision, naive_time = new_naive_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose, edge_num, edges, list_order)
                    # t_new_naive_graft = time.time() - t_new_naive_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_new_naive_Graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_new_naive_Graft]))/len(active_space_new_naive_Graft)
                    # print(precision)
                    # likelihood_new_graft = compute_likelihood(mn_new_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_new_graft)
                    # new_naive_graft_recall_vec.append(recall)
                    # new_naive_graft_precision_vec.append(precision)
                    # new_naive_graft_likelihood_vec.append(likelihood_new_graft)
                    # t_new_naive_graft_vec.append(t_new_naive_graft)
                    # print('time')
                    # print(t_new_naive_graft)

                    # print(' ----------------------------------- NEW STRUCTURED PRIORITY GRAFT -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_new_strcutured_priority_graft = time.time()
                    # mn_new_strcutured_priority_graft, active_space_new_strcutured_priority_graft, new_strcutured_suff_stats_list, new_strcutured_recall, new_strcutured_precision, new_strcutured_time = new_strcutured_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose, edge_num, edges, list_order)
                    # t_new_strcutured_priority_graft = time.time() - t_new_strcutured_priority_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_new_strcutured_priority_graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_new_strcutured_priority_graft]))/len(active_space_new_strcutured_priority_graft)
                    # print(precision)
                    # likelihood_new_strcutured_priority_graft = compute_likelihood(mn_new_strcutured_priority_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_new_strcutured_priority_graft)
                    # new_strcutured_priority_graft_recall_vec.append(recall)
                    # new_strcutured_priority_graft_precision_vec.append(precision)
                    # new_strcutured_priority_graft_likelihood_vec.append(likelihood_new_strcutured_priority_graft)
                    # t_new_strcutured_priority_graft_vec.append(t_new_strcutured_priority_graft)
                    # print('NEW STRUCTURED PRIORITY GRAFT time')
                    # print(t_new_strcutured_priority_graft)


                    print(' ----------------------------------- NEW STRUCTURED PRIORITY GRADIENT GRAFT -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_new_strcutured_priority_gradient_graft = time.time()
                    mn_new_strcutured_priority_gradient_graft, active_space_new_strcutured_priority_gradient_graft, new_strcutured_suff_stats_list, new_strcutured_recall, new_strcutured_precision, new_strcutured_time, new_strcutured_iterations = new_strcutured_priority_gradient_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose, edge_num, edges, list_order, zero_threshold, plot_queue)
                    t_new_strcutured_priority_gradient_graft = time.time() - t_new_strcutured_priority_gradient_graft
                    print('recall')
                    recall = float(len([x for x in active_space_new_strcutured_priority_gradient_graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_new_strcutured_priority_gradient_graft]))/len(active_space_new_strcutured_priority_gradient_graft)
                    print(precision)
                    likelihood_new_strcutured_priority_gradient_graft = compute_likelihood(mn_new_strcutured_priority_gradient_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_new_strcutured_priority_gradient_graft)
                    # new_strcutured_priority_gradient_graft_recall_vec.append(recall)
                    # new_strcutured_priority_gradient_graft_precision_vec.append(precision)
                    # new_strcutured_priority_gradient_graft_likelihood_vec.append(likelihood_new_strcutured_priority_graft)
                    # t_new_strcutured_gradient_priority_graft_vec.append(t_new_strcutured_gradient_priority_graft)
                    print('NEW STRUCTURED GRADIENT PRIORITY GRAFT time')
                    print(t_new_strcutured_priority_gradient_graft)




                    print(' ----------------------------------- QUEUE -----------------------------------')
                    print('l1 coeff: '+ str(l1_coeff))
                    t_queue_graft = time.time()
                    mn_queue_graft, active_space_queue_Graft, queue_suff_stats_list, queue_recall, queue_precision, queue_time, queue_iterations = queue_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose, edge_num, edges, list_order, zero_threshold, plot_queue)
                    t_queue_graft = time.time() - t_queue_graft
                    print('recall')
                    recall = float(len([x for x in active_space_queue_Graft if x in edges]))/len(edges)
                    print(recall)
                    print('precision')
                    precision = float(len([x for x in edges if x in active_space_queue_Graft]))/len(active_space_queue_Graft)
                    print(precision)
                    likelihood_queue_graft = compute_likelihood(mn_queue_graft, num_attributes, test_data)
                    print('Likelihood Graft')
                    print(likelihood_queue_graft)
                    # new_naive_graft_recall_vec.append(recall)
                    # new_naive_graft_precision_vec.append(precision)
                    # new_naive_graft_likelihood_vec.append(likelihood_new_graft)
                    # t_new_naive_graft_vec.append(t_new_naive_graft)
                    print('time')
                    print(t_queue_graft)






                    plt.plot(naive_iterations,'-b', label='Naive', linewidth=5)
                    plt.plot(new_strcutured_iterations, '-r', label='Structured', linewidth=5)
                    plt.plot(queue_iterations, '-g', label='Queue', linewidth=5)
                    plt.legend(loc='best')
                    plt.savefig('Sel_time/Selection_' + str(len(variables)) +'Nodes_MRF.png') 
                    plt.close()

                    plt.plot(naive_suff_stats_list,'-b', label='Naive', linewidth=5)
                    plt.plot(new_strcutured_suff_stats_list, '-r', label='Structured', linewidth=5)
                    plt.plot(queue_suff_stats_list, '-g', label='Queue', linewidth=5)
                    plt.legend(loc='best')
                    plt.savefig('Sel_time/Suff_Stats_' + str(len(variables)) +'Nodes_MRF.png') 
                    plt.close()




                    # edge_activated_struct = range(1, len(new_strcutured_suff_stats_list) + 1)
                    # edge_activated_naive = range(1, len(naive_suff_stats_list) + 1)
                    # plt.plot(naive_suff_stats_list, edge_activated_naive ,'-b', label='Naive', linewidth=5)
                    # plt.plot(new_strcutured_suff_stats_list, edge_activated_struct, '-r', label='Structured', linewidth=5)
                    # plt.xlim(0, max(naive_suff_stats_list[-1],new_strcutured_suff_stats_list[-1]))
                    # plt.legend(loc='best')
                    # plt.xlabel('Number of sufficent statistics')
                    # plt.ylabel('Number of activated edge')
                    # plt.title('Edge activation MRF_'+ str(len(variables)) + '_Nodes')
                    # plt.savefig('Recall_Precision_0/Edge_Activation_' + str(len(variables)) +'Nodes_MRF.png') 
                    # plt.close()

                    # plt.plot(naive_suff_stats_list, naive_recall,'-b', label='Naive', linewidth=5)
                    # plt.plot(new_strcutured_suff_stats_list, new_strcutured_recall,  '-r', label='Structured', linewidth=5)
                    # plt.xlim(0, 100)
                    # plt.legend(loc='best')
                    # plt.xlabel('Number of sufficent statistics')
                    # plt.ylabel('Recall ')
                    # plt.title('Recall MRF_'+ str(len(variables)) + '_Nodes')
                    # plt.savefig('Recall_Precision_0/Recall_' + str(len(variables)) +'Nodes_MRF.png') 
                    # plt.close()

                    # plt.plot(naive_suff_stats_list, naive_precision, '-b', label='Naive', linewidth=5)
                    # plt.plot(new_strcutured_suff_stats_list, new_strcutured_precision,  '-r', label='Structured', linewidth=5)
                    # plt.xlim(0, max(naive_suff_stats_list[-1],new_strcutured_suff_stats_list[-1]))
                    # plt.legend(loc='best')
                    # plt.xlabel('Number of sufficent statistics')
                    # plt.ylabel('Precision')
                    # plt.title('Precision MRF_'+ str(len(variables)) + '_Nodes')
                    # plt.savefig('Recall_Precision_0/Precision_' + str(len(variables)) +'Nodes_MRF.png') 
                    # plt.close()


                    # plt.plot(naive_time, naive_recall, '-b', label='Naive', linewidth=5)
                    # plt.plot(new_strcutured_time, new_strcutured_recall, '-r', label='Structured', linewidth=5)
                    # plt.xlim(0, max(new_strcutured_time[-1],naive_time[-1]))
                    # plt.legend(loc='best')
                    # plt.xlabel('Time (s)')
                    # plt.ylabel('Recall')
                    # plt.title('Recall MRF_'+ str(len(variables)) + '_Nodes')
                    # plt.savefig('Recall_Precision_0/time_recall_' + str(len(variables)) +'Nodes_MRF.png') 
                    # plt.close()




                    # print(' ----------------------------------- STRCUTURED PRIORITY GRAFT -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_strcutured_priority_graft = time.time()
                    # mn_strcutured_priority_graft, active_space_strcutured_priority_graft = strcutured_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose)
                    # t_strcutured_priority_graft = time.time() - t_strcutured_priority_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_strcutured_priority_graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_strcutured_priority_graft]))/len(active_space_strcutured_priority_graft)
                    # print(precision)
                    # likelihood_strcutured_priority_graft = compute_likelihood(mn_strcutured_priority_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_strcutured_priority_graft)
                    # strcutured_priority_graft_recall_vec.append(recall)
                    # strcutured_priority_graft_precision_vec.append(precision)
                    # strcutured_priority_graft_likelihood_vec.append(likelihood_strcutured_priority_graft)
                    # t_strcutured_priority_graft_vec.append(t_strcutured_priority_graft)


                    # print(' ----------------------------------- PRIORITY GRAFT (STRUCTURE) -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_priority_graft = time.time()
                    # mn_graft, active_space_mod_Graft = priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg,  2, 0, max_priority_grafting_iter, max_num_states, verbose)
                    # t_priority_graft = time.time() - t_priority_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_mod_Graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_mod_Graft]))/len(active_space_mod_Graft)
                    # print(precision)
                    # likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_graft)
                    # priority_graft_recall_vec.append(recall)
                    # priority_graft_precision_vec.append(precision)
                    # priority_graft_likelihood_vec.append(likelihood_graft)
                    # t_priority_graft_vec.append(t_priority_graft)


                    # print(' ----------------------------------- PRIORITY GRAFT (NAIVE) -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_naive_graft = time.time()
                    # mn_graft, active_space_naive_Graft = naive_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose)
                    # t_naive_graft = time.time() - t_naive_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_naive_Graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_naive_Graft]))/len(active_space_naive_Graft)
                    # print(precision)
                    # likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_graft)
                    # naive_graft_recall_vec.append(recall)
                    # naive_graft_precision_vec.append(precision)
                    # naive_graft_likelihood_vec.append(likelihood_graft)
                    # t_naive_graft_vec.append(t_naive_graft)
                    

                    # print(' ----------------------------------- GENERAL PRIORITY GRAFT -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_mod_graft = time.time()
                    # mn_graft, active_space_mod_Graft = mod_priority_graft( variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg,  2, 0, max_priority_grafting_iter, max_num_states, verbose)
                    # t_mod_graft = time.time() - t_mod_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_mod_Graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_mod_Graft]))/len(active_space_mod_Graft)
                    # print(precision)
                    # likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_graft)
                    # mod_graft_recall_vec.append(recall)
                    # mod_graft_precision_vec.append(precision)
                    # mod_graft_likelihood_vec.append(likelihood_graft)
                    # t_mod_graft_vec.append(t_mod_graft)


                    # print(' ----------------------------------- GRAFT -----------------------------------')
                    # print('l1 coeff: '+ str(l1_coeff))
                    # t_graft = time.time()
                    # mn_graft, active_space_Graft = graft(variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_grafting_iter, max_num_states, verbose)
                    # t_graft = time.time() - t_graft
                    # print('recall')
                    # recall = float(len([x for x in active_space_Graft if x in edges]))/len(edges)
                    # print(recall)
                    # print('precision')
                    # precision = float(len([x for x in edges if x in active_space_Graft]))/len(active_space_Graft)
                    # print(precision)
                    # likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
                    # print('Likelihood Graft')
                    # print(likelihood_graft)
                    # graft_recall_vec.append(recall)
                    # graft_precision_vec.append(precision)
                    # graft_likelihood_vec.append(likelihood_graft)
                    # t_graft_vec.append(t_graft)

                    print('~~~~~~~~~ EXEC TIME ~~~~~~~~~')
                    # print('STRUCTURED PRIORITY GRAFT time')
                    # print(t_strcutured_priority_graft)

                    # print('PRIORITY GRAFT (STRUCTURE) time')
                    # print(t_priority_graft)

                    # print('PRIORITY GRAFT (NAIVE) time')
                    # print(t_naive_graft)

                    # print('NEW PRIORITY GRAFT (NAIVE) time')
                    # print(t_new_naive_graft)

                    # print('NEW STRUCTURED PRIORITY GRAFT time')
                    # print(t_new_strcutured_priority_graft)

                    print('NEW STRUCTURED PRIORITY GRADIENT GRAFT time')
                    print(t_new_strcutured_priority_gradient_graft)


                    print('NEW PRIORITY Gradient GRAFT (NAIVE) time')
                    print(t_new_naive_gradient_graft)


                    print('QUEUE GRAFT time')
                    print(t_queue_graft)
                    

                    # print('GENERAL PRIORITY GRAFT time')
                    # print(t_mod_graft)

                    # print('GRAFT time')
                    # print(t_graft)


    # print('+++++++++++ STRCUTURED PRIORITY GRAFT +++++++++++')
    # print('Exec time')
    # print(t_strcutured_priority_graft_vec)
    # print('Number of nodes')
    # print(num_nodes_vec)
    # print('Likelihoods')
    # print(strcutured_priority_graft_likelihood_vec)
    # print('recall')
    # print(strcutured_priority_graft_recall_vec)
    # print('precision')
    # print(strcutured_priority_graft_precision_vec)


    # # print('+++++++++++ GRAFT +++++++++++')
    # # print('Exec time')
    # # print(t_graft_vec)
    # # print('Number of nodes')
    # # print(num_nodes_vec)
    # # print('Likelihoods')
    # # print(graft_likelihood_vec)
    # # print('recall')
    # # print(graft_recall_vec)
    # # print('precision')
    # # print(graft_precision_vec)

    # print('+++++++++++ PRIORITY GRAFT (NAIVE)+++++++++++')
    # print('Exec time')
    # print(t_naive_graft_vec)
    # print('Number of nodes')
    # print(num_nodes_vec)
    # print('Likelihoods')
    # print(naive_graft_likelihood_vec)
    # print('recall')
    # print(naive_graft_recall_vec)
    # print('precision')
    # print(naive_graft_precision_vec)

    # # print('+++++++++++ PRIORITY GRAFT (STRUCTURE) +++++++++++')
    # # print('Exec time')
    # # print(t_priority_graft_vec)
    # # print('Likelihoods')
    # # print('Number of nodes')
    # # print(num_nodes_vec)
    # # print(priority_graft_likelihood_vec)
    # # print('recall')
    # # print(priority_graft_recall_vec)
    # # print('precision')
    # # print(priority_graft_precision_vec)

    # # print('+++++++++++ GENERAL PRIORITY GRAFT +++++++++++')
    # # print('Exec time')
    # # print(t_mod_graft_vec)
    # # print('Likelihoods')
    # # print('Number of nodes')
    # # print(num_nodes_vec)
    # # print(mod_graft_likelihood_vec)
    # # print('recall')
    # # print(mod_graft_recall_vec)
    # # print('precision')
    # # print(mod_graft_precision_vec)




if  __name__ =='__main__':
    main()