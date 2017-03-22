import time
from DataCleaning import load_instances, load_attribute_values
import matplotlib.pyplot as plt
from priority_graft import priority_graft
from grafting_util import compute_likelihood
from queue_grafting import queue_graft
from naive_priority_graft import naive_priority_graft
from mod_priority_graft import mod_priority_graft
from graft import graft
import numpy as np
import matplotlib.pyplot as plt
from generate_synthetic_data import generate_synthetic_data

L1COEFF = .7
MAX_ITER_GRAFTING = 500
verbose = True


def main():
    print('Simulating data...')
    variables, data, max_num_states, num_states = generate_synthetic_data()
    num_attributes = len(variables)
    train_data = data[0 : len(data) - 4001]
    test_data = data[len(data) - 4000 : len(data) - 1]
    print('nodes')
    print(variables)
    print('Number of Data Points')
    print(len(data))
    print('Number of Attributes')
    print(num_attributes)
    print('Number of States per Attribute')
    print(num_states)

    likelihoods, exec_time = dict(), dict()
    pr_gt, q_gt, npr_gt, md_pr_gt, gt = list(), list(), list(), list(), list()
    t_pr_gt, t_q_gt, t_npr_gt, t_md_pr_gt, t_gt = list(), list(), list(), list(), list()

    for L1COEFF in list(np.arange(0.5, 1, 0.1)):

        print('L1COEFF: ' + str(L1COEFF))
        print('Priority Graft Start...')
        t = time.time()
        mn_priority_graft, weights_priority_graft, Weights_dict_priority_graft, active_space_priority_graft= priority_graft(variables, num_states, train_data, L1COEFF, 3, .1, False, MAX_ITER_GRAFTING, max_num_states, verbose)
        elapsed_priority_graft = time.time() - t
        print('Priority Graft ENDED After')
        print(elapsed_priority_graft)
        likelihood_priority_graft = compute_likelihood(mn_priority_graft, num_attributes, test_data)
        print('Likelihood Priority Graft')
        print(likelihood_priority_graft)
        pr_gt.append(likelihood_priority_graft)
        t_pr_gt.append(elapsed_priority_graft)

        print('Queue Grafting Start...')
        t = time.time()
        mn_queuey_graft, weights_queue_graft, weights_dict_queue_graft, active_space_queue_graft = queue_graft(variables, num_states, train_data, L1COEFF, MAX_ITER_GRAFTING, max_num_states, verbose)
        elapsed_queue_graft = time.time() - t
        print('Queue Grafting ENDED AFTER')
        print(elapsed_queue_graft)
        likelihood_queue_graft = compute_likelihood(mn_queuey_graft, num_attributes, test_data)
        print('Likelihood Queue Graft')
        print(likelihood_queue_graft)
        q_gt.append(likelihood_queue_graft)
        t_q_gt.append(elapsed_queue_graft)

        print('Naive Priority Grafting Start...')
        t = time.time()
        mn_naive_priority_graft, weights_naive_priority_graft, weights_dict_naive_priority_praft, active_space_naive_priority_graft = naive_priority_graft(variables, num_states, train_data, L1COEFF, MAX_ITER_GRAFTING, max_num_states, verbose)
        elapsed_naive_priority_graft = time.time() - t
        print('Naive Priority Grafting END AFTER')
        print(elapsed_naive_priority_graft)
        likelihood_naive_priority_graft = compute_likelihood(mn_naive_priority_graft, num_attributes, test_data)
        print('Likelihood Naive Priority Graft')
        print(likelihood_naive_priority_graft)
        npr_gt.append(likelihood_naive_priority_graft)
        t_npr_gt.append(elapsed_naive_priority_graft)

        print('Modified Priority Graft Start...')
        t = time.time()
        mn_mod_priority_graft, weights_mod_priority_graft, weights_dict_mod_priority_graft, active_space_mod_priority_graft = mod_priority_graft(variables, num_states, train_data, L1COEFF, 3, .1, False, MAX_ITER_GRAFTING, max_num_states, verbose)
        elapsed_mod_priority_graft = time.time() - t
        print('Modified Priority Graft END')
        print(elapsed_mod_priority_graft)
        likelihood_mod_priority_graft = compute_likelihood(mn_mod_priority_graft, num_attributes, test_data)
        print('Likelihood Modified Priority Graft')
        print(likelihood_mod_priority_graft)
        md_pr_gt.append(likelihood_mod_priority_graft)
        t_md_pr_gt.append(elapsed_mod_priority_graft)

        print('Grafting Start...')
        t = time.time()
        mn_graft, weights_graft, weights_dict_graft, active_space_Graft = graft(variables, num_states, train_data, L1COEFF, MAX_ITER_GRAFTING, max_num_states, verbose)
        elapsed_graft = time.time() - t
        print('Grafting ENDED AFTER')
        print(elapsed_graft)
        likelihood_graft = compute_likelihood(mn_graft, num_attributes, test_data)
        print('Likelihood Graft')
        print(likelihood_graft)
        gt.append(likelihood_graft)
        t_gt.append(elapsed_graft)

    plt.plot(pr_gt, 'r')
    plt.plot(q_gt, 'g')
    plt.plot(npr_gt, 'b')
    plt.plot(md_pr_gt, 'y')
    plt.plot(gt)
    plt.show()
    plt.close()

    plt.plot(t_pr_gt, 'r')
    plt.plot(t_q_gt, 'g')
    plt.plot(t_npr_gt, 'b')
    plt.plot(t_md_pr_gt, 'y')
    plt.plot(t_gt)
    plt.show()
    plt.close()

if  __name__ =='__main__':
    main()