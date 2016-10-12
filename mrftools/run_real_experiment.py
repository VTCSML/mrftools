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
from strcutured_priority_graft import strcutured_priority_graft
from new_naive_priority_graft import new_naive_priority_graft
from new_strcutured_priority_graft import new_strcutured_priority_graft
from random import shuffle


def run_real_experiment(edge_num, num_states, train_data, test_data):
    """
    Inputs:
    - edge_num: desired number of edges
    - num_states: a dict indicating the number of states per variable
    - train_data: list of data instances for training
    - test_data: list of data instances for learning
    """
    variables = list(num_states.keys())
    verbose = False
    l1_coeff = 0
    l2_coeff = 0
    edge_reg = 0
    var_reg = 0
    var_regs = 0
    edge_regs = 0
    l1_coeffs = .01
    max_grafting_iter = 2500
    max_priority_grafting_iter = 250
    num_attributes = len(variables)
    max_num_states = max(list(num_states.values))

    print('================================= ///////////////////START//////////////// ========================================= ')

    print(' ----------------------------------- NEW STRUCTURED PRIORITY GRAFT -----------------------------------')
    t_new_strcutured_priority_graft = time.time()
    mn_new_strcutured_priority_graft, active_space_new_strcutured_priority_graft = stuctured_priority_graft_real_data(variables, num_states, train_data, l1_coeff, l2_coeff, var_reg, edge_reg, max_priority_grafting_iter, max_num_states, verbose, edge_num)
    print('Done learning')
    t_new_strcutured_priority_graft = time.time() - t_new_strcutured_priority_graft
    print('Testing...')
    likelihood_new_strcutured_priority_graft = compute_likelihood(mn_new_strcutured_priority_graft, num_attributes, test_data)
    print('Likelihood METHOD')
    print(likelihood_new_strcutured_priority_graft)
    print('time')
    print(t_new_naive_graft)
