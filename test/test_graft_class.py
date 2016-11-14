import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from Graft import Graft



if __name__ == '__main__':
	edge_reg = .075
	print('Simulating data...')
	model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(5000, 3, 6, 10)
	print('EDGES')
	print(edges)
	edge_num = len(edges) + 7
	num_attributes = len(variables)
	list_order = range(0,(len(variables) ** 2 - len(variables)) / 2, 1)
	shuffle(list_order)
	grafter = Graft(variables, num_states, max_num_states, data, list_order)
	grafter.on_show_metrics()
	# spg.on_verbose()
	grafter.setup_learning_parameters(edge_reg)
	learned_mn, final_active_set, suff_stats_list, recall, precision = grafter.learn_structure(edge_num, edges)
