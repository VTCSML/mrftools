import time
import matplotlib.pyplot as plt
import numpy as np
from generate_synthetic_data import generate_synthetic_data
from random import shuffle
from scipy import signal as sg
from StructuredPriorityGraft import StructuredPriorityGraft



if __name__ == '__main__':
	edge_reg = .075
	method = 'naive'
	print('Simulating data...')
	model, variables, data, max_num_states, num_states, edges = generate_synthetic_data(5000, 3, 6, 10)
	print('EDGES')
	print(edges)
	edge_num = len(edges) + 7
	num_attributes = len(variables)
	spg = StructuredPriorityGraft(variables, num_states, max_num_states, data, method)

	spg.on_show_metrics()
	# spg.on_verbose()
	spg.on_plot_queue()
	spg.setup_learning_parameters(edge_reg)
	learned_mn, final_active_set, suff_stats_list, recall, precision, iterations = spg.learn_structure(edge_num, edges)
