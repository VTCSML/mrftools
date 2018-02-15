import unittest
from mrftools import *
import numpy as np

def batch_load_images(size):
    dir = "/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse"
    IFL = ImageFeatureLoader(max_width=size, max_height=size)
    models, labels, names = IFL.load_all_features_labels(dir, "dataset", 2)
    return models, labels, names

def plot_objective_ConvexMatrixBP(models, labels, size):
    plt.clf()
    learner = Learner(ConvexBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    # plotter = ObjectivePlotter(func=learner.objective)
    # #weights = learner.learn(initial_weights, callback=None)
    # weights = learner.learn(initial_weights, callback=plotter.callback)
    # filename = "ConvexMBP_S%d.jpg"%size
    # plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)
    #return weights




if __name__ == '__main__':
    size = 100
    models, labels, names = batch_load_images(size)
    ww = plot_objective_ConvexMatrixBP(models, labels, size)
    print "end"
