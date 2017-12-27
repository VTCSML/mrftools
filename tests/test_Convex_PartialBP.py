import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
import itertools


def batch_load_images(size):
    dir = "/Users/youlu/Documents/workspace/mrftools/tests/train_data/"
    IL = ImageLoader(max_width=size, max_height=size)
    images, models, labels, names = IL.load_all_images_and_labels(dir, 2, num_images=np.inf)
    return images, models, labels, names

def plot_objective_ConvexPartialBP(N, images, models, labels, names, size):
    #plt.clf()
    learner = PartialLearner(N, PartialConvexBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)

    for model, label in zip(models, labels):
        learner.add_data(label, model)

    plotter = ObjectivePlotter(func=learner.objective)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "ConvexPBP_N%d_S%d.jpg"%(N,size)
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)

    return weights

def plot_objective_ConvexMatrixBP(images, models, labels, names, size):
    plt.clf()
    learner = Learner(ConvexBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    plotter = ObjectivePlotter(func=learner.objective)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "ConvexMBP_S%d.jpg"%size
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)
    return weights

def plot_dualobjective_ConvexPartialBP(N, images, models, labels, names, size):
    #plt.clf()
    learner = PartialLearner(N, PartialConvexBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)

    for model, label in zip(models, labels):
        learner.add_data(label, model)

    plotter = ObjectivePlotter(func=learner.dual_obj)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "dual_ConvexPBP_N%d_S%d.jpg"%(N,size)
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)

    return weights

def plot_dualobjective_ConvexMatrixBP(images, models, labels, names, size):
    plt.clf()
    learner = Learner(ConvexBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    plotter = ObjectivePlotter(func=learner.dual_obj)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "dual_ConvexMBP_S%d.jpg"%size
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)
    return weights




if __name__ == '__main__':
    size = 10
    N = 10
    images, models, labels, names = batch_load_images(size)
    start = time.time()


    ww = plot_objective_ConvexPartialBP(N, images, models, labels, names, size)
    #ww = plot_objective_ConvexMatrixBP(images, models, labels, names, size)

    #ww = plot_dualobjective_ConvexMatrixBP(images, models, labels, names, size)
    #ww = plot_dualobjective_ConvexPartialBP(N, images, models, labels, names, size)





    end = time.time()
    print "running time:%f"%(end-start)
    print "end"
