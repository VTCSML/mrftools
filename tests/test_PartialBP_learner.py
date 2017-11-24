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

def plot_objective_PartialBP(N, images, models, labels, names, size):
    #plt.clf()
    learner = PartialLearner(N, PartialMatrixBP)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)

    for model, label in zip(models, labels):
        learner.add_data(label, model)

    plotter = ObjectivePlotter(func=learner.objective)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "PBP_N%d_S%d.jpg"%(N,size)
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)

    return weights

def plot_objective_MatrixBP(images, models, labels, names, size):
    plt.clf()
    learner = Learner(MatrixBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    plotter = ObjectivePlotter(func=learner.objective)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "MBP_S%d.jpg"%size
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)
    return weights


def train_CRF_PartialBP(N, images, models, labels, names):
    learner = PartialLearner(N, PartialMatrixBP)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)

    for model, label in zip(models, labels):
        learner.add_data(label, model)
    weights = learner.learn(initial_weights)
    return weights

def train_CRF_MatrixBP(images, models, labels, names):
    learner = Learner(MatrixBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    weights = learner.learn(initial_weights)
    return weights

def test_difference(N, images, models, labels, names):
    start = time.time()
    MBP_w = train_CRF_MatrixBP(images, models, labels, names)
    end = time.time()
    print "Matrix BP running time:%f"%(end-start)
    start = time.time()
    PBP_w = train_CRF_PartialBP(N, images, models, labels, names)
    end = time.time()
    print "Partial BP running time:%f"%(end-start)
    var = np.var(abs(MBP_w - PBP_w))
    mean = np.mean(abs(MBP_w - PBP_w))
    print "variance: %f" %var
    print "mean: %f" %mean
    print MBP_w
    print PBP_w

def output_PartialBP_beliefs(images, models, labels, names):
    unary_beliefs = list()
    learner = PartialLearner(N, PartialMatrixBP)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)

    for model, label in zip(models, labels):
        learner.add_data(label, model)

    weights = learner.learn(initial_weights)
    for bp in learner.belief_propagators:
        bp.load_beliefs()
        unary_beliefs.append(bp.var_beliefs)
    return unary_beliefs

def output_MatrixBP_beliefs(images, models, labels, names):
    unary_beliefs = list()
    learner = Learner(MatrixBeliefPropagator)
    num_states = 2
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    weights = learner.learn(initial_weights)

    for bp in learner.belief_propagators:
        bp.load_beliefs()
        unary_beliefs.append(bp.var_beliefs)
    return unary_beliefs

def compare_beliefs(pbp_beliefs, mbp_beliefs):
    means = list()
    for (pbp_belief, mbp_belief) in zip(pbp_beliefs, mbp_beliefs):
        means.append(np.mean(abs(pbp_belief - mbp_belief).flat))
    return means



if __name__ == '__main__':
    size = 100
    N = 50
    images, models, labels, names = batch_load_images(size)
    start = time.time()
    #ww = train_CRF_PartialBP(N, images, models, labels, names)
    #print ww

    #test_difference(N, images, models, labels, names)

    ww = plot_objective_PartialBP(N, images, models, labels, names, size)
    #ww = plot_objective_MatrixBP(images, models, labels, names, size)

    #pbp_beliefs = output_PartialBP_beliefs(images, models, labels, names)
    #mbp_beliefs = output_MatrixBP_beliefs(images, models, labels, names)
    #means = compare_beliefs(pbp_beliefs, mbp_beliefs)
    #print means



    end = time.time()
    print "running time:%f"%(end-start)
    print "end"
