import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

def batch_load_images():
    dir = "/Users/youlu/Documents/workspace/mrftools/tests/train_data/"
    IL = ImageLoader(max_width=130, max_height=107)
    images, models, labels, names = IL.load_all_images_and_labels(dir, 2, num_images=np.inf)
    return images, models, labels, names


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
    diff = np.sum(abs(MBP_w - PBP_w)) / len(MBP_w)
    mean = np.mean(abs(MBP_w - PBP_w))
    print "diff: %f     mean: %f"%(diff, mean)


if __name__ == '__main__':
    N = 10
    images, models, labels, names = batch_load_images()
    start = time.time()
    #ww = train_CRF_PartialBP(N, images, models, labels, names)
    #print ww
    test_difference(N, images, models, labels, names)
    end = time.time()
    print "running time:%f"%(end-start)
    print "end"
