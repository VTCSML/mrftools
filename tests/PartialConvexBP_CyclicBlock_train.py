import unittest
import sys
sys.path.append('../')
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import itertools
from mrftools import save_load_weights
import skimage.color
import skimage.util
from skimage.io import imsave
from os.path import splitext
#import evaluate_model
import argparse
Inference_names = {"ConvexBeliefPropagator":ConvexBeliefPropagator,\
                   "MatrixBeliefPropagator":MatrixBeliefPropagator, \
                   "PartialConvexBeliefPropagator":PartialConvexBeliefPropagator,\
                   "PartialMatrixBP":PartialMatrixBP, \
                   "PartialConvexBP_CyclicBlock":PartialConvexBP_CyclicBolck}


def batch_load_images_features(dir, size, dataset, num_class):
    #dir = "/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse"
    IFL = ImageFeatureLoader(max_width=size, max_height=size)
    models, labels, names = IFL.load_all_features_labels(dir, dataset, num_class)
    return models, labels, names

def train_model(num_R, num_C, models, labels, num_class, inference_type, plot_path, output_dir):
    plt.clf()
    learner = PartialLearner_CyclicBlock(num_R, num_C, inference_type)
    num_states = num_class
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)

    for model, label in zip(models, labels):
        learner.add_data(label, model)

    plotter = ObjectivePlotter(func=learner.objective)
    weights = learner.learn(initial_weights, output_dir, callback=plotter.callback)
    plt.savefig(plot_path)

    return weights

def FCN_features_train(dir, dataset, output_dir, size, num_R, num_C, num_class, output_name, inference_type, plot_path):
    models, labels, names = batch_load_images_features(dir, size, dataset, num_class)
    weights = train_model(num_R, num_C, models, labels, num_class, inference_type, plot_path, output_dir)
    output_path = osp.join(output_dir, "%s.txt"%output_name)
    save_load_weights.save_weights(weights, output_path)
    return weights


if __name__ == '__main__':
    print "Cyclic Partial BP"

    parser = argparse.ArgumentParser(description='BP accuracy')
    parser.add_argument("-i", "--input_dir", default="/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse", type=str, help="path to the input data")
    parser.add_argument("-o", "--out_dir", default="/Users/youlu/Documents/workspace/mrftools/tests/test_results/horse_FCN/", type=str, help="path to the results")
    parser.add_argument("-p", "--plot_path", default="/Users/youlu/Documents/workspace/mrftools/tests/test_results/ConvexPartialBP_100.jpg", type=str, help="path to save the plot")
    parser.add_argument("-x", "--row_size", default=10, type=int, help="number of rows separated")
    parser.add_argument("-y", "--column_size", default=10, type=int, help="number of columns separated")
    parser.add_argument("-s", "--size", default=100, type=int, help="iamge size")
    parser.add_argument("-c", "--num_class", default=2, type=int, help="number of class")
    parser.add_argument("-n", "--output_name", default = "ConvexPartialBP_weights", type = str, help="the weights name")
    parser.add_argument("-t", "--inference_type", default="PartialConvexBP_CyclicBlock", type = str, help="the inference type")
    args = parser.parse_args()
    inference = Inference_names[args.inference_type]
    FCN_features_train(args.input_dir, "subset", args.out_dir, args.size, args.row_size, args.column_size, args.num_class, args.output_name, inference, args.plot_path)
