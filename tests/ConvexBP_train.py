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
import evaluate_model
import argparse
Inference_names = {"ConvexBeliefPropagator":ConvexBeliefPropagator,\
                   "MatrixBeliefPropagator":MatrixBeliefPropagator, \
                   "PartialConvexBeliefPropagator":PartialConvexBeliefPropagator,\
                   "PartialMatrixBP":PartialMatrixBP}

if __name__ == '__main__':
    print "Convex BP"

    parser = argparse.ArgumentParser(description='BP accuracy')
    parser.add_argument("-i", "--input_dir", default="/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse", type=str, help="path to the input data")
    parser.add_argument("-o", "--out_dir", default="/Users/youlu/Documents/workspace/mrftools/tests/test_results/horse_FCN/", type=str, help="path to the results")
    parser.add_argument("-p", "--plot_path", default="/Users/youlu/Documents/workspace/mrftools/tests/test_results/ConvexBP_100.jpg", type=str, help="path to save the plot")
    parser.add_argument("-s", "--size", default=100, type=int, help="iamge size")
    parser.add_argument("-c", "--num_class", default=2, type=int, help="number of class")
    parser.add_argument("-n", "--output_name", default = "ConvexBP_weights", type = str, help="the weights name")
    parser.add_argument("-t", "--inference_type", default="ConvexBeliefPropagator", type = str, help="the inference type")
    args = parser.parse_args()
    inference = Inference_names[args.inference_type]
    evaluate_model.FCN_features_train(args.input_dir, "train", args.out_dir, args.size, args.num_class, args.output_name, inference, args.plot_path)

