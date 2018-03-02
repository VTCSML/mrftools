import unittest
from mrftools import *
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import itertools
from mrftools import save_load_weights
import skimage.color
import skimage.util
from skimage.io import imsave
from os.path import splitext
import argparse

def load_weights(path):
    weights = np.loadtxt(file(path))
    return weights


def load_objective_values(path):
    objective_values = np.loadtxt(file(path))

    return objective_values


def load_time(path):
    lines = file(path).readlines()
    iterations = list()
    running_time = list()
    for line in lines:
        iteration, time = line.strip().split("\t")
        iterations.append(int(iteration))
        running_time.append(float(time))
    return iterations, running_time


def load_all_weights(iterations, dir):
    all_weights = list()
    for it in iterations:
        weights = load_weights(osp.join(dir, "weights_%d.txt"%it))
        all_weights.append(weights)
    return all_weights


def compute_norm(all_weights, weight_star):
    norm_list = np.zeros(len(all_weights))
    for i in range(0, len(all_weights)):
        l2_norm = np.linalg.norm(all_weights[i]-weight_star)
        norm_list[i] = l2_norm
    return norm_list


def output_processed_results(running_time, obj_list, dir, name, algorithm):
    f = open(osp.join(dir, "%s.csv"%name), "w")
    for i in range(0, len(running_time)):
        time = running_time[i]
        obj = obj_list[i]
        f.write(str(time) + "," + str(obj) + "," + algorithm)
        f.write("\n")
    f.close()


def batch_load_images_features(dir, size, dataset, num_class):
    IFL = ImageFeatureLoader(max_width=size, max_height=size)
    models, labels, names = IFL.load_all_features_labels(dir, dataset, num_class)
    return models, labels, names


def get_true_objective_values(weights_list, models, labels, interval, running_time, iterations):
    obj_value_list = list()
    new_weights_list = list()
    new_running_time_list = list()
    for i in range(0, len(weights_list)):
        if iterations[i]% interval == 0:
            new_weights_list.append(weights_list[i])
            new_running_time_list.append(running_time[i])
    if iterations[-1] % interval != 0:
        new_weights_list.append(weights_list[-1])
        new_running_time_list.append(running_time[-1])

    learner = Learner(ConvexBeliefPropagator)
    for model, label in zip(models, labels):
        learner.add_data(label, model)
    print "end loading data"
    for i in range(0, len(new_weights_list)):
        print i
        weights = new_weights_list[i]
        obj_value = learner.subgrad_obj(weights)
        obj_value_list.append(obj_value)
    return new_running_time_list, obj_value_list


if __name__ == '__main__':
    print "process results"

    parser = argparse.ArgumentParser(description='BP accuracy')
    parser.add_argument("-p", "--input_dir", default="/Users/youlu/Documents/workspace/mrftools/tests/test_results/horse_FCN/cyclic_partialBP_10_new", type=str, help="path to the weights")
    parser.add_argument("-d", "--data_dir", default="/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse", type=str, help="path to the data")
    parser.add_argument("-s", "--size", default=100, type=int, help="iamge size")
    parser.add_argument("-a", "--algorithm", default="cyclic_partialBP_10", type=str, help="algorithm name")
    parser.add_argument("-w", "--weight_star", default="/Users/youlu/Documents/workspace/mrftools/tests/test_results/horse_FCN/convexBP_new/ConvexBP_weights.txt", type=str, help="path to weight star")
    parser.add_argument("-c", "--num_class", default=2, type=int, help="number of class")
    args = parser.parse_args()

    weights_star = load_weights(args.weight_star)
    iterations, running_time = load_time(osp.join(args.input_dir, "time.txt"))
    all_weights = load_all_weights(iterations, args.input_dir)

    # norm_list = compute_norm(all_weights, weights_star)
    # output_processed_results(running_time, norm_list, args.input_dir, "weights_norm", args.algorithm)

    models, labels, names = batch_load_images_features(args.data_dir, args.size, "train", args.num_class)
    new_running_time_list, obj_value_list = get_true_objective_values(all_weights, models, labels, 50, running_time, iterations)
    output_processed_results(new_running_time_list, obj_value_list, args.input_dir, "true_objective_values", args.algorithm)











