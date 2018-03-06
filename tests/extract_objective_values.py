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


def select_objectives(interval, iterations, objective_values, running_time):
    new_obj_list = list()
    new_time_list = list()
    for i in range(0, len(objective_values)):
        if iterations[i]% interval == 0:
            new_obj_list.append(objective_values[i])
            new_time_list.append(running_time[i])
    if iterations[-1] % interval != 0:
        new_obj_list.append(objective_values[-1])
        new_time_list.append(running_time[-1])

    return new_obj_list, new_time_list


def output_processed_results(running_time, obj_list, dir, name, algorithm):
    f = open(osp.join(dir, "%s.csv"%name), "w")
    for i in range(0, len(running_time)):
        time = running_time[i]
        obj = obj_list[i]
        f.write(str(time) + "," + str(obj) + "," + algorithm)
        f.write("\n")
    f.close()


if __name__ == '__main__':

    name = "extracted_objective"
    algorithm = "convexBP"
    interval = 50


    horse_dir = "/Users/youlu/Documents/workspace/mrftools/tests/test_results/horse_FCN/convexBP_new"
    horse_obj_path = osp.join(horse_dir, "objective.txt")
    horse_time_path = osp.join(horse_dir, "time.txt")

    scene_dir = "/Users/youlu/Documents/workspace/mrftools/tests/test_results/scene_FCN/convexBP"
    scene_obj_path = osp.join(scene_dir, "objective.txt")
    scene_time_path = osp.join(scene_dir, "time.txt")


    horse_obj = load_objective_values(horse_obj_path)
    iterations, running_time = load_time(horse_time_path)
    new_obj_list, new_time_list = select_objectives(interval, iterations, horse_obj, running_time)
    output_processed_results(new_time_list, new_obj_list, horse_dir, name, algorithm)


    scene_obj = load_objective_values(scene_obj_path)
    iterations, running_time = load_time(scene_time_path)
    new_obj_list, new_time_list = select_objectives(interval, iterations, scene_obj, running_time)
    output_processed_results(new_time_list, new_obj_list, scene_dir, name, algorithm)


