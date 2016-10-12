import numpy as np
import os
import csv
import time
import random
from collections import Counter
import operator

from run_real_experiment import run_real_experiment

def load_openimages():
    data_dir = '/Users/bert/Dropbox/Research/openimages/'

    concepts = dict()

    all_concepts = set()

    default_dict = dict()

    with open(data_dir + 'dict.csv', 'r') as dict_file:
        dict_csv = csv.reader(dict_file)

        for row in dict_csv:
            concepts[row[0]] = row[1]
            all_concepts.add(row[1])

            default_dict[row[1]] = 0

    data = dict()

    random.seed(0)

    label_frequency = Counter()

    subsample = 1.0 # proportion of qualifying images to use
    num_variables = 1000 # number of most frequent variables to consider

    num_lines = float('inf') # number of lines to read from file

    with open(data_dir + 'labels.csv', 'r') as label_file:
        label_csv = csv.DictReader(label_file)

        for i, row in enumerate(label_csv):
            if row["ImageID"] not in data:
                data[row["ImageID"]] = set()
            data[row["ImageID"]].add(concepts[row["LabelName"]])

            label_frequency[concepts[row["LabelName"]]] += 1

            if i % 100000 == 0:
                print "Finished loading line %d" % i

            if i > num_lines:
                break

    sorted_labels = sorted(label_frequency, key=label_frequency.get, reverse=True)

    # get top num_variables labels

    considered_labels = set(sorted_labels[:num_variables])

    default_labels = dict()
    for label in considered_labels:
        default_labels[label] = 0

    # generate data dictionary

    examples = []

    for image_labels in data.values():
        if random.random() < subsample:
            num_labels = 0

            example = default_labels.copy()

            for label in image_labels:
                if label in considered_labels:
                    num_labels += 1
                    example[label] = 1

            if num_labels >= 10:
                examples.append(example)


    print "Loaded %d examples. First example:" % len(examples)
    print examples[0]
    for label in considered_labels:
        if examples[0][label] == 1:
            print label


    variables = considered_labels

    num_states = dict()
    for var in variables:
        num_states[var] = 2

    training_data = examples


    return variables, num_states, training_data

def main():
    variables, num_states, training_data = load_openimages()

    print "Data loaded"

    num_edges = 300

    mn = run_real_experiment(num_edges, num_states, training_data)

    mn.load_factors_from_matrices()

    print "Learned edges:"

    for var in mn.variables:
        for neighbor in mn.neighbors[var]:
            if var < neighbor:
                if np.linalg.norm(mn.get_potential((var, neighbor))) > 1e-3:
                    print "(%s, %s)" % (var, neighbor)

if __name__ == "__main__":
    main()
