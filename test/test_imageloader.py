import unittest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os
from mrftools import *
from scipy.optimize import minimize, check_grad


class TestImageLoader(unittest.TestCase):

    def test_load_draw(self):

        loader = ImageLoader()

        train_dir = os.path.join(os.path.dirname(__file__), 'train')

        images, models, labels, names = loader.load_all_images_and_labels(train_dir, 2)
        files = [x for x in os.listdir(train_dir) if x.endswith(".jpg") or x.endswith('.png')]
        for i, filename in enumerate(files):
            full_name = os.path.join(train_dir, filename)
            img = Image.open(full_name)
            features = models[i].unary_features
            edge_features = models[i].edge_features
            edges = ImageLoader.get_all_edges(img)
            assert np.allclose(len(labels[i]), img.width * img.height), "the size of labels is right"
            assert np.allclose(len(features), img.width * img.height), "the size of features is right"
            assert np.allclose(len(edge_features) / 2, len(edges)), "the size of edge features is right"

    def test_unary_only(self):
        num_features = 65
        num_states = 2

        all_pixel, all_label = load_all_images_and_labels(os.path.join(os.path.dirname(__file__), 'train'), num_features, 1)

        initial_w = np.zeros(num_features * num_states)
        res = minimize(objective, initial_w, method="L-BFGS-B", args=(all_pixel, all_label, num_features, num_states),
                       jac=gradient)
        weights = res.x

        accuracy_training = accuracy(weights, all_pixel, all_label, num_features, num_states)
        print ("accuracy on training set: %f" % (accuracy_training))
        assert (accuracy_training >= 0.9), "Unary classification accuracy on training data is less than 0.9"

        all_pixel, all_label = load_all_images_and_labels(os.path.join(os.path.dirname(__file__), 'test'), num_features, 1)
        accuracy_testing = accuracy(weights, all_pixel, all_label, num_features, num_states)
        print ("accuracy on testing set: %f" % (accuracy_testing))
        assert (accuracy_testing >= 0.7), "Unary classification accuracy on testing data is less than 0.7"



def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted), 1, keepdims=True)

def objective(weights, features, label_vec, num_features, num_states):
    probabilities = np.dot(features, weights.reshape((num_features, num_states)))
    probabilities = softmax(probabilities)

    label_mat = np.zeros((probabilities.shape))

    for i in range(num_states):
        label_mat[:, i] = np.where(label_vec == i, 1, 0)

    return -np.sum(label_mat * np.nan_to_num(np.log(probabilities))) + np.dot(weights.ravel(), weights.ravel())
    # return -np.sum(label_mat * np.nan_to_num(np.log(probabilities)))

def gradient(weights, features, label_vec, num_features, num_states):
    probabilities = np.dot(features, weights.reshape((num_features, num_states)))
    probabilities = softmax(probabilities)

    label_mat = np.zeros((probabilities.shape))

    for i in range(num_states):
        label_mat[:, i] = np.where(label_vec == i, 1, 0)

    g = -features.T.dot(label_mat - probabilities).ravel() + 2 * weights.ravel()
    # g = -features.T.dot(label_mat - probabilities).ravel()
    return g

def accuracy(weights, features, label_vec,  num_features, num_states):
    total_error = 0
    probabilities = np.dot(features, weights.reshape((num_features, num_states)))
    probabilities = softmax(probabilities)
    positive_vec = np.argmax(probabilities, axis=1)
    error = np.sum(np.abs(positive_vec - label_vec))
    num_pixels = np.shape(label_vec)

    total_error = total_error + error
    accuracy = 1 - (total_error / num_pixels)

    return accuracy

def load_all_images_and_labels(path, num_features, num_images):
    loader = ImageLoader()
    all_pixel = np.random.randn(0, num_features)
    all_label = []

    # files = [x for x in os.listdir('./train') if x.endswith(".jpg") or x.endswith('.png')]
    files = [x for x in os.listdir(path) if x.endswith(".jpg") or x.endswith('.png')]
    for i, filename in enumerate(files):
        if i < num_images:
            full_name = os.path.join(path, filename)
            # full_name = os.path.join('./train', filename)
            img = Image.open(full_name)
            height = img.size[1]
            width = img.size[0]

            features, edge_features = ImageLoader.compute_features(img)
            pixel = np.asarray(list(features.values()))
            all_pixel = np.concatenate((all_pixel, pixel), axis=0)

            label_dict = loader.load_label_dict(full_name)
            label_vec = np.asarray(list(label_dict.values()))
            all_label = np.concatenate((all_label, label_vec), axis=0)

    return all_pixel, all_label


if __name__ == '__main__':
    unittest.main()