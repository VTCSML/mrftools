import itertools
import os
import time
import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .LogLinearModel import LogLinearModel
import tqdm
from torch.autograd import Variable
import torch
from torch.utils import data
import os.path as osp
from PIL import Image
from torch.nn.functional import upsample
from numpy.linalg import norm

import matplotlib.pyplot as plt


class ImageFeatureLoader(object):

    def __init__(self, max_width=0, max_height=0):
        """
        Initialize an ImageLoader

        :param max_width: maximum width of image to load. This object will resize any images that are wider than this.
        :type max_width: int
        :param max_height: maximum height of image to load. This object will resize any images that are taller.
        :type max_height: int
        """
        self.max_width = max_width
        self.max_height = max_height


    def load_images(self, dir, set):
        images = {}
        set_path = osp.join(dir, "%s.txt"%set)
        for line in file(set_path).readlines():
            name = line.strip()
            image_path = osp.join(dir, "images/%s.jpg"%name)
            image = self.load_image(image_path)
            images[name] = image
        return images

    def load_image(self, path):
        """
        Load image at path and resize according to our maximum size parameters
        :param path: location of image in file system
        :type path: string
        :return: PIL image object
        """
        img = Image.open(path)
        img1 = img

        if self.max_width > 0 and self.max_height > 0:
            img = img.resize((self.max_width, self.max_height), resample=PIL.Image.BICUBIC)

        return img

    def preprocess_label(self, label):
        h,w = label.size()
        resize_label = label.view(1,1,h,w)
        resize_label = upsample(resize_label, size=[self.max_width, self.max_height], mode = 'bilinear')
        label_data = resize_label.data.numpy()[0,0,:,:]
        label_data = np.round(label_data)
        label_data = label_data.astype(int)
        return label_data

    def preprocess_features(self, image):
        c,h,w = image.size()
        image= image.view(1,c,h,w)
        image = upsample(image, size=[self.max_width, self.max_height], mode = 'bilinear')
        image_data = image.data.numpy()[0,:,:,:]
        return image_data

    def load_features(self, dir, set_path):
        image_features = {}
        labels = {}
        names = list()
        for line in file(set_path).readlines():
            name = line.strip()
            image_file = osp.join(dir, "%s.pth.tar"%name)
            image = torch.load(image_file, map_location=lambda storage, loc: storage)
            features = image["feature"]
            ll = image["label"]
            features = self.preprocess_features(features)
            label = self.preprocess_label(ll)
            image_features[name] = features
            labels[name] = label
            names.append(name)
        return image_features, labels, names


    def load_label_dict(self, label):
        label_dict = dict()
        (h,w) = label.shape
        for x in range(0,h):
            for y in range(0,w):
                label_dict[(x, y)] = label[x, y]
        return label_dict

    def draw_image_and_label(self, name, image_path, label):
        """
        Draw an image and its ground truth label.

        :param name: path to image file
        :type name: string
        :return: None
        """
        img = self.load_image(osp.join(image_path, "%s.jpg"%name))

        plt.subplot(121)
        plt.imshow(img, interpolation='nearest')
        plt.xlabel('Original Image')
        plt.subplot(122)
        plt.imshow(label, interpolation='nearest')
        plt.xlabel("Labels")
        plt.show()

    def load_all_features_labels(self, dir, index, num_states):
        #images_path = osp.join(dir, "images")
        models = list()
        labels = list()
        features_path = osp.join(dir, "features")
        index_path = osp.join(dir, "%s.txt"%index)
        start = time.time()
        image_features, image_labels, names = self.load_features(features_path, index_path)
        i = 0
        num_images = len(names)
        for name in names:
            #print name
            model = ImageFeatureLoader.create_model(image_features[name], num_states, name)
            label_vec = self.load_label_dict(image_labels[name])
            models.append(model)
            labels.append(label_vec)
            i = i + 1

            if i % 10 == 0 or i == num_images - 1:
                elapsed = time.time() - start
                eta = np.true_divide(elapsed, i + 1) * (num_images - i - 1)
                print("Loaded %d of %d. Time elapsed: %f. ETA: %f" % (i + 1, num_images, elapsed, eta))

        return models, labels, names

    @staticmethod
    def create_model(image_features,num_states, name):
        model = LogLinearModel()
        (c,h,w) = image_features.shape
        tree_prob = ImageFeatureLoader.calculate_tree_probabilities_snake_shape(h, w)
        model.tree_probabilities = tree_prob
        feature_dict, edge_feature_dict = ImageFeatureLoader.compute_features(image_features, name)

        # create pixel variables
        for pixel, feature_vec in feature_dict.items():
            model.declare_variable(pixel, num_states)
            model.set_unary_features(pixel, feature_vec)
            model.set_unary_factor(pixel, np.zeros(num_states))

        # create edge variables
        for edge, edge_feature_vec in edge_feature_dict.items():
            model.set_edge_features(edge, edge_feature_vec)
            model.set_edge_factor(edge, np.eye(num_states))

        model.create_matrices()

        return model

    @staticmethod
    def show_images(images):
        """
        Draw images onscreen.

        :param images: iterable of images
        :type images: iterable
        :return: None
        """
        plt.clf()
        total = len(images)

        rows = np.ceil(np.sqrt(total))
        cols = rows

        for i, img in enumerate(images):
            plt.clf()
            plt.imshow(img, interpolation='nearest')
            plt.pause(1e-10)

    @staticmethod
    def get_all_edges(height,width):

        edges = []

        # add horizontal edges
        for x in range(width - 1):
            for y in range(height):
                edge = ((x, y), (x + 1, y))
                edges.append(edge)

        # add vertical edges
        for x in range(width):
            for y in range(height - 1):
                edge = ((x, y), (x, y + 1))
                edges.append(edge)

        return edges

    @staticmethod
    def calculate_tree_probabilities_snake_shape(width, height):
        """
        Calculate spanning-tree edge appearance probabilities by considering two "snakes" that cover the graph going
        north-to-south and east-to-west.

        :param width: width of grid MRF
        :type width: int
        :param height: height of grid MRF
        :type height: int
        :return: dictionary of edge appearance probabilities under the two-snake spanning tree distribution
        :rtype: dict
        """
        tree_prob = dict()
        for x in range(width):
            for y in range(height - 1):
                if x == 0 or x == width - 1:
                    tree_prob[((x, y), (x, y + 1))] = 0.75
                    tree_prob[((x, y + 1), (x, y))] = 0.75
                else:
                    tree_prob[((x, y), (x, y + 1))] = 0.5
                    tree_prob[((x, y + 1), (x, y))] = 0.5

        for x in range(width - 1):
            for y in range(height):
                if y == 0 or y == height - 1:
                    tree_prob[((x, y), (x + 1, y))] = 0.75
                    tree_prob[((x + 1, y), (x, y))] = 0.75
                else:
                    tree_prob[((x, y), (x + 1, y))] = 0.5
                    tree_prob[((x + 1, y), (x, y))] = 0.5

        return tree_prob


    @staticmethod
    def compute_features(image_features, name):
        (channel, height, width) = image_features.shape
        feature_dict = {}
        nthresh = 10

        for x in range(width):
            for y in range(height):
                feature_dict[(x,y)] = image_features[:,x,y]

        edges = ImageFeatureLoader.get_all_edges(height, width)

        edge_feature_mat = np.zeros((len(edges), nthresh + 1))
        diff_list = list()
        for j, edge in enumerate(edges):
            diff = 0
            edge_feats_vec = np.zeros(nthresh + 1)
            pair_wise_feature = image_features[:,edge[0][0], edge[0][1]] - image_features[:,edge[1][0], edge[1][1]]
            diff = norm(pair_wise_feature, 2)
            diff_list.append(diff)
            # for n in range(nthresh):
            #     thresh = .5 * n / nthresh
            #     edge_feats_vec[n] = 1 * (diff > thresh)
            # edge_feats_vec[-1] = 1.0  # add bias feature
            # edge_feature_mat[j, :] = edge_feats_vec

        edge_feature_vectors = [np.array(x) for x in edge_feature_mat.tolist()]
        edge_feature_dict = dict(zip(edges, edge_feature_vectors))
        # plt.hist(diff_list, normed=False, bins=10)
        # plt.savefig("/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/results/%s.jpg"%name)
        # plt.clf()

        return feature_dict, edge_feature_dict













