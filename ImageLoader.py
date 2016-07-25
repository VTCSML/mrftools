import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os
import itertools
from LogLinearModel import LogLinearModel
import time


class ImageLoader(object):

    def __init__(self, max_width=0, max_height=0):
        self.max_width = max_width
        self.max_height = max_height

    def load_image(self, path):
        img = Image.open(path)
        if self.max_width > 0 and self.max_height > 0:
            img = img.resize((self.max_width, self.max_height), resample=PIL.Image.BICUBIC)
        return img

    def load_label_img(self, image_name):
        label_file = os.path.splitext(image_name)[0] + '_label.txt'
        label_mat = np.loadtxt(label_file)

        label_img = Image.fromarray(label_mat.astype(np.uint8))

        if self.max_width > 0 and self.max_height > 0:
            label_img = label_img.resize((self.max_width, self.max_height), resample=PIL.Image.NEAREST)

        return label_img

    def load_label_dict(self, image_name):
        label_img = self.load_label_img(image_name)

        label_pixels = label_img.load()
        label_dict = dict()
        for x in range(label_img.width):
            for y in range(label_img.height):
                label_dict[(x, y)] = label_pixels[x, y]

        return label_dict

    def draw_image_and_label(self, name):
        img = self.load_image(name)
        labels = self.load_label_img(name)
        features = ImageLoader.compute_features(img)

        plt.subplot(121)
        plt.imshow(img, interpolation='nearest')
        plt.xlabel('Original Image')
        plt.subplot(122)
        plt.imshow(labels, interpolation='nearest')
        plt.xlabel("Labels")
        plt.show()

    def load_all_images_and_labels(self, directory, num_states, num_images):
        images = []
        models = []
        labels = []
        names = []
        files = [x for x in os.listdir(directory) if x.endswith(".jpg") or x.endswith('.png')]
        start = time.time()
        for i, filename in enumerate(files):
            if i < num_images:
                full_name = os.path.join(directory, filename)
                img = self.load_image(full_name)
                model = ImageLoader.create_model(img, num_states)
                label_vec = self.load_label_dict(full_name)

                names.append(filename)
                images.append(img)
                models.append(model)

                labels.append(label_vec)

                if i % 10 == 0 or i == num_images-1:
                    elapsed = time.time() - start
                    eta = np.true_divide(elapsed, i + 1) * (len(files) - i - 1)
                    print("Loaded %d of %d. Time elapsed: %f. ETA: %f" % (i+1, num_images, elapsed, eta))

        return images, models, labels, names

    @staticmethod
    def create_model(img, num_states):
        model = LogLinearModel()

        feature_dict, edge_feature_dict = ImageLoader.compute_features(img)

        # create pixel variables
        for pixel, feature_vec in feature_dict.items():
            model.declare_variable(pixel, num_states)
            model.set_unary_features(pixel, feature_vec)
            model.set_unary_factor(pixel, np.zeros(num_states))

        # create edge variable
        for edge, edge_feature_vec in edge_feature_dict.items():
            model.set_edge_features(edge, edge_feature_vec)
            model.set_edge_factor(edge, np.eye(num_states))

        model.create_matrices()

        return model

    @staticmethod
    def show_images(images):
        plt.clf()
        total = len(images)

        rows = np.ceil(np.sqrt(total))
        cols = rows

        for i, img in enumerate(images):
            plt.clf()
            plt.imshow(img, interpolation='nearest')
            plt.pause(1e-10)

    @staticmethod
    def get_all_edges(img):
        edges = []

        # add horizontal edges
        for x in range(img.width-1):
            for y in range(img.height):
                edge = ((x, y), (x+1, y))
                edges.append(edge)

        # add vertical edges
        for x in range(img.width):
            for y in range(img.height-1):
                edge = ((x, y), (x, y+1))
                edges.append(edge)

        return edges

    @staticmethod
    def compute_features(img):
        pixels = img.load()

        base_features = np.zeros((img.width * img.height, 5))
        pixel_ids = []
        nthresh = 10
        edge_ids = []

        i = 0
        for x in range(img.width):
            for y in range(img.height):
                base_features[i, :3] = pixels[x, y]
                base_features[i, 3:] = (x, y)

                pixel_ids.append((x, y))

                i += 1

        base_features /= [255, 255, 255, img.width, img.height]

        # perform fourier expansion

        coeffs = list(itertools.product([0, 1], repeat=5))
        coeffs = np.column_stack(coeffs)

        prod = base_features.dot(coeffs)
        feature_mat = np.hstack((np.sin(prod), np.cos(prod), np.ones((img.width * img.height, 1))))

        if img.mode == 'RGB':
            channels = 3
        elif img.mode == 'L':
            channels = 1
        else:
            print("Unknown mode: %s" % img.mode)

        j = 0
        for edge in ImageLoader.get_all_edges(img):

            edge_ids.append(edge)
            diff = 0
            edge_feats_vec = []
            for z in range(channels):
                diff += np.true_divide((pixels[edge[0]][z]-pixels[edge[1]][z]), 255) ** 2

            diff = np.sqrt(diff)
            for n in range(nthresh):
                thresh = .5 * n / nthresh
                edge_feats_vec.append(1*(diff > thresh))

            if j == 0:
                edge_feature_mat = edge_feats_vec
            else:
                edge_feature_mat = np.vstack((edge_feature_mat, edge_feats_vec))

            j += 1

        # package up feature matrix as feature dictionary
        feature_vectors = [np.array(x) for x in feature_mat.tolist()]
        feature_dict = dict(zip(pixel_ids, feature_vectors))

        edge_feature_vectors = [np.array(x) for x in edge_feature_mat.tolist()]
        edge_feature_dict = dict(zip(edge_ids, edge_feature_vectors))

        return feature_dict, edge_feature_dict

def main():
    """test loading"""

    loader = ImageLoader()
    images, models, labels, names = loader.load_all_images_and_labels('./tests/train', 2)

    files = [x for x in os.listdir('./tests/train') if x.endswith(".jpg") or x.endswith('.png')]
    for i, filename in enumerate(files):
        full_name = os.path.join('./tests/train', filename)
        loader.draw_image_and_label(full_name)

if __name__ == '__main__':
    main()

