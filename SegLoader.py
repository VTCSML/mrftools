import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os
import itertools


class SegLoader(object):

    def __init__(self, max_width=0, max_height=0):
        self.images = []
        self.names = []
        self.features = []
        self.pixel_ids = []
        self.labels = []
        self.max_width = max_width
        self.max_height = max_height

    def load_image(self, path):
        img = Image.open(path)
        if self.max_width > 0 and self.max_height > 0:
            img = img.resize((self.max_width, self.max_height), resample=PIL.Image.NEAREST)

        self.names.append(path)
        self.images.append(img)
        features, pixel_ids = self.compute_features(img)
        self.features.append(features)
        self.pixel_ids.append(pixel_ids)

    def load_images(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                self.load_image(os.path.join(directory, filename))

    def show_images(self):
        plt.clf()
        total = len(self.images)

        rows = np.ceil(np.sqrt(total))
        cols = rows

        for i, img in enumerate(self.images):
            plt.clf()
            plt.imshow(img, interpolation='nearest')
            plt.pause(1e-10)

    def compute_features(self, img):
        pixels = img.load()

        base_features = np.zeros((img.width * img.height, 5))
        pixel_ids = []

        i = 0
        for x in range(img.height):
            for y in range(img.width):
                base_features[i, :3] = pixels[y, x]
                base_features[i, 3:] = (y, x)

                pixel_ids.append((y, x))

                i += 1

        base_features /= [255, 255, 255, img.width, img.height]

        # perform fourier expansion

        coeffs = list(itertools.product([0, 1], repeat=5))
        coeffs = np.column_stack(coeffs)

        prod = base_features.dot(coeffs)

        feature_mat = np.hstack((np.sin(prod), np.cos(prod)))

        return feature_mat, pixel_ids

    def load_all_labels(self):
        for i, name in enumerate(self.names):
            label_file = os.path.splitext(name)[0] + '_label.txt'
            label_mat = np.loadtxt(label_file)

            label_img = Image.fromarray(label_mat.astype(np.uint8))

            if self.max_width > 0 and self.max_height > 0:
                label_img = label_img.resize((self.max_width, self.max_height), resample=PIL.Image.NEAREST)

            self.labels.append(np.array(list(label_img.getdata())))

    def draw_image_and_label(self, i):
        plt.subplot(131)
        plt.imshow(self.images[i], interpolation='nearest')
        plt.xlabel('Original Image')
        plt.subplot(132)
        # k = np.random.randint(0, self.features[i].shape[1])
        k = 4
        plt.imshow(self.features[i][:, k].reshape(self.images[i].height, self.images[i].width), interpolation='nearest')
        plt.xlabel("Feature %d" % k)
        plt.subplot(133)
        plt.imshow(self.labels[i].reshape(self.images[i].height, self.images[i].width), interpolation='nearest')
        plt.xlabel("Labels")
        plt.show()

    def get_all_edges(self, i):
        edges = []

        # add horizontal edges
        for x in range(self.images[i].width-1):
            for y in range(self.images[i].height):
                edge = ((x, y), (x+1, y))
                edges.append(edge)

        # add vertical edges
        for x in range(self.images[i].width):
            for y in range(self.images[i].height-1):
                edge = ((x, y), (x, y+1))
                edges.append(edge)

        return edges


def main():
    """test loading"""

    loader = SegLoader()
    loader.load_images('./train')
    # loader.show_images()
    loader.load_all_labels()

    loader.draw_image_and_label(3)

    # print loader.get_all_edges(1)


if __name__ == '__main__':
    main()

