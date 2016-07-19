import unittest
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import os
from ImageLoader import ImageLoader


class TestImageLoader(unittest.TestCase):

    def test_load_draw(self):

        loader = ImageLoader()
        images, models, labels, names = loader.load_all_images_and_labels('./train', 2)
        files = [x for x in os.listdir('./train') if x.endswith(".jpg") or x.endswith('.png')]
        for i, filename in enumerate(files):
            full_name = os.path.join('./train', filename)
            img = Image.open(full_name)
            features = models[i].unaryFeatures
            assert np.allclose(len(labels[i]), img.width * img.height), "the size of labels is right"
            assert np.allclose(len(features), img.width * img.height), "the size of features is right"


if __name__ == '__main__':
    unittest.main()

