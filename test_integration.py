from SegLoader import SegLoader
from Learner import Learner
import numpy as np
from MatrixBeliefPropagator import MatrixBeliefPropagator

import unittest

class IntegrationTest(unittest.TestCase):

    def test_loading_and_learning(self):
        loader = SegLoader(16, 16)

        images, models, labels, names = loader.load_all_images_and_labels('./train', 2)

        learner = Learner(MatrixBeliefPropagator)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        weights = np.zeros(64 * 2)

        learner.learn(weights)

