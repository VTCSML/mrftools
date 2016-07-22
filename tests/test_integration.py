from ImageLoader import ImageLoader
from Learner import Learner
import numpy as np
from MatrixBeliefPropagator import MatrixBeliefPropagator

import unittest
import matplotlib.pyplot as plt

class IntegrationTest(unittest.TestCase):

    def test_loading_and_learning(self):
        loader = ImageLoader(32, 32)

        images, models, labels, names = loader.load_all_images_and_labels('./train', 2)

        learner = Learner(MatrixBeliefPropagator)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        weights = np.zeros(65 * 2 + 4)

        new_weights = learner.learn(weights)

        unary_mat = new_weights[:65 * 2].reshape((65, 2))
        pair_mat = new_weights[65*2:].reshape((2, 2))
        print("Unary weights:\n" + repr(unary_mat))
        print("Pairwise weights:\n" + repr(pair_mat))

        # test inference with weights

        i = 1

        models[i].set_weights(new_weights)
        bp = MatrixBeliefPropagator(models[i])
        bp.infer()
        bp.load_beliefs()

        beliefs = np.zeros((images[i].height, images[i].width))
        errors = 0
        baseline = 0

        for x in range(images[i].width):
            for y in range(images[i].height):
                beliefs[y, x] = np.exp(bp.var_beliefs[(x, y)][1])
                errors += np.abs(labels[i][(x, y)] - np.round(beliefs[y, x]))
                baseline += labels[i][(x, y)]

        # # uncomment this to plot the beliefs
        # plt.subplot(121)
        # plt.imshow(images[i], interpolation="nearest")
        # plt.subplot(122)
        # plt.imshow(beliefs, interpolation="nearest")
        # plt.show()

        print("Error rate: %f" % np.true_divide(errors, images[i].width * images[i].height))
        print("Baseline from guessing all background: %f" % np.true_divide(baseline, images[i].width * images[i].height))
        assert errors < baseline, "Learned model did no better than guessing all background."

if __name__ == '__main__':
    unittest.main()