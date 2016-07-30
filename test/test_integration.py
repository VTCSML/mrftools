from mrftools import *
import numpy as np
import unittest
import os
import matplotlib.pyplot as plt


class TestIntegration(unittest.TestCase):

    def test_loading_and_learning(self):
        loader = ImageLoader(20, 20)

        images, models, labels, names = loader.load_all_images_and_labels(os.path.join(os.path.dirname(__file__), 'train'), 2, 3)

        learner = Learner(MatrixBeliefPropagator)

        learner.set_regularization(0.0, 0.00001)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

        d_unary = 65
        num_states = 2
        d_edge = 10

        weights = np.zeros(d_unary * num_states + d_edge * num_states**2)

        new_weights = learner.learn(weights)

        unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
        pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states**2))
        print("Unary weights:\n" + repr(unary_mat))
        print("Pairwise weights:\n" + repr(pair_mat))

        # test inference with weights

        i = 1

        models[i].set_weights(new_weights)
        bp = MatrixBeliefPropagator(models[i])
        bp.infer(display='final')
        bp.load_beliefs()

        beliefs = np.zeros((images[i].height, images[i].width))
        label_img = np.zeros((images[i].height, images[i].width))
        errors = 0
        baseline = 0

        for x in range(images[i].width):
            for y in range(images[i].height):
                beliefs[y, x] = np.exp(bp.var_beliefs[(x, y)][1])
                label_img[y, x] = labels[i][(x, y)]
                errors += np.abs(labels[i][(x, y)] - np.round(beliefs[y, x]))
                baseline += labels[i][(x, y)]

        # # uncomment this to plot the beliefs
        # plt.subplot(131)
        # plt.imshow(images[i], interpolation="nearest")
        # plt.subplot(132)
        # plt.imshow(label_img, interpolation="nearest")
        # plt.subplot(133)
        # plt.imshow(beliefs, interpolation="nearest")
        # plt.show()

        print("Error rate: %f" % np.true_divide(errors, images[i].width * images[i].height))
        print("Baseline from guessing all background: %f" % np.true_divide(baseline, images[i].width * images[i].height))
        assert errors < baseline, "Learned model did no better than guessing all background."

    def test_consistency(self):
        loader = ImageLoader(1, 2)

        images, models, labels, names = loader.load_all_images_and_labels(
            os.path.join(os.path.dirname(__file__), 'train'), 2, 1)
        i = 0

        d_unary = 65
        num_states = 2
        d_edge = 10

        new_weights = 0.1 * np.random.randn(d_unary * num_states + d_edge * num_states ** 2)

        models[i].set_weights(new_weights)
        bp = MatrixBeliefPropagator(models[i])
        bp.infer(display='full')
        bp.load_beliefs()

        models[i].load_factors_from_matrices()

        bf = BruteForce(models[i])
        unary_error = np.sum(np.abs(bf.unary_marginal((0, 0)) - np.exp(bp.var_beliefs[(0, 0)])))
        print "Unary marginal error compared to brute force: %e" % unary_error
        edge = models[i].edges[0]
        edge_error = np.sum(np.abs(bf.pairwise_marginal(edge[0], edge[1]) - np.exp(bp.pair_beliefs[edge])))
        print "Pair %s marginal error compared to brute force: %e" % (edge, edge_error)
        assert unary_error < 1e-3, "Unary error was too large compared to brute force"
        assert edge_error < 1e-3, "Pairwise error was too large compared to brute force"

        for var in bp.mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            assert np.allclose(np.sum(unary_belief), 1.0), "Unary belief not normalized"
            for neighbor in bp.mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                assert np.allclose(np.sum(pair_belief), 1.0), "Pair belief not normalized"
                print pair_belief, unary_belief
                assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"


if __name__ == '__main__':
    unittest.main()