import numpy as np
import matplotlib.pyplot as plt
from Learner import Learner
from ImageLoader import ImageLoader
from MatrixBeliefPropagator import MatrixBeliefPropagator


class Evaluation(object):

    def __init__(self, max_width=0, max_height=0):
        self.max_width = max_width
        self.max_height = max_height

    def evaluation_images(self, directory, weights, num_states):

        loader = ImageLoader(self.max_width, self.max_height)

        images, models, labels, names = loader.load_all_images_and_labels(directory, num_states)

        average_errors = 0

        for i in range(len(images)):
            models[i].set_weights(weights)
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

            error_rate = np.true_divide(errors, images[i].width * images[i].height)
            baseline_rate = np.true_divide(baseline, images[i].width * images[i].height)

            # uncomment this to plot the beliefs
            plt.subplot(131)
            plt.imshow(images[i], interpolation="nearest")
            plt.subplot(132)
            plt.imshow(label_img, interpolation="nearest")
            plt.subplot(133)
            plt.imshow(beliefs, interpolation="nearest")
            plt.show()

            print("Results for the %dth image:" % (i + 1))
            print("Error rate: %f" % error_rate)
            print("Baseline from guessing all background: %f" % baseline_rate)
            assert errors < baseline, "Learned model did no better than guessing all background."

            average_errors += error_rate

        average_errors = np.true_divide(average_errors, i + 1)

        return average_errors


def main():
    """test evaluation"""

    loader = ImageLoader(16, 16)

    images, models, labels, names = loader.load_all_images_and_labels('./tests/train', 2)

    learner = Learner(MatrixBeliefPropagator)

    learner.set_regularization(0.0, 0.00001)

    for model, states in zip(models, labels):
        learner.add_data(states, model)

    d_unary = 65
    num_states = 2
    d_edge = 10

    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

    new_weights = learner.learn(weights)

    Eval = Evaluation(16, 16)
    average_errors = Eval.evaluation_images('./tests/test', new_weights, 2)
    print ("Average Error rate: %f" % average_errors)


if __name__ == '__main__':
    main()

