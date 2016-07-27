import numpy as np
import matplotlib.pyplot as plt
from Learner import Learner
from ImageLoader import ImageLoader
from MatrixBeliefPropagator import MatrixBeliefPropagator


class Evaluator(object):

    def __init__(self, max_width=0, max_height=0):
        self.max_width = max_width
        self.max_height = max_height

    def evaluate_images(self, directory, weights, num_states, num_images, inference_type, max_iter= 300, inc='false', plot = 'true'):
        np.set_printoptions(precision=10)
        loader = ImageLoader(self.max_width, self.max_height)

        images, models, labels, names = loader.load_all_images_and_labels(directory, num_states, num_images)

        average_errors = 0
        total_inconsistency = 0

        for i in range(len(images)):
            if i < num_images:
                models[i].set_weights(weights)
                bp = inference_type(models[i])
                bp.set_max_iter(max_iter)
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

                if plot == 'true':
                    self.draw_results(images[i], label_img, beliefs)

                print("Results for the %dth image:" % (i + 1))
                print("Error rate: %f" % error_rate)
                print("Baseline from guessing all background: %f" % baseline_rate)
                if inc == "true":
                    inconsistency = bp.compute_inconsistency()
                    total_inconsistency += inconsistency
                    print("inconsistency of %s: %f" % (names[i], inconsistency))

                average_errors += error_rate

            average_errors = np.true_divide(average_errors, i + 1)
        if inc == "true":
            print("Overall inconsistency: %f" % np.sum(total_inconsistency))


        return average_errors

    def draw_results(self, image, label, beliefs):
        plt.subplot(131)
        plt.imshow(image, interpolation="nearest")
        plt.subplot(132)
        plt.imshow(label, interpolation="nearest")
        plt.subplot(133)
        plt.imshow(beliefs, interpolation="nearest")
        plt.show()


def main():
    """test evaluation"""

    loader = ImageLoader(16, 16)

    images, models, labels, names = loader.load_all_images_and_labels('./tests/train', 2, 2)

    learner = Learner(MatrixBeliefPropagator)

    learner.set_regularization(0.0, 0.00001)

    for model, states in zip(models, labels):
        learner.add_data(states, model)

    d_unary = 65
    num_states = 2
    d_edge = 10

    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

    new_weights = learner.learn(weights)

    Eval = Evaluator(16, 16)
    average_errors = Eval.evaluation_images('./tests/test', new_weights, 2, 2)
    print ("Average Error rate: %f" % average_errors)


if __name__ == '__main__':
    main()

