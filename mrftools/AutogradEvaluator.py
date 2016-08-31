try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
import matplotlib.pyplot as plt
from AutogradLearner import AutogradLearner
from ImageLoader import ImageLoader
from MatrixBeliefPropagator import MatrixBeliefPropagator


class AutogradEvaluator(object):

    def __init__(self, max_width=0, max_height=0):
        self.max_width = max_width
        self.max_height = max_height

    def plot_images(self, images, models, labels, names, weights, num_states, num_images, inference_type, max_iter= 300):
        np.set_printoptions(precision=10)
        loader = ImageLoader(self.max_width, self.max_height)

        beliefs_dic = {}

        for i in range(len(images)):
            if i < num_images:
                for key in weights.keys():
                    w = weights[key]
                    models[i].set_weights(w)
                    bp = inference_type(models[i])
                    bp.set_max_iter(max_iter)
                    bp.infer(display='off')
                    bp.load_beliefs()

                    beliefs = np.zeros((images[i].height, images[i].width))
                    label_img = np.zeros((images[i].height, images[i].width))
                    for x in range(images[i].width):
                        for y in range(images[i].height):
                            beliefs[y, x] = np.exp(bp.var_beliefs[(x, y)][1])
                            if (x, y) in labels[i]:
                                label_img[y, x] = labels[i][(x, y)]

                    beliefs_dic[key] = beliefs

                self.draw_results(images[i], label_img, beliefs_dic)

    def evaluate_training_images(self, images, models, labels, names, weights, num_states, num_images, inference_type, max_iter= 300, inc='false', plot = 'true', display='final'):
        np.set_printoptions(precision=10)
        loader = ImageLoader(self.max_width, self.max_height)
        # images, models, labels, names = loader.load_all_images_and_labels(directory, num_states, num_images)

        average_errors = 0
        total_inconsistency = 0

        for i in range(len(images)):
            if i < num_images:
                models[i].set_weights(weights)
                bp = inference_type(models[i])
                bp.set_max_iter(max_iter)
                bp.infer(display='off')
                bp.load_beliefs()

                beliefs = np.zeros((images[i].height, images[i].width))
                label_img = np.zeros((images[i].height, images[i].width))
                errors = 0
                baseline = 0
                num_latent = 0

                for x in range(images[i].width):
                    for y in range(images[i].height):
                        beliefs[y, x] = np.exp(bp.var_beliefs[(x, y)][1])
                        if (x, y) in labels[i]:
                            label_img[y, x] = labels[i][(x, y)]
                            errors += np.abs(labels[i][(x, y)] - np.round(beliefs[y, x]))
                            baseline += labels[i][(x, y)]
                        else:
                            num_latent += 1


                error_rate = np.true_divide(errors, images[i].width * images[i].height - num_latent)
                baseline_rate = np.true_divide(baseline, images[i].width * images[i].height)

                if plot == True:
                    self.draw_results(images[i], label_img, beliefs)

                if display == 'full':
                    print("Results for the %dth image:" % (i + 1))
                    print("Error rate: %f" % error_rate)
                    print("Baseline from guessing all background: %f" % baseline_rate)
                if inc == True:
                    inconsistency = bp.compute_inconsistency()
                    total_inconsistency += inconsistency
                    if display == 'full':
                        print("inconsistency of %s: %f" % (names[i], inconsistency))

                average_errors += error_rate

            average_errors = np.true_divide(average_errors, i + 1)
        if inc == True:
            print("Overall inconsistency: %f" % total_inconsistency)
            return average_errors, total_inconsistency

        return average_errors

    def evaluate_testing_images(self, directory, weights, num_states, num_images, inference_type, max_iter= 300, inc= False, plot = True, display = 'final'):
        np.set_printoptions(precision=10)
        loader = ImageLoader(self.max_width, self.max_height)

        images, models, labels, names = loader.load_all_images_and_labels(directory, num_states, num_images)
        if inc == True:
            average_errors, total_inconsistency = self.evaluate_training_images(images, models, labels, names, weights, num_states, num_images, inference_type,
                                     max_iter, inc, plot)
            return average_errors, total_inconsistency


        average_errors = self.evaluate_training_images(self, images, models, labels, names, weights, num_states, num_images, inference_type,
                                 max_iter, inc, plot)

        return average_errors

    def draw_results(self, image, label, beliefs):
        if isinstance(beliefs, dict):
            num_methods = len(beliefs)
            p = num_methods + 2
            plt.subplot(1, p, 1)
            plt.title('true image')
            plt.imshow(image, interpolation="nearest")
            plt.subplot(1, p, 2)
            plt.title('true label')
            plt.imshow(label, interpolation="nearest")
            c = 0
            for key in beliefs.keys():
                plt.subplot(1, p, c + 3)
                kk = str(key).split('.')
                ttl =  kk[len(kk)-1][:-2]
                plt.title(ttl)
                plt.imshow(beliefs[key], interpolation="nearest")
                c += 1
            plt.show()
        else:
            plt.subplot(131)
            plt.imshow(image, interpolation="nearest")
            plt.subplot(132)
            plt.imshow(label, interpolation="nearest")
            plt.subplot(133)
            plt.imshow(beliefs, interpolation="nearest")
            plt.show()


    def evaluate_objective(self, method_list, path):

        for i in range(0,len(method_list)):
            m_dic = method_list[i]
            obj_time = m_dic['time']
            obj = m_dic['objective']
            method_name = m_dic['method']
            kk = str(method_name).split('.')
            ttl = kk[len(kk) - 1][:-2]

            plt.plot(obj_time, obj, '-', linewidth=2, label=ttl)
        plt.xlabel('time(seconds)')
        plt.ylabel('objective')
        plt.legend(loc='upper right')

        plt.title('objective function trend')
        plt.savefig(path + '/objective')

        # plt.show()

    def evaluate_training_accuracy(self, method_list, path):
        plt.clf()
        for i in range(0,len(method_list)):
            m_dic = method_list[i]
            obj_time = m_dic['time']
            method_name = m_dic['method']
            accuracy = m_dic['training_error']

            kk = str(method_name).split('.')
            ttl = kk[len(kk) - 1][:-2]
            plt.plot(obj_time, accuracy, '-', linewidth=2, label=ttl)

        plt.xlabel('time(seconds)')
        plt.ylabel('training error')
        plt.legend(loc='upper right')

        plt.title('training error trend')
        plt.savefig(path + '/training_error')

        # plt.show()

def main():
    """test evaluation"""

    loader = ImageLoader(16, 16)

    images, models, labels, names = loader.load_all_images_and_labels('./tests/train', 2, 2)

    learner = AutogradLearner(MatrixBeliefPropagator)

    learner.set_regularization(0.0, 0.00001)

    for model, states in zip(models, labels):
        learner.add_data(states, model)

    d_unary = 65
    num_states = 2
    d_edge = 10

    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

    new_weights = learner.learn(weights)

    Eval = AutogradEvaluator(16, 16)
    average_errors = Eval.evaluation_images('./tests/test', new_weights, 2, 2)
    print ("Average Error rate: %f" % average_errors)


if __name__ == '__main__':
    main()