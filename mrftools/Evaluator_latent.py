import numpy as np
import matplotlib.pyplot as plt
from Learner import Learner
from ImageLoader import ImageLoader
from MatrixBeliefPropagator import MatrixBeliefPropagator
import pylab
plt.switch_backend('agg')


class Evaluator_latent(object):

    def __init__(self, max_width=0, max_height=0):
        self.max_width = max_width
        self.max_height = max_height

    def plot_images(self, saved_path, images, models, labels, names, weights, num_states, num_images, inference_type, max_iter= 300):
        np.set_printoptions(precision=10)

        beliefs_dic = {}
        for i in range(len(images)):
            if i < num_images:
                for key in weights.keys ( ):
                    w = weights[key]
                    models[i].set_weights(w)
                    bp = inference_type(models[i])
                    # bp.set_max_iter(max_iter)
                    bp.infer(display='off')
                    bp.load_beliefs()
                    beliefs = np.zeros ( (images[i].height, images[i].width) )
                    label_img = np.zeros ( (images[i].height, images[i].width) )
                    for x in range ( images[i].height ):
                        for y in range ( images[i].width ):
                            beliefs[x, y] = np.argmax(np.exp(bp.var_beliefs[(y, x)]))
                            if (y, x) in labels[i]:
                                label_img[x, y] = labels[i][(y,x)]
                            else:
                                label_img[x, y] = -100

                    beliefs_dic[key] = beliefs

                self.draw_results ( images[i], label_img, beliefs_dic, names[i], saved_path )


    def evaluate_training_images(self, saved_path, images, models, labels, names, weights, num_states, num_images, inference_type, max_iter= 300, inc='false', plot = 'true', display='final'):
        np.set_printoptions(precision=10)
        loader = ImageLoader(self.max_width, self.max_height)
        # images, models, labels, names = loader.load_all_images_and_labels(directory, num_states, num_images)

        average_errors = 0
        total_inconsistency = 0
        for i in range(len(images)):
            if i < num_images:
                models[i].set_weights(weights)
                bp = inference_type(models[i])
                # bp.set_max_iter(max_iter)
                bp.infer(display='off')
                bp.load_beliefs()

                beliefs = np.zeros((images[i].height, images[i].width))
                errors = 0
                baseline = 0
                num_latent = 0
                for x in range(images[i].height):
                    for y in range(images[i].width):
                        beliefs[x,y] = np.argmax(np.exp(bp.var_beliefs[(y,x)]))

                        if (y,x) in labels[i]:
                            if beliefs[x,y] != labels[i][(y,x)]:
                                errors += 1
                        else:
                            num_latent += 1

                error_rate = np.true_divide ( errors, images[i].width * images[i].height - num_latent )
                baseline_rate = np.true_divide ( baseline, images[i].width * images[i].height )


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

        average_errors = np.true_divide(average_errors, num_images)

        if inc == True:
            print("Overall inconsistency: %f" % total_inconsistency)
            return average_errors, total_inconsistency

        return average_errors

    def evaluate_testing_images(self, saved_path, directory, weights, num_states, num_images, inference_type, max_iter= 300, inc= False, plot = True, display = 'final'):
        np.set_printoptions(precision=10)
        loader = ImageLoader(self.max_width, self.max_height)

        images, models, labels, names = loader.load_all_images_and_labels(directory, num_states, num_images)
        if inc == True:
            average_errors, total_inconsistency = self.evaluate_training_images(saved_path, images, models, labels, names, weights, num_states, num_images, inference_type,
                                     max_iter, inc, plot)
            return average_errors, total_inconsistency


        average_errors = self.evaluate_training_images( saved_path, images, models, labels, names, weights, num_states, num_images, inference_type,
                                 max_iter, inc, plot)

        return average_errors

    def draw_results(self, image, label, beliefs, name ,saved_path):
        plt.clf()
        if isinstance(beliefs, dict):
            num_methods = len(beliefs)
            p = num_methods + 2
            col = 3
            row = p/col
            if  p % 3 > 0 :
                row = row + 1
            plt.subplot(row, col, 1)
            plt.title('true image')
            plt.imshow(image, interpolation="nearest")
            plt.subplot(row, col, 2)
            plt.title('true label.png')
            seg_label = self.create_img ( label )
            plt.imshow ( seg_label )
            c = 1
            for key in beliefs.keys():
                plt.subplot(row, col, c+2)
                plt.title(str(key))
                seg_label = self.create_img ( beliefs[key] )
                res = plt.imshow(seg_label)
                c += 1
            plt.savefig ( saved_path + name )

        else:
            plt.subplot(131)
            plt.imshow(image, interpolation="nearest")
            plt.subplot(132)
            plt.imshow(label, interpolation="nearest")
            plt.subplot(133)
            plt.imshow(beliefs, interpolation="nearest")
            # plt.show()
            plt.savefig(saved_path + name)

    def create_img(self, label):


        color_dic = {0: [[160, 160, 160], 'gray=sky'], 1: [[153, 153, 0], 'dark green=tree'], 2: [[102, 0, 204], 'purple=road'],
                     3: [[0, 153, 76], 'green=grass'], 4: [[54, 145, 236], 'blue=water'], 5: [[213, 31, 31], 'dark red=building'],
                     6: [[153, 76, 0], 'brown=mountain'], 7: [[255, 153, 51], 'orange=foreground'], -100:[[255,255,255], 'white=latent']}

        h = label.shape[0]
        w = label.shape[1]


        new_seg = np.empty ( (h, w, 3) ,dtype= np.uint8)
        for i in range ( 0, h ):
            for j in range ( 0, w ):
                # print label[i,j]
                # print color_dic[label[i, j]][0]
                new_seg[i, j, :] = color_dic[label[i, j]][0]

        return new_seg

    def evaluate_objective(self, method_list, path):
        plt.clf ( )
        for i in range(0,len(method_list)):
            m_dic = method_list[i]
            obj_time = np.arange(100)
            # obj_time = m_dic['time']
            obj = m_dic['objective']
            ttl = m_dic['learner_name']

            plt.plot(obj_time, obj, '-', linewidth=2, label=ttl)

        plt.xlabel('time(seconds)')
        plt.ylabel('objective')
        plt.legend(loc='upper right')

        plt.title('objective function trend')
        if 'Loss' in method_list[0]['learner_name']:
            plt.savefig(path + '/objective_loss_augmented')
        else:
            plt.savefig ( path + '/objective' )

        # plt.show()

    def evaluate_training_accuracy(self, method_list, path,mode):
        plt.clf()
        for i in range(0,len(method_list)):
            m_dic = method_list[i]
            # obj_time = m_dic['time']
            obj_time = np.arange ( 100 )
            if mode == 'train':
                accuracy = m_dic['training_error']
            else:
                accuracy = m_dic['testing_error']

            ttl = m_dic['learner_name']
            plt.plot(obj_time, accuracy, '-', linewidth=2, label=ttl)


        if mode == 'train':
            plt.xlabel ( 'time(seconds)' )
            plt.ylabel ( 'training error' )
            plt.legend ( loc='upper right' )

            plt.title ( 'training error trend' )
            plt.savefig ( path + '/training_error' )
        else:
            plt.xlabel ( 'time(seconds)' )
            plt.ylabel ( 'test error' )
            plt.legend ( loc='upper right' )

            plt.title ( 'test error trend' )
            plt.savefig ( path + '/test_error' )





        # plt.show()

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