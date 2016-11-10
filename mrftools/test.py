import numpy as np
from ImageLoader import ImageLoader
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from EM import EM
from PairedDual import PairedDual
from PrimalDual import PrimalDual
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from Evaluator_latent import Evaluator_latent
import os
from opt import WeightRecord
import pickle
import copy
import sys
from MaxProductBeliefPropagator import MaxProductBeliefPropagator
from MaxProductLinearProgramming import MaxProductLinearProgramming
from ConvexBeliefPropagator import ConvexBeliefPropagator
from opt import *
import matplotlib.pyplot as plt




def learn_image(saved_path, data_path, learn_method, inference_type, models, labels, num_states, names, images, num_training_images, max_iter, max_height,
                max_width, weights,optm,noise,loss_augmented = False, regularizer = [0, 1], num_testing_images=1, MAP_Convex = False):

    learner = learn_method(inference_type )
    learner.MAP_Convex_inference = MAP_Convex

    learner.set_regularization(regularizer[0], regularizer[1])
    if loss_augmented == True:
        learner.loss_augmented = True
        kk = str ( learn_method ).split ( '.' )
        learner_name = kk[len ( kk ) - 1][:-2]
        learner_name = 'Loss_Augmented_' + learner_name
    else:
        kk = str ( learn_method ).split ( '.' )
        learner_name = kk[len ( kk ) - 1][:-2]


    for model, states in zip(models, labels):
        learner.add_data(states, model)

    if max_iter > 0:
        for bp in learner.belief_propagators_q:
            bp.set_max_iter(max_iter)
        for bp in learner.belief_propagators:
            bp.set_max_iter(max_iter)

    wr_obj = WeightRecord()
    new_weight = learner.learn(weights,optm , wr_obj.callback)
    weight_record = wr_obj.weight_record
    time_record = wr_obj.time_record
    Eval = Evaluator_latent(max_width, max_height)

    obj_list = []
    my_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    l = (weight_record.shape)[0]
    t = time_record[0][0]

    loader = ImageLoader(max_width, max_height)
    test_images, test_models, test_labels, test_names = loader.load_all_images_and_labels ( data_path + '/test', num_states,
                                                                        num_testing_images )
    div = l/100

    for k in range(100):
        i = k * div
        my_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(weight_record[i, :]))
        # print learner.subgrad_obj(weight_record[i, :])
        ave_error = Eval.evaluate_training_images(saved_path, images, models, labels, names, weight_record[i, :], num_states, num_training_images,
                                          inference_type, max_iter, inc='false', plot='false')
        train_accuracy_list.append ( ave_error )

        ave_error_test = Eval.evaluate_training_images ( saved_path, test_images, test_models, test_labels, test_names, weight_record[i, :],
                                                num_states, num_testing_images,
                                                inference_type, max_iter, inc='false', plot='false' )

        test_accuracy_list.append(ave_error_test)


    if i != l-1:
        i = l-1
        assert np.array_equal(weight_record[i, :] , new_weight)
        my_list.append ( time_record[i] - t )
        obj_list.append ( learner.subgrad_obj ( weight_record[i, :] ) )
        # print learner.subgrad_obj(weight_record[i, :])
        ave_error = Eval.evaluate_training_images ( saved_path, images, models, labels, names, weight_record[i, :],
                                                    num_states, num_training_images,
                                                    inference_type, max_iter, inc='false', plot='false' )
        train_accuracy_list.append ( ave_error )

        ave_error_test = Eval.evaluate_training_images ( saved_path, test_images, test_models, test_labels, test_names,
                                                         weight_record[i, :],
                                                         num_states, num_testing_images,
                                                         inference_type, max_iter, inc='false', plot='false' )

        test_accuracy_list.append ( ave_error_test )



    kk = str(inference_type).split ( '.' )
    inference_name = kk[len ( kk ) - 1][:-2]

    kk = str(optm).split ( ' ' )
    optimizer_name =kk[1]

    learner_dic = {}
    learner_dic['learner_name'] = learner_name
    learner_dic['time'] = my_list
    learner_dic['objective'] = obj_list
    learner_dic['training_error'] = train_accuracy_list
    learner_dic['final_weight'] = new_weight
    learner_dic['weights'] = weight_record
    learner_dic['inference_name'] = inference_name
    learner_dic['ave_error'] = ave_error
    learner_dic['testing_error'] = test_accuracy_list
    learner_dic['optimizer'] = optimizer_name

    x = np.linspace ( -1, 1, 21 )
    y = np.zeros ( 21 )
    for i in range ( len ( x ) ):
        mod_weight = new_weight + x[i] * noise
        y[i] = learner.subgrad_obj(mod_weight)

    learner_dic['neighbors'] = y

    return learner_dic


def make_latent(labels, latent_type, max_width, max_height,block_size):
    if latent_type == "every_four":
    # # every four pixel is unknown
        for k,v in labels[0].items():
            if (np.remainder(k[0],4) == 0) and (np.remainder(k[1],4) == 0):
                labels[0][(k[0],k[1])] = -100

    elif latent_type == "block_middle":
        # a block in the middle of image is unknown
        # block_size = [3, 3]
        x_position = max_width / 2
        y_position = max_height / 2

        for l in range(len(labels)):
            for i in range ( (x_position - block_size[0] / 2), (x_position + block_size[0] / 2) ):
                for j in range ( (y_position - block_size[1] / 2), (y_position + block_size[1] / 2) ):
                    labels[l][i, j] = -100

            for k in labels[l].keys ( ):
                if labels[l][k] == -100:
                    del labels[l][k]

    elif latent_type == "every_other_row":
        # # every other row is unknown
        for l in range ( len ( labels ) ):
            for k, v in labels[l].items ( ):
                if (np.remainder ( k[0], 2 ) == 0):
                    for jj in range ( max_width ):
                        labels[l][k[0], jj] = -100

            for lbl in labels[l].keys ( ):
                if labels[l][lbl] == -100:
                    del labels[l][lbl]
    elif latent_type == "right_half":
        # # right half of the image is latent
        for l in range ( len ( labels ) ):
            for k, v in labels[l].items ( ):
                if k[1] >= (max_height/2):
                    for jj in range(max_width):
                        labels[l][jj,k[1]] = -100

            for lbl in labels[l].keys ( ):
                if labels[l][lbl] == -100:
                    del labels[l][lbl]

    return labels


def test_errors():
    dataset = "background"
    d_unary = 65
    d_edge = 11

    if dataset == "horse":
        num_states = 2
        weights = pickle.load ( open ( 'initial_weights_horse.txt', 'r' ) )
    elif dataset == "background":
        num_states = 8
        weights = pickle.load ( open ( 'initial_weights.txt', 'r' ) )

    num_training_images = 1
    sizes = [200]
    lambdas = [0.00001, 0.000001,0.0001,0.001,0.01,0.1,1]
    for s in sizes:
        for l in lambdas:
            image_size = s
            max_iter = 0
            saved_path = ''
            path = os.path.abspath ( os.path.join ( os.path.dirname ( 'settings.py' ), os.path.pardir ) )

            learner_type = PairedDual
            inference_type = MatrixBeliefPropagator
            eval = Evaluator_latent ( image_size, image_size )
            data_path = path + '/data/' + dataset

            loader = ImageLoader ( image_size, image_size )

            images, models, labels, names = loader.load_all_images_and_labels (
                data_path + '/train', num_states, num_training_images )

            treu_labels = copy.deepcopy ( labels )

            # make latent variable

            for label in labels:
                # print "Number of labels: %d" % len(label)
                for x in range(image_size / 2):
                    for y in range(image_size / 2):
                        del label[(x, y)]
                # print "Number of labels after removing quadrant: %d" % len(label)


            learner = learner_type ( inference_type )
            learner.set_regularization ( 0.0, l)
            # np.round ( np.true_divide ( 1, image_size * image_size ) )

            eval_learner = learner_type ( inference_type )

            for model, states in zip ( models, labels ):
                learner.add_data ( states, model )

            errors = []
            objectives = []
            obj_list = []
            train_accuracy_list = []
            test_accuracy_list = []
            wr_obj = WeightRecord ( )
            new_weight = learner.learn ( weights, wr_obj.callback )


            i = 0
            models[i].set_weights ( new_weight )
            bp = MatrixBeliefPropagator ( models[i] )
            bp.infer ( display='final' )
            bp.load_beliefs ( )

            beliefs = np.zeros ( (images[i].height, images[i].width) )
            label_img = np.zeros ( (images[i].height, images[i].width) )
            errors = 0
            baseline = 0
            num_latent= 0
            for x in range ( images[i].width ):
                for y in range ( images[i].height ):
                    beliefs[y, x] = np.argmax(np.exp ( bp.var_beliefs[(x, y)] ))
                    label_img[y, x] = treu_labels[i][(x, y)]
                    if (x, y) in labels[i]:
                        if beliefs[y, x] != label_img[y, x]:
                            errors += 1
                    else:
                        num_latent += 1

            errors = np.true_divide ( errors, images[i].width * images[i].height - num_latent )

            print 'for image size '+ str(s)+' and regularizer '+ str(l)+ ' training error is ' + str(errors)



    # print errors
    # plt.subplot(131)
    # plt.imshow(images[i], interpolation="nearest")
    # plt.subplot(132)
    # plt.imshow(label_img, interpolation="nearest")
    # plt.subplot(133)
    # plt.imshow(beliefs, interpolation="nearest")
    # plt.show()



    # weight_record = wr_obj.weight_record
    # time_record = wr_obj.time_record
    # l = (weight_record.shape)[0]
    # div = l / 100
    #
    # for k in range ( 100 ):
    #     i = k * div
    #     obj_list.append ( learner.subgrad_obj ( weight_record[i, :] ) )
    #     ave_error = eval.evaluate_training_images ( saved_path, images, models, labels, names, weight_record[i, :],
    #                                                 num_states, num_training_images,
    #                                                 inference_type, max_iter, inc='false', plot='false' )
    #     train_accuracy_list.append ( ave_error )
    #
    # print obj_list[-1]
    # print ave_error
    # # print obj_list
    #
    # plt.clf ( )
    # plt.plot ( obj_list )
    # plt.show ( )
    #
    # plt.clf ( )
    # plt.plot ( train_accuracy_list )
    # plt.show ( )

def plot_optmizer():
    # # # ###############################*******Initialization*********###############################

    dataset = "background"

    d_unary = 65
    d_edge = 11
    # max_height = 240
    # max_width = 320
    max_height = 6
    max_width = 6
    num_training_images = 1
    num_testing_images = 1
    max_iter = 0
    inc = 'true'

    path = os.path.abspath ( os.path.join ( os.path.dirname ( 'settings.py' ), os.path.pardir ) )
    plot = 'true'
    data_path = path + '/data/' + dataset
    if not os.path.exists ( path + '/saved/' + dataset + '/' ):
        os.makedirs ( path + '/saved/' + dataset + '/' )
    saved_path = path + '/saved/' + dataset + '/'

    if dataset == "horse":
        num_states = 2
        initial_weights = pickle.load ( open ( 'initial_weights_horse.txt', 'r' ) )
    elif dataset == "background":
        num_states = 8
        initial_weights = pickle.load ( open ( 'initial_weights.txt', 'r' ) )

    noise = 0.1 * np.random.randn ( len ( initial_weights ) )
    # # # ###############################*******Load Images and make some labes latent*********###############################
    loader = ImageLoader ( max_width, max_height )
    images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/train', num_states,
                                                                        num_training_images )
    true_label = copy.deepcopy ( labels )
    latent_types = ["every_four", "block_middle", "every_other_row", "right_half"]
    block_size = [3, 3]
    labels = make_latent ( labels, latent_types[2], max_width, max_height, block_size )

    # # # ###############################*******for each learner examine different optimizer*********###############################
    learners = [Learner, EM, PairedDual, PrimalDual]
    # learners = [Learner]
    marginal_inferences = [MatrixBeliefPropagator]
    # optimizers = [adam, rms_prop, ada_grad]
    optimizers = [adam,ada_grad,rms_prop]
    # optimizers = [rms_prop]
    regularizers = [0, 0.0001]
    loss_aug = False
    method_list = []

    Eval = Evaluator_latent ( max_width, max_height )

    for infr in marginal_inferences:
        if str ( infr ) == 'ConvexBeliefPropagator_MAP':
            inference_type = ConvexBeliefPropagator
            MAP_Convex = True
            inferece_name = str ( infr )
        else:
            inference_type = infr
            MAP_Convex = False
            inferece_name = str ( inference_type ).split ( '.' )[-1][:-2]

        if not os.path.exists ( saved_path + inferece_name + '/' ):
            os.makedirs ( saved_path + '/' + inferece_name + '/' )

        saved_path_inference = saved_path + inferece_name
        for learner_type in learners:
            for optm in optimizers:
                learner_name = str ( learner_type ).split ( '.' )[-1][:-2]
                lnr_dic = learn_image ( saved_path, data_path, learner_type, inference_type, models, labels, num_states,
                                        names,
                                        images, num_training_images,
                                        max_iter, max_height, max_width, initial_weights, optm, noise ,loss_aug, regularizers,
                                        num_testing_images,
                                        MAP_Convex=MAP_Convex )

                method_list.append ( lnr_dic )

    # # # ###############################*******for each learner examine different optimizer*********###############################

    for lnr in learners:
        kk = str ( lnr ).split ( '.' )
        learner_name = kk[len ( kk ) - 1][:-2]
        lnr_method = []
        for mtd in method_list:
            if mtd['learner_name'] == learner_name:
                lnr_method.append ( mtd )

        plt.clf ( )
        for i in range ( 0, len ( lnr_method ) ):
            m_dic = lnr_method[i]
            # obj_time = np.arange ( 100 )
            obj_time = m_dic['time']
            obj = m_dic['objective']
            ttl = m_dic['optimizer']
            plt.plot ( obj_time, obj, '-', linewidth=2, label=ttl )

        plt.xlabel ( 'time(seconds)' )
        plt.ylabel ( 'objective' )
        plt.legend ( loc='upper right' )

        plt.title ( 'objective function trend for ' + m_dic['learner_name'] )
        plt.savefig ( saved_path_inference + '/objective_' + m_dic['learner_name'] )


    # # # # # ###############################*******plot objective to check local optima*********###############################

    for lnr in learners:
        kk = str ( lnr ).split ( '.' )
        learner_name = kk[len ( kk ) - 1][:-2]
        lnr_method = []
        for mtd in method_list:
            if mtd['learner_name'] == learner_name:
                lnr_method.append ( mtd )

        plt.clf ( )
        for j in range ( 0, len ( lnr_method ) ):
            m_dic = lnr_method[j]
            x = np.linspace ( -1, 1, 21 )
            y = m_dic['neighbors']

            ttl = m_dic['optimizer']
            plt.plot ( x, y, label=ttl )

        plt.ylabel ( 'objective' )
        plt.legend ( loc='upper right' )

        plt.title ( 'objective function trend for learner ' + m_dic['learner_name'] )
        plt.savefig ( saved_path_inference + '/' + m_dic['learner_name'] )



def main(arg):
    # test_errors()
    plot_optmizer()


if __name__ == "__main__":
    main(sys.argv[1:])