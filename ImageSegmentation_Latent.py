import numpy as np
from ImageLoader import ImageLoader
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from EM import EM
from PairedDual import PairedDual
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from Evaluator import Evaluator
import os
from opt import WeightRecord
import pickle
import copy
from MaxProductBeliefPropagator import MaxProductBeliefPropagator


def learn_image(learn_method, inference_type, models, labels, num_states, names, images, num_training_images, max_iter, max_height,
                max_width, weights):
    # learner = PairedDual(inference_type)
    learner = learn_method(inference_type )

    learner.set_regularization(0.0, 1.0)

    for model, states in zip(models, labels):
        learner.add_data(states, model)

    if max_iter > 0:
        for bp in learner.belief_propagators_q:
            bp.set_max_iter(max_iter)
        for bp in learner.belief_propagators:
            bp.set_max_iter(max_iter)

    wr_obj = WeightRecord()
    new_weight = learner.learn(weights, wr_obj.callback)
    weight_record = wr_obj.weight_record
    time_record = wr_obj.time_record

    Eval = Evaluator(max_height, max_width)

    obj_list = []
    my_list = []
    train_accuracy_list = []

    l = (weight_record.shape)[0]
    t = time_record[0][0]
    for i in range(l):
        my_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(weight_record[i, :]))
        ave_error = Eval.evaluate_training_images(images, models, labels, names, weight_record[i, :], num_states, num_training_images,
                                          inference_type, max_iter, inc='false', plot='false')
        train_accuracy_list.append(ave_error)


    learner_dic = {}
    learner_dic['method'] = learn_method
    learner_dic['time'] = my_list
    learner_dic['objective'] = obj_list
    learner_dic['training_error'] = train_accuracy_list
    learner_dic['final_weight'] = new_weight
    learner_dic['weights'] = weight_record

    return learner_dic


def main():
    mode = "latent"
    # mode = "observed"
    # dataset = "horse"
    dataset = "background"

    d_unary = 65
    d_edge = 11
    max_height = 10
    max_width = 10
    num_training_images = 1
    num_testing_images = 1
    max_iter = 0
    inc = 'true'
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
    plot = 'true'
    data_path = path + '/data/' + dataset
    file_path = path + '/saved_files'

    # inference_type = MatrixBeliefPropagator
    # inference_type = MatrixTRBeliefPropagator
    inference_type = MaxProductBeliefPropagator

    if dataset == "horse":
        num_states = 2
    elif dataset == "background":
        num_states = 8

    loader = ImageLoader(max_height, max_width)

    images, models, labels, names = loader.load_all_images_and_labels(data_path + '/train', num_states, num_training_images)

    true_label = []
    true_label = copy.deepcopy(labels)


    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)
    Eval = Evaluator ( max_height, max_width )

    if mode == "latent":
        # # every four pixel is unknown
        # for k,v in labels[0].items():
        #     if (np.remainder(k[0],4) == 0) and (np.remainder(k[1],4) == 0):
        #         labels[0][(k[0],k[1])] = -100

        # a block in the middle of image is unknown
        block_size = [8, 8]
        x_position = max_width / 2
        y_position = max_height / 2

        for i in range ( (x_position - block_size[0] / 2), (x_position + block_size[0] / 2) ):
            for j in range ( (y_position - block_size[1] / 2), (y_position + block_size[1] / 2) ):
                labels[0][i, j] = -100

        for k in labels[0].keys ( ):
            if labels[0][k] == -100:
                del labels[0][k]

        method_list = []
        # ########################## subgradient Objective ###########################
        sub_dic = learn_image(Learner, inference_type, models, labels, num_states , names, images, num_training_images, max_iter,
                                max_height, max_width, weights)
        method_list.append ( sub_dic )
        f = open ( file_path + '/subgrad_time.txt', 'w')
        pickle.dump ( sub_dic['time'], f )
        f.close ( )

        f = open ( file_path + '/subgrad_step_weight.txt', 'w' )
        pickle.dump ( sub_dic['weights'], f )
        f.close ( )

        # ########################## EM Objective ###########################
        EM_dic = learn_image ( EM, inference_type, models, labels, num_states ,names, images, num_training_images, max_iter,
                               max_height,
                               max_width, weights )
        method_list.append ( EM_dic )
        f = open ( file_path + '/EM_time.txt', 'w' )
        pickle.dump ( EM_dic['time'], f )
        f.close ( )

        f = open ( file_path + '/EM_step_weight.txt', 'w' )
        pickle.dump ( EM_dic['weights'], f )
        f.close ( )

        # ########################## pairedDual Objective ###########################
        paired_dic = learn_image ( PairedDual, inference_type, models, labels, num_states,names, images, num_training_images,
                                   max_iter,
                                   max_height, max_width, weights )
        method_list.append ( paired_dic )

        f = open ( file_path + '/pairedDual_time.txt', 'w' )
        pickle.dump ( paired_dic['time'], f )
        f.close ( )

        f = open ( file_path + '/pairedDual_step_weight.txt', 'w' )
        pickle.dump ( paired_dic['weights'], f )
        f.close ( )

        Eval.evaluate_objective ( method_list, file_path )
        Eval.evaluate_training_accuracy ( method_list, file_path )

    elif mode == "observed":
        method_list = []
        learner_dic = learn_image(Learner, inference_type, models, labels, num_states ,names, images, num_training_images, max_iter,
                                max_height, max_width, weights )
        method_list.append ( learner_dic )

    ########################## plot images ###########################
    weights_dic = {}
    for i in range ( 0, len ( method_list ) ):
        print '-------------------'
        new_weight = method_list[i]['final_weight']
        weights_dic[method_list[i]['method']] = new_weight
        print new_weight

        if num_training_images > 0:
            if inc == True:
                train_errors, train_total_inconsistency = Eval.evaluate_training_images(images, models, labels, names, new_weight, num_states, num_training_images, inference_type, max_iter, inc, plot = 'false')
            else:
                train_errors = Eval.evaluate_training_images(images, models, labels, names, new_weight, num_states, num_training_images, inference_type, max_iter, inc, plot = 'false')
            print ("Average Train Error rate: %f" % train_errors + " for method " +  str(method_list[i]['method']))


        if num_testing_images > 0:
            if inc == True:
                test_errors, test_total_inconsistency = Eval.evaluate_testing_images(data_path + '/test', new_weight, num_states, num_testing_images, inference_type, max_iter, inc, plot = 'false')
            else:
                test_errors = Eval.evaluate_testing_images(data_path + '/test', new_weight, num_states, num_testing_images, inference_type, max_iter, inc, plot = 'false')
            print ("Average Test Error rate: %f" % test_errors + " for method " + str(method_list[i]['method']))



    Eval.plot_images( images, models, true_label, names, weights_dic, num_states, num_training_images,
                           inference_type, max_iter )

    images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/test', num_states,
                                                                        num_testing_images )
    Eval.plot_images( images, models, labels, names, weights_dic, num_states, num_testing_images,
                               inference_type, max_iter )



if __name__ == "__main__":
    main()
