import numpy as np
from ImageLoader import ImageLoader
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from EM import EM
from PairedDual import PairedDual
from PrimalDual import PrimalDual
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from Evaluator import Evaluator
import os
from opt import WeightRecord
import pickle
import copy
from MaxProductBeliefPropagator import MaxProductBeliefPropagator
from MaxProductLinearProgramming import MaxProductLinearProgramming


def learn_image(saved_path, learn_method, inference_type, models, labels, num_states, names, images, num_training_images, max_iter, max_height,
                max_width, weights,loss_augmented = False):

    learner = learn_method(inference_type )

    learner.set_regularization(0.0, 1.0)
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
    new_weight = learner.learn(weights, wr_obj.callback)
    weight_record = wr_obj.weight_record
    time_record = wr_obj.time_record

    Eval = Evaluator(max_width, max_height)

    obj_list = []
    my_list = []
    train_accuracy_list = []



    l = (weight_record.shape)[0]
    t = time_record[0][0]
    for i in range(l):
        my_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(weight_record[i, :]))
        ave_error = Eval.evaluate_training_images(saved_path, images, models, labels, names, weight_record[i, :], num_states, num_training_images,
                                          inference_type, max_iter, inc='false', plot='false')
        train_accuracy_list.append(ave_error)

    assert np.array_equal ( new_weight, weight_record[-1]), "not equal weights"

    kk = str(inference_type).split ( '.' )
    inference_name = kk[len ( kk ) - 1][:-2]

    learner_dic = {}
    learner_dic['learner_name'] = learner_name
    learner_dic['time'] = my_list
    learner_dic['objective'] = obj_list
    learner_dic['training_error'] = train_accuracy_list
    learner_dic['final_weight'] = new_weight
    learner_dic['weights'] = weight_record
    learner_dic['inference_name'] = inference_name
    learner_dic['ave_error'] = ave_error

    return learner_dic


def main():

    # <editor-fold desc="Initialization">

    # dataset = "horse"
    dataset = "background"
    d_unary = 65
    d_edge = 11
    max_height = 240
    max_width = 320
    # max_height = 6
    # max_width = 6
    num_training_images = 31
    num_testing_images = 11
    # num_training_images = 10
    # num_testing_images = 2
    max_iter = 0
    inc = 'true'
    path = os.path.abspath ( os.path.join ( os.path.dirname ( 'settings.py' ), os.path.pardir ) )
    plot = 'true'
    data_path = path + '/data/' + dataset
    if not os.path.exists ( path + '/saved/' ):
        os.makedirs ( path + '/saved/' )
    saved_path = path + '/saved/'
    inferences = [MatrixBeliefPropagator, MatrixTRBeliefPropagator, MaxProductLinearProgramming, MaxProductBeliefPropagator]
    # inferences= [MaxProductBeliefPropagator]
    learners = [Learner, EM, PairedDual, PrimalDual]
    loss_augmented = [True, False]
    if dataset == "horse":
        num_states = 2
    elif dataset == "background":
        num_states = 8
    weights = np.zeros ( d_unary * num_states + d_edge * num_states ** 2 )

    # </editor-fold>

    # <editor-fold desc="Load Images and set up Evaluator">
    loader = ImageLoader(max_width, max_height)
    images, models, labels, names = loader.load_all_images_and_labels(data_path + '/train', num_states, num_training_images)
    true_label = copy.deepcopy(labels)
    Eval = Evaluator ( max_width, max_height )
    # </editor-fold>

    # <editor-fold desc="Set up hidden pixels">
    # # every four pixel is unknown
    # for k,v in labels[0].items():
    #     if (np.remainder(k[0],4) == 0) and (np.remainder(k[1],4) == 0):
    #         labels[0][(k[0],k[1])] = -100

    # a block in the middle of image is unknown
    block_size = [2, 2]
    x_position = max_width / 2
    y_position = max_height / 2

    for i in range ( (x_position - block_size[0] / 2), (x_position + block_size[0] / 2) ):
        for j in range ( (y_position - block_size[1] / 2), (y_position + block_size[1] / 2) ):
            labels[0][i, j] = -100

    for k in labels[0].keys ( ):
        if labels[0][k] == -100:
            del labels[0][k]
    # </editor-fold>

    # <editor-fold desc="train with different learners and inferences and plot the results">
    for inference_type in inferences:
        kk = str ( inference_type ).split ( '.' )
        inference_name = kk[len ( kk ) - 1][:-2]
        if not os.path.exists ( saved_path + inference_name ):
            os.makedirs ( saved_path + inference_name )
        saved_path_inference = saved_path + inference_name
        method_list = list ( [] )
        result_file = open ( saved_path_inference + '/' +'result.txt', 'w' )
        result_file.write('******************TRAINING*************************')
        result_file.write ( "\n" )
        for loss_aug in loss_augmented:
            for learner_type in learners:
                if inference_type == MaxProductBeliefPropagator:
                    if learner_type in [Learner, EM]:
                        continue;
                print loss_aug, inference_type
                lnr_dic = learn_image ( saved_path_inference , learner_type, inference_type, models, labels, num_states, names, images, num_training_images,
                              max_iter,
                              max_height, max_width, weights, loss_aug )

                method_list.append ( lnr_dic )

                f = open ( saved_path_inference + '/ '+  lnr_dic['learner_name'] +'_time.txt', 'w' )
                pickle.dump ( lnr_dic['time'], f )
                f.close ( )

                f = open ( saved_path_inference + '/ '+ lnr_dic['learner_name'] + '_step_weight.txt', 'w' )
                pickle.dump ( lnr_dic['weights'], f )
                f.close ( )

                result_file.write (
                    "For learner " + str(lnr_dic['learner_name']) + "Average Train Error rate: " + str(lnr_dic["ave_error"]  ))
                result_file.write ( "\n" )

        Eval.evaluate_training_accuracy ( method_list, saved_path_inference )
        Eval.evaluate_objective ( method_list, saved_path_inference )
        if num_testing_images > 0:
            result_file.write ( '******************TESTING*************************' )
            result_file.write("\n")

        weights_dic = {}
        for i in range(0,len(method_list)):
            new_weight = method_list[i]['final_weight']
            weights_dic[method_list[i]['learner_name']] = new_weight
            if num_testing_images > 0 :
                test_errors = Eval.evaluate_testing_images('', data_path + '/test', new_weight, num_states, num_testing_images, inference_type, max_iter, inc, plot = 'false')
                result_file.write (
                "For learner " + str ( method_list[i]['learner_name'] ) + "Average Test Error rate: " + str (
                    test_errors ))
                result_file.write ( "\n" )

        # ##### plot training images #########
        training_names = []
        for nm in names:
            training_names.append('training_' + nm)
        Eval.plot_images ( saved_path_inference + '/ ' , images, models, true_label, training_names, weights_dic, num_states, num_training_images,
                                        inference_type, max_iter )

        images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/test', num_states,
                                                                            num_testing_images )
        test_names = []
        for nm in names:
            test_names.append('test_' + nm)
        Eval.plot_images ( saved_path_inference + '/ ', images, models, true_label, test_names, weights_dic,
                           num_states, num_training_images,
                           inference_type, max_iter )

    # </editor-fold>



if __name__ == "__main__":
    main()
