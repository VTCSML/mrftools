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


def main(arg):
# def main():

    # <editor-fold desc="Initialization">

    dataset = "background"
    d_unary = 65
    d_edge = 11
    # max_height = 240
    # max_width = 320
    max_height = 6
    max_width = 6
    num_training_images = 2
    num_testing_images = 2
    max_iter = 0
    inc = 'true'
    path = os.path.abspath ( os.path.join ( os.path.dirname ( 'settings.py' ), os.path.pardir ) )
    plot = 'true'
    data_path = path + '/data/' + dataset
    if not os.path.exists ( path + '/saved/' ):
        os.makedirs ( path + '/saved/' )
    saved_path = path + '/saved/'



    if dataset == "horse":
        num_states = 2
    elif dataset == "background":
        num_states = 8

    loader = ImageLoader(max_width, max_height)
    images, models, labels, names = loader.load_all_images_and_labels(data_path + '/train', num_states, num_training_images)
    true_label = copy.deepcopy(labels)
    Eval = Evaluator_latent ( max_width, max_height )



    # ###############################*******Load files and plot****************###############################
    inferences = ['ConvexBeliefPropagator_MAP',MatrixTRBeliefPropagator,ConvexBeliefPropagator,MatrixBeliefPropagator, MaxProductBeliefPropagator]
    for inference_type in inferences:
        if inference_type == 'ConvexBeliefPropagator_MAP':
            inference_type = ConvexBeliefPropagator
            inference_name = 'ConvexBeliefPropagator_MAP'
        else:
            kk = str ( inference_type ).split ( '.' )
            inference_name = kk[len ( kk ) - 1][:-2]

        saved_path_inference = saved_path + '/'+ dataset +'/' +inference_name
        method_list = list ( [] )
        result_file = open ( saved_path_inference + '/' +'result.csv', 'w' )
        result_file.write('******************TRAINING*************************')
        result_file.write ( "\n" )
        for file in os.listdir ( saved_path_inference ):
            if file.endswith ( ".txt" ):
                read_dic = pickle.load ( open ( saved_path_inference + '/' + file ) )
                learner_dic = {}
                learner_dic['learner_name'] = read_dic['learner_name']
                learner_dic['time'] = read_dic['time']
                learner_dic['objective'] = read_dic['objective']
                learner_dic['training_error'] = read_dic['training_error']
                learner_dic['final_weight'] = read_dic['final_weight']
                learner_dic['weights'] = read_dic['weights']
                learner_dic['inference_name'] = read_dic['inference_name']
                learner_dic['ave_error'] =  read_dic['ave_error']
                learner_dic['testing_error'] = read_dic['testing_error']
                # print '-----------------------'
                # print learner_dic['learner_name']
                # print learner_dic['objective']
                # print learner_dic['time']

                method_list.append ( learner_dic )
                result_file.write (
                    "For learner " + str(learner_dic['learner_name']) + " Average Train Error rate: " + str(learner_dic["ave_error"]  ))
                result_file.write ( "\n" )
        loss_method_list = []
        original_method_list = []
        for m in method_list:
            if 'Loss' in m['learner_name']:
                loss_method_list.append(m)
            else:
                original_method_list.append(m)

        Eval.evaluate_training_accuracy ( method_list, saved_path_inference ,'train')
        Eval.evaluate_training_accuracy ( method_list, saved_path_inference, 'test' )

        if len(loss_method_list) != 0:
            Eval.evaluate_objective ( loss_method_list, saved_path_inference )

        if len(original_method_list) != 0:
            Eval.evaluate_objective ( original_method_list, saved_path_inference )
        weights_dic = {}
        if num_testing_images > 0:
            result_file.write ( '******************TESTING*************************' )
            result_file.write("\n")
        for i in range(0,len(method_list)):
            new_weight = method_list[i]['final_weight']
            weights_dic[method_list[i]['learner_name']] = new_weight
            if num_testing_images > 0 :
                test_errors = Eval.evaluate_testing_images('', data_path + '/test', new_weight, num_states, num_testing_images, inference_type, max_iter, inc, plot = 'false')
                result_file.write (
                "For learner " + str ( method_list[i]['learner_name'] ) + " Average Test Error rate: " + str (
                    test_errors ))
                result_file.write ( "\n" )

        # ##### plot training images #########

        images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/train', num_states,
                                                                            num_training_images )

        # # every other row is latent
        # for l in range ( len ( labels ) ):
        #     for k, v in labels[l].items ( ):
        #         if (np.remainder ( k[0], 2 ) == 0):
        #             for jj in range ( max_width ):
        #                 labels[l][k[0], jj] = -100
        #
            # for lbl in labels[l].keys ( ):
            #     if labels[l][lbl] == -100:
            #         del labels[l][lbl]


        training_names = []
        for nm in names:
            training_names.append('training_' + nm)
        Eval.plot_images ( saved_path_inference + '/ ' , images, models, labels, training_names, weights_dic, num_states, num_training_images,
                                        inference_type, max_iter )
        # ##### plot testing images #########
        images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/test', num_states,
                                                                            num_testing_images )
        test_names = []
        for nm in names:
            test_names.append('test_' + nm)
        Eval.plot_images ( saved_path_inference + '/ ', images, models, labels, test_names, weights_dic,
                           num_states, num_training_images,
                           inference_type, max_iter )





if __name__ == "__main__":
    main(sys.argv[1:])
    # main()