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

def learn_image(saved_path, data_path, learn_method, inference_type, models, labels, num_states, names, images, num_training_images, max_iter, max_height,
                max_width, weights,loss_augmented = False, regularizer = [0, 1], num_testing_images=1, MAP_Convex = False):

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

    wr_obj = WeightRecord()
    new_weight = learner.learn(weights, wr_obj.callback)
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

    for k in range(10):
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



    # for i in range(l):
    #     my_list.append(time_record[i] - t)
    #     obj_list.append(learner.subgrad_obj(weight_record[i, :]))
    #     ave_error = Eval.evaluate_training_images(saved_path, images, models, labels, names, weight_record[i, :], num_states, num_training_images,
    #                                       inference_type, max_iter, inc='false', plot='false')
    #     train_accuracy_list.append(ave_error)


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
    learner_dic['testing_error'] = test_accuracy_list

    return learner_dic



def main(arg):
# def main():

    dataset = "background"
    d_unary = 65
    d_edge = 11
    # max_height = 240
    # max_width = 320
    max_height = 6
    max_width = 6
    num_training_images = 1
    num_testing_images = 2
    # num_training_images = 6
    # num_testing_images = 2
    max_iter = 0
    inc = 'true'
    path = os.path.abspath ( os.path.join ( os.path.dirname ( 'settings.py' ), os.path.pardir ) )
    plot = 'true'
    data_path = path + '/data/' + dataset
    if not os.path.exists ( path + '/saved/'+dataset+'/' ):
        os.makedirs ( path + '/saved/' +dataset+'/')
    saved_path = path + '/saved/'+dataset+'/'



    if dataset == "horse":
        num_states = 2
        weights = pickle.load ( open ( 'initial_weights_horse.txt','r' ) )
    elif dataset == "background":
        num_states = 8
        weights = pickle.load ( open ( 'initial_weights.txt', 'r' ) )
        # weights = np.zeros ( d_unary * num_states + d_edge * num_states ** 2 )
    loader = ImageLoader(max_width, max_height)
    images, models, labels, names = loader.load_all_images_and_labels(data_path + '/train', num_states, num_training_images)
    true_label = copy.deepcopy(labels)


    # # every four pixel is unknown
    # for k,v in labels[0].items():
    #     if (np.remainder(k[0],4) == 0) and (np.remainder(k[1],4) == 0):
    #         labels[0][(k[0],k[1])] = -100

    ## a block in the middle of image is unknown
    # block_size = [3, 3]
    # x_position = max_width / 2
    # y_position = max_height / 2
    #
    # for l in range(len(labels)):
    #     for i in range ( (x_position - block_size[0] / 2), (x_position + block_size[0] / 2) ):
    #         for j in range ( (y_position - block_size[1] / 2), (y_position + block_size[1] / 2) ):
    #             labels[l][i, j] = -100
    #
    #     for k in labels[l].keys ( ):
    #         if labels[l][k] == -100:
    #             del labels[l][k]


    # every other row is latent
    for l in range ( len ( labels ) ):
        for k, v in labels[l].items ( ):
            if (np.remainder ( k[0], 2 ) == 0):
                for jj in range ( max_width ):
                    labels[l][k[0], jj] = -100

        for lbl in labels[l].keys ( ):
            if labels[l][lbl] == -100:
                del labels[l][lbl]

    # # right half of the image is latent
    for l in range ( len ( labels ) ):
        for k, v in labels[l].items ( ):
            if k[1] >= (max_height/2):
                for jj in range(max_width):
                    labels[l][jj,k[1]] = -100

        for lbl in labels[l].keys ( ):
            if labels[l][lbl] == -100:
                del labels[l][lbl]


# # ## ********************************************************
    inferences = [MatrixBeliefPropagator]
#     learners = [PrimalDual]
#     inferences = [ConvexBeliefPropagator,MatrixTRBeliefPropagator, MatrixBeliefPropagator]
    # learners = [EM]
    # learners = [EM,Learner,PairedDual,PrimalDual]
    # inferences = [MatrixBeliefPropagator]
    learners = [Learner,PairedDual]
    regularizers = [0,0.1]
    loss_aug = False
    MAP_Convex = False
    for infr in inferences:
        for learner_type in learners:
            inference_type = infr
            inferece_name = str ( inference_type ).split ( '.' )[-1][:-2]
            learner_name = str(learner_type).split('.')[-1][:-2]


            lnr_dic = learn_image ( saved_path,data_path, learner_type, inference_type, models, labels, num_states, names,
                                    images, num_training_images,
                                    max_iter, max_height, max_width, weights, loss_aug, regularizers, num_testing_images, MAP_Convex=MAP_Convex)
#
            if not os.path.exists ( saved_path +  inferece_name + '/'):
                os.makedirs ( saved_path + '/'+ inferece_name + '/' )
            f = open ( saved_path  +'/' +inferece_name + '/' + str(regularizers)+ '_' +inferece_name+ '_' + learner_name + '_'+ str(loss_aug) + '.txt', 'w' )
            pickle.dump ( lnr_dic, f )
            f.close ( )

    # #############plot

    Eval = Evaluator_latent ( max_width, max_height )
    for inference_type in inferences:
        if inference_type == 'ConvexBeliefPropagator_MAP':
            inference_type = ConvexBeliefPropagator
            inference_name = 'ConvexBeliefPropagator_MAP'
        else:
            kk = str ( inference_type ).split ( '.' )
            inference_name = kk[len ( kk ) - 1][:-2]


        saved_path_inference = saved_path + '/' + inference_name
        method_list = list ( [] )
        result_file = open ( saved_path_inference + '/' + 'result.csv', 'w' )
        result_file.write ( '******************TRAINING*************************' )
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
                learner_dic['ave_error'] = read_dic['ave_error']
                learner_dic['testing_error'] = read_dic['testing_error']
                # print '-----------------------'
                # print learner_dic['learner_name']
                # print learner_dic['objective']
                # print learner_dic['time']

                method_list.append ( learner_dic )
                result_file.write (
                    "For learner " + str ( learner_dic['learner_name'] ) + " Average Train Error rate: " + str (
                        learner_dic["ave_error"] ) )
                result_file.write ( "\n" )
        loss_method_list = []
        original_method_list = []
        for m in method_list:
            if 'Loss' in m['learner_name']:
                loss_method_list.append ( m )
            else:
                original_method_list.append ( m )

        Eval.evaluate_training_accuracy ( method_list, saved_path_inference, 'train' )
        Eval.evaluate_training_accuracy ( method_list, saved_path_inference, 'test' )

        if len ( loss_method_list ) != 0:
            Eval.evaluate_objective ( loss_method_list, saved_path_inference )

        if len ( original_method_list ) != 0:
            Eval.evaluate_objective ( original_method_list, saved_path_inference )
        weights_dic = {}
        if num_testing_images > 0:
            result_file.write ( '******************TESTING*************************' )
            result_file.write ( "\n" )
        for i in range ( 0, len ( method_list ) ):
            new_weight = method_list[i]['final_weight']
            weights_dic[method_list[i]['learner_name']] = new_weight
            if num_testing_images > 0:
                test_errors = Eval.evaluate_testing_images ( '', data_path + '/test', new_weight, num_states,
                                                             num_testing_images, inference_type, max_iter, inc,
                                                             plot='false' )
                result_file.write (
                    "For learner " + str ( method_list[i]['learner_name'] ) + " Average Test Error rate: " + str (
                        test_errors ) )
                result_file.write ( "\n" )

        # ##### plot training images #########

        images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/train', num_states,
                                                                            num_training_images )

        training_names = []
        for nm in names:
            training_names.append ( 'training_' + nm )
        Eval.plot_images ( saved_path_inference + '/ ', images, models, labels, training_names, weights_dic,
                           num_states, num_training_images,
                           inference_type, max_iter )
        # ##### plot testing images #########
        images, models, labels, names = loader.load_all_images_and_labels ( data_path + '/test', num_states,
                                                                            num_testing_images )
        test_names = []
        for nm in names:
            test_names.append ( 'test_' + nm )
        Eval.plot_images ( saved_path_inference + '/ ', images, models, labels, test_names, weights_dic,
                           num_states, num_training_images,
                           inference_type, max_iter )










        # ########## MAP
        # #     # inferences = ['ConvexBeliefPropagator_MAP']
        # #     # learners = [PrimalDual]
        #     inferences = [MaxProductBeliefPropagator, 'ConvexBeliefPropagator_MAP']
        #     learners = [EM, Learner,PairedDual,PrimalDual]
        # #     # learners = [EM]
        #     regularizers = [0,0.1]
        #     loss_aug = True
        #     for infr in inferences:
        #         for learner_type in learners:
        #             if str(infr) == 'ConvexBeliefPropagator_MAP':
        #                 inference_type = ConvexBeliefPropagator
        #                 MAP_Convex = True
        #                 inferece_name = str(infr)
        #             else:
        #                 inference_type = infr
        #                 MAP_Convex = False
        #                 inferece_name = str ( inference_type ).split ( '.' )[-1][:-2]
        #
        #             learner_name = str(learner_type).split('.')[-1][:-2]
        #
        #
        #             lnr_dic = learn_image ( saved_path,data_path, learner_type, inference_type, models, labels, num_states, names,
        #                                     images, num_training_images,
        #                                     max_iter, max_height, max_width, weights, loss_aug, regularizers, num_testing_images, MAP_Convex=MAP_Convex)
        #
        #             if not os.path.exists ( saved_path +  inferece_name + '/'):
        #                 os.makedirs ( saved_path + '/'+ inferece_name + '/' )
        #             f = open ( saved_path  +'/' +inferece_name + '/' + str(regularizers)+ '_' +inferece_name+ '_' + learner_name + '_'+ str(loss_aug) + '.txt', 'w' )
        #             pickle.dump ( lnr_dic, f )
        #             f.close ( )
        # # # #     # # #
        # #     # # # ***************************************************
        #
        #












    # # ###############################*******train and save files*********###############################
    # infr = arg[0]
    # learner_type = eval(arg[1])
    # loss_aug = eval(arg[2])
    # regularizers = np.array(arg[3].split(','))
    # regularizers = regularizers.astype ( np.float )
    # # weights = np.zeros( d_unary * num_states + d_edge * num_states ** 2)
    # # weights = np.random.rand ( d_unary * num_states + d_edge * num_states ** 2 ) * 0.001
    # # f = open ( 'initial_weights_horse.txt','w' )
    # # pickle.dump(weights,f)
    # # f.close()
    #
    # if str(infr) == 'ConvexBeliefPropagator_MAP':
    #     inference_type = ConvexBeliefPropagator
    #     MAP_Convex = True
    #     inferece_name = str(infr)
    # else:
    #     inference_type = eval ( infr )
    #     MAP_Convex = False
    #     inferece_name = str ( inference_type ).split ( '.' )[-1][:-2]
    #
    #
    #
    # learner_name = str(learner_type).split('.')[-1][:-2]
    #
    #
    # if not os.path.isfile(saved_path + '/'+ inferece_name + '/' +arg[3]+ '_' +inferece_name+ '_' + learner_name + '_'+ str(loss_aug) + '.txt'):
    #     lnr_dic = learn_image ( saved_path,data_path, learner_type, inference_type, models, labels, num_states, names,
    #                             images, num_training_images,
    #                             max_iter, max_height, max_width, weights, loss_aug, regularizers, num_testing_images,MAP_Convex=MAP_Convex)
    #
    #
    #     if not os.path.exists ( saved_path + inferece_name + '/' ):
    #         os.makedirs ( saved_path + '/' + inferece_name + '/' )
    #
    #     f = open ( saved_path + '/'+ inferece_name + '/' +arg[3]+ '_' +inferece_name+ '_' + learner_name + '_'+ str(loss_aug) + '.txt', 'w' )
    #     pickle.dump ( lnr_dic, f )
    #     f.close ( )





if __name__ == "__main__":
    main(sys.argv[1:])
    # main()