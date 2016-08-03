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


def learn_image(learn_method, inference_type, models, labels, names, images, num_training_images, max_iter, max_height, max_width, weights):

    learner = learn_method(inference_type)
    # learner = PairedDual(inference_type)

    learner.set_regularization(0.0, 1.0)

    for model, states in zip(models, labels):
     learner.add_data(states, model)

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
        obj_list.append(learner.subgrad_obj(weight_record[i,:]))
        train_accuracy_list.append(Eval.evaluate_training_images(images, models, labels, names, weight_record[i,:], 2, num_training_images, inference_type, max_iter, inc = 'false', plot = 'false'))

    learner_dic = {}
    learner_dic['method'] = learn_method
    learner_dic['time'] = my_list
    learner_dic['objective'] = obj_list
    learner_dic['training_accuracy'] = train_accuracy_list
    learner_dic['final_weight'] = new_weight

    return learner_dic

def main():

    d_unary = 65
    num_states = 2
    d_edge = 11
    max_height = 30
    max_width = 30
    num_training_images = 1
    num_testing_images = 1
    max_iter = 5
    inc = 'true'
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
    plot = 'true'

    # inference_type = MatrixBeliefPropagator
    inference_type = MatrixTRBeliefPropagator

    loader = ImageLoader(max_height, max_width)

    images, models, labels, names = loader.load_all_images_and_labels(path+'/test/train', 2, num_training_images)

    # # every four pixel is unknown
    # for k,v in labels[0].items():
    #     if (np.remainder(k[0],4) == 0) and (np.remainder(k[1],4) == 0):
    #         labels[0][(k[0],k[1])] = -100


    # a block in the middle of image is unknown
    block_size = [8,8]
    x_position = max_width/2
    y_position = max_height/2

    for i in range((x_position - block_size[0]/2),(x_position + block_size[0]/2)):
        for j in range((y_position - block_size[1]/2),(y_position + block_size[1]/2)):
            labels[0][i,j] = -100



    for k in labels[0].keys():
        if labels[0][k] == -100:
            del labels[0][k]


    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)
    method_list = []
    # ########################## subgradient Objective ###########################
    sub_dic = learn_image(Learner, inference_type, models, labels, names, images, num_training_images, max_iter, max_height, max_width, weights)
    method_list.append(sub_dic)
    # ########################## EM Objective ###########################
    EM_dic = learn_image(EM, inference_type, models, labels, names, images, num_training_images, max_iter, max_height, max_width, weights)
    method_list.append(EM_dic)
    # ########################## pairedDual Objective ###########################
    paired_dic = learn_image(PairedDual, inference_type, models, labels, names, images, num_training_images, max_iter, max_height, max_width, weights)
    method_list.append(paired_dic)


    Eval = Evaluator(max_height, max_width)
    Eval.evaluate_objective(method_list)
    Eval.evaluate_training_accuracy(method_list)


    ########################## plot images ###########################
    for i in range(0,len(method_list)):
        learner_dic = method_list[i]
        new_weights = learner_dic['final_weight']

        unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
        pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states ** 2))
        print("Unary weights:\n" + repr(unary_mat))
        print("Pairwise weights:\n" + repr(pair_mat))

        if num_training_images > 0:
            print("Training:")
            if inc == "true":
                train_errors, train_total_inconsistency = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2, num_training_images, inference_type, max_iter, inc, plot)
            else:
                train_errors = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2, num_training_images, inference_type, max_iter, inc, plot)
            print ("Average Train Error rate: %f" % train_errors)

        if num_testing_images > 0:
            print("Test:")
            if inc == "true":
                test_errors, test_total_inconsistency = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images, inference_type, max_iter, inc, plot)
            else:
                test_errors = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images, inference_type, max_iter, inc, plot)
            print ("Average Test Error rate: %f" % test_errors)





if __name__ == "__main__":
    main()