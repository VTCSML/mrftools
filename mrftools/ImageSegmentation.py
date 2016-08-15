import numpy as np
from ImageLoader import ImageLoader
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from Evaluator import Evaluator
import os
import time
from PairedDual import PairedDual
import xlwt
from xlwt import Workbook

def main():

    num_states = 2
    d_unary = 65
    d_edge = 11
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
    max_height = 10
    max_width = 10
    num_training_images = 2
    num_testing_images = 2
    # inference_type = MatrixBeliefPropagator
    # inference_type = MatrixTRBeliefPropagator
    inference_type = ConvexBeliefPropagator
    inference_type_name = 'ConvexBP'
    max_iter = 5
    inc = True
    plot = False
    initialization_flag = True
    objective_type = 'primal'
    l2_regularization = 1.0

    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_RIGHT
    style = xlwt.XFStyle()
    style.alignment = alignment
    style.num_format_str = '#,##0.0000'

    loader = ImageLoader(max_height, max_width)
    images, models, labels, names = loader.load_all_images_and_labels(path+'/test/train', 2, num_training_images)

    configuration = (max_iter, inference_type_name, inference_type, objective_type, l2_regularization)
    image_segmentation(configuration, images, models, labels, names, num_states, d_unary, d_edge, path, max_height, max_width, num_training_images, num_testing_images,
                       inc, plot, initialization_flag, style)


def image_segmentation(configuration, images, models, labels, names, num_states, d_unary, d_edge, path, max_height, max_width, num_training_images, num_testing_images,
                       inc, plot, initialization_flag, style):

    wb = Workbook()
    sheet1 = wb.add_sheet('Results')

    sheet1.write(0,0,'Max_iter', style)
    sheet1.write(0,1,'TRBP or BP', style)
    sheet1.write(0,2,'Primal or Dual', style)
    sheet1.write(0,3, 'L2 regularization', style)
    sheet1.write(0,4,'training error', style)
    sheet1.write(0,5,'training incon', style)
    sheet1.write(0,6,'testing error', style)
    sheet1.write(0,7,'testing incon', style)
    sheet1.write(0,8,'training time', style)

    sheet1.col(0).width = 3000
    sheet1.col(1).width = 3000
    sheet1.col(2).width = 3000
    sheet1.col(3).width = 3000
    sheet1.col(4).width = 3000
    sheet1.col(5).width = 3000
    sheet1.col(6).width = 3000
    sheet1.col(7).width = 3000
    sheet1.col(7).width = 3000

    if len(configuration) == 5:
        max_iter = configuration[0]
        inference_type_name = configuration[1]
        inference_type = configuration[2]
        objective_type = configuration[3]
        l2_regularization = configuration[4]

    start = time.time()

    sheet1.write(1, 0, max_iter)
    sheet1.write(1, 1, inference_type_name, style)
    sheet1.write(1, 2, objective_type, style)
    sheet1.write(1, 3, l2_regularization, style)

    if objective_type is 'dual':
        learner = PairedDual(inference_type, max_iter)
    else:
        learner = Learner(inference_type)

    learner._set_initialization_flag(initialization_flag)

    learner.set_regularization(0.0, l2_regularization)

    for model, states in zip(models, labels):
     learner.add_data(states, model)

    for bp in learner.belief_propagators_q:
     bp.set_max_iter(max_iter)
    for bp in learner.belief_propagators:
     bp.set_max_iter(max_iter)

    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

    new_weights = learner.learn(weights)

    unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
    pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states ** 2))
    print("Unary weights:\n" + repr(unary_mat))
    print("Pairwise weights:\n" + repr(pair_mat))

    Eval = Evaluator(max_height, max_width)
    if num_training_images > 0:
        print("Training:")
        if inc == True:
            train_errors, train_total_inconsistency = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2, num_training_images, inference_type, max_iter, inc, plot)
        else:
            train_errors = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2, num_training_images, inference_type, max_iter, inc, plot)
        print ("Average Train Error rate: %f" % train_errors)

    sheet1.write(1, 4, train_errors, style)
    if inc == True:
        sheet1.write(1, 5, train_total_inconsistency, style)
    else:
        sheet1.write(1, 5, 'Not calculated', style)

    if num_testing_images > 0:
        print("Test:")
        if inc == True:
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2,
                                                                                 num_testing_images, inference_type,
                                                                                 max_iter, inc, plot)
        else:
            test_errors = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images,
                                                       inference_type, max_iter, inc, plot)
        print ("Average Test Error rate: %f" % test_errors)

        sheet1.write(1, 6, test_errors, style)
        if inc == True:
            sheet1.write(1, 7, test_total_inconsistency, style)
        else:
            sheet1.write(1, 7, 'Not calculated', style)

    else:
        sheet1.write(1, 6, 'Not calculated', style)
        sheet1.write(1, 7, 'Not calculated', style)

    elapsed = time.time() - start

    print ("Time elapsed: %f" % elapsed)

    sheet1.write(1, 8, elapsed, style)
    saved_file_name = str(max_iter) + inference_type_name + str(objective_type) + str(l2_regularization)
    wb.save(saved_file_name + 'Results.xls')


if __name__ == "__main__":
    main()
