import numpy as np
from ImageLoader import ImageLoader
from PairedDual import PairedDual
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
from Evaluator import Evaluator
import time
import os
from xlwt import Workbook
import xlwt
import ImageSegmentation



def main():

    max_iters = [5]
    objective_types = ['primal', 'dual']
    l2_regularizations = [1.0]
    initialization_flag = True
    inference_types = {'BP': MatrixBeliefPropagator}
    # inference_types = {'BP': MatrixBeliefPropagator, 'TRBP': MatrixTRBeliefPropagator, 'ConvexBP': ConvexBeliefPropagator}

    max_height = 10
    max_width = 10
    num_training_images = 1
    num_testing_images = 1

    d_unary = 65
    num_states = 2
    d_edge = 11
    inc = True
    plot = False

    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))

    loader = ImageLoader(max_height, max_width)

    images, models, labels, names = loader.load_all_images_and_labels(path+'/data/horse/train', 2, num_training_images)


    comparing_set = []
    for max_iter in max_iters:
        for inference_type_name in inference_types:
            inference_type = inference_types[inference_type_name]
            for objective_type in objective_types:
                for l2_regularization in l2_regularizations:
                    configuration = (max_iter, inference_type_name, inference_type, objective_type, l2_regularization)
                    comparing_set.append(configuration)


    # chart frame:
    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_RIGHT
    style = xlwt.XFStyle()
    style.alignment = alignment
    style.num_format_str = '#,##0.0000'


    start = time.time()
    results_unp = []
    for configuration in comparing_set:
        result = ImageSegmentation.image_segmentation(configuration, images, models, labels, names, num_states, d_unary,
                                                      d_edge, path, max_height, max_width, num_training_images,
                                                      num_testing_images, inc, plot, initialization_flag, style)
        results_unp.append(result)

    elapsed_unp = time.time() - start
    print("Time elaplsed: %f" % elapsed_unp)


if __name__ == "__main__":
    main()
