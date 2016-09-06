import unittest
from joblib import Parallel, delayed
import multiprocessing
import time
from mrftools import *
# from MatrixBeliefPropagator import MatrixBeliefPropagator
# from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
# from ConvexBeliefPropagator import ConvexBeliefPropagator
# import ImageSegmentation
import xlwt
import os
# from ImageLoader import ImageLoader


class TestParallelization(unittest.TestCase):

    def test_parallelization(self):
        max_iters = [10]
        objective_types = ['primal']
        l2_regularizations = [0.001, 0.01, 0.1, 1.0]
        inference_types = {'BP': MatrixBeliefPropagator}

        comparing_set = []
        for max_iter in max_iters:
            for inference_type_name in inference_types:
                inference_type = inference_types[inference_type_name]
                for objective_type in objective_types:
                    for l2_regularization in l2_regularizations:
                        configuration = (
                        max_iter, inference_type_name, inference_type, objective_type, l2_regularization)
                        comparing_set.append(configuration)

        num_states = 2
        d_unary = 65
        d_edge = 11
        path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
        max_height = 5
        max_width = 5
        num_training_images = 2
        num_testing_images = 0
        inc = True
        plot = False
        initialization_flag = True

        loader = ImageLoader(max_height, max_width)
        images, models, labels, names = loader.load_all_images_and_labels(path + '/test/data/horse/train', 2,
                                                                          num_training_images)

        alignment = xlwt.Alignment()
        alignment.horz = xlwt.Alignment.HORZ_RIGHT
        style = xlwt.XFStyle()
        style.alignment = alignment
        style.num_format_str = '#,##0.0000'

        # Unparallelized
        start = time.time()
        for configuration in comparing_set:
            ImageSegmentation.image_segmentation(configuration, images, models, labels, names, num_states, d_unary, d_edge,
                               path, max_height, max_width, num_training_images, num_testing_images,
                               inc, plot, initialization_flag, style)

        elapsed_unp = time.time() - start



        # Parallelized
        start = time.time()
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(
        delayed(ImageSegmentation.image_segmentation)(configuration, images, models, labels,
                                                      names, num_states, d_unary, d_edge, path, max_height,
                                                      max_width,
                                                      num_training_images, num_testing_images,
                                                      inc, plot, initialization_flag,
                                                      style) for configuration in comparing_set)

        elapsed_par = time.time() - start
        print("Time elaplsed for unparallelized: %f" % elapsed_unp)
        print("Time elaplsed for parallelized: %f" % elapsed_par)

        print("Number of Corse: %d" % num_cores)

        np.testing.assert_allclose(elapsed_unp, elapsed_par * num_cores, rtol=1e-1, atol=0)






