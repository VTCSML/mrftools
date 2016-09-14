from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
import time
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
import ImageSegmentation
import xlwt
import os
from ImageLoader import ImageLoader
import resource
from memory_profiler import profile


def main():
    max_iters = [10]
    objective_types = ['primal', 'dual']
    l2_regularizations = [1.0]
    inference_types = {'BP': MatrixBeliefPropagator}

    num_states = 2
    d_unary = 65
    d_edge = 11
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
    max_height = 5
    max_width = 5
    num_training_images = 1
    num_testing_images = 1
    inc = True
    plot = False
    initialization_flag = True

    loader = ImageLoader(max_height, max_width)
    images, models, labels, names = loader.load_all_images_and_labels(path + '/data/horse/train', 2,
                                                                      num_training_images)

    comparing_set = []
    for max_iter in max_iters:
        for inference_type_name in inference_types:
            inference_type = inference_types[inference_type_name]
            for objective_type in objective_types:
                for l2_regularization in l2_regularizations:
                    configuration = (max_iter, inference_type_name, inference_type, objective_type, l2_regularization)
                    comparing_set.append(configuration)

    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_RIGHT
    style = xlwt.XFStyle()
    style.alignment = alignment
    style.num_format_str = '#,##0.0000'

    # Unparallelized
    start = time.time()
    results_unp = []
    for configuration in comparing_set:
        result = ImageSegmentation.image_segmentation(configuration, images, models, labels, names, num_states, d_unary,
                                                      d_edge, path, max_height, max_width, num_training_images,
                                                      num_testing_images, inc, plot, initialization_flag, style)
        results_unp.append(result)
    elapsed_unp = time.time() - start

    # Parallelized
    start = time.time()
    num_cores = multiprocessing.cpu_count()
    results_par = Parallel(n_jobs=num_cores, backend='threading')(
        delayed(ImageSegmentation.image_segmentation)(configuration, images, models, labels, names, num_states, d_unary,
                                                      d_edge, path, max_height, max_width, num_training_images,
                                                      num_testing_images, inc, plot, initialization_flag, style) for
        configuration in comparing_set)

    elapsed_par = time.time() - start
    print("Number of Cores: %d" % num_cores)
    print("Time elaplsed for unparallelized: %f" % elapsed_unp)
    print("Time elaplsed for parallelized: %f" % elapsed_par)

    # print "results unparallelized:"
    # print results_unp
    # print "results parallelized:"
    # print results_par


    n = len(results_par)
    for i in range(n):
        assert results_par[i] in results_unp, "Parallelized result different from unparallelized"

if __name__ == "__main__":
    main()






