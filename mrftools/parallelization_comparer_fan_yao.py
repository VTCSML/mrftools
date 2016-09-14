from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
import time
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
import ImageSegmentation
import resource
from memory_profiler import profile
from ImageLoader import ImageLoader
import os
import xlwt
import time

def main():

    memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print 'Initial Memory usage: %s (kb)' % memory_start
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
    max_height = 20
    max_width = 20
    num_training_images = 1
    num_testing_images = 1

    num_states = 2
    d_unary = 65
    d_edge = 11
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
    inc = True
    plot = False
    initialization_flag = True

    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_RIGHT
    style = xlwt.XFStyle()
    style.alignment = alignment
    style.num_format_str = '#,##0.0000'


    loader = ImageLoader(max_width, max_height)
    images, models, labels, names = loader.load_all_images_and_labels(path+'/data/horse/train', 2, num_training_images)

    max_iters = [3, 10]
    objective_types = ['primal', 'dual']
    l2_regularizations = [1.0]
    # inference_types = {'BP': MatrixBeliefPropagator, 'TRBP': MatrixTRBeliefPropagator, 'ConvexBP': ConvexBeliefPropagator}
    inference_types = {'BP': MatrixBeliefPropagator}

    comparing_set = []
    for max_iter in max_iters:
        for inference_type_name in inference_types:
            inference_type = inference_types[inference_type_name]
            for objective_type in objective_types:
                for l2_regularization in l2_regularizations:
                    configuration = (max_iter, inference_type_name, inference_type, objective_type, l2_regularization)
                    comparing_set.append(configuration)

    start = time.time()

    #pool = Pool()
    #results = pool.map(ImageSegmentation.image_segmentation, comparing_set)


    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, backend = "threading")(
        delayed(ImageSegmentation.image_segmentation)(configuration, images, models, labels, names, num_states, d_unary, d_edge, path, max_height, max_width, num_training_images, num_testing_images, inc, plot, initialization_flag, style) for configuration in comparing_set)



    elapsed = time.time() - start
    print ("Time elapsed: %f" % elapsed)



if __name__ == "__main__":
    main()




