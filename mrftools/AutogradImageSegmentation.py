try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
from ImageLoader import ImageLoader
from AutogradLearner import AutogradLearner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
from AutogradEvaluator import AutogradEvaluator
import os
import time

def main():

    start = time.time()

    d_unary = 65
    num_states = 2
    d_edge = 11
    max_height = 10
    max_width = 10
    num_training_images = 2
    num_testing_images = 1
    max_iter = 5
    inc = True
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'),os.path.pardir))
    plot = False
    initialization_flag = True

#   inference_type = MatrixBeliefPropagator
#   inference_type = MatrixTRBeliefPropagator
    inference_type = ConvexBeliefPropagator

    loader = ImageLoader(max_height, max_width)

    images, models, labels, names = loader.load_all_images_and_labels(path+'/test/train', 2, num_training_images)

    learner = AutogradLearner(inference_type)

    learner._set_initialization_flag(initialization_flag)

    learner.set_regularization(0.0, 1.0)

    for model, states in zip(models, labels):
     learner.add_data(states, model)

    for bp in learner.belief_propagators_q:
     bp.set_max_iter(max_iter)
    for bp in learner.belief_propagators:
     bp.set_max_iter(max_iter)

    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

    # =====================================
    # Primal
    # =====================================

    print "\n------------Primal-------------"

    new_weights = learner.learn(weights)

    Eval = AutogradEvaluator(max_height, max_width)
    if num_training_images > 0:
        print("Training:")
        if inc == True:
            train_errors, train_total_inconsistency = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2, num_training_images, inference_type, max_iter, inc, plot)
        else:
            train_errors = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2, num_training_images, inference_type, max_iter, inc, plot)
        print ("Average Train Error rate: %f" % train_errors)

    if num_testing_images > 0:
        print("Test:")
        if inc == True:
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images, inference_type, max_iter, inc, plot)
        else:
            test_errors = Eval.evaluate_testing_images(path+'/test/test', new_weights, 2, num_testing_images, inference_type, max_iter, inc, plot)
        print ("Average Test Error rate: %f" % test_errors)

        elapsed = time.time() - start

    print ("Time elapsed: %f" % elapsed)

    # =====================================
    # Dual
    # =====================================

    print "\n------------Dual-------------"

    new_weights = learner.learn_dual(weights)

    if num_training_images > 0:
        print("Training:")
        if inc == True:
            train_errors, train_total_inconsistency = Eval.evaluate_training_images(images, models, labels, names,
                                                                                    new_weights, 2, num_training_images,
                                                                                    inference_type, max_iter, inc, plot)
        else:
            train_errors = Eval.evaluate_training_images(images, models, labels, names, new_weights, 2,
                                                         num_training_images, inference_type, max_iter, inc, plot)
        print ("Average Train Error rate: %f" % train_errors)

    if num_testing_images > 0:
        print("Test:")
        if inc == True:
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images(path + '/test/test', new_weights, 2,
                                                                                 num_testing_images, inference_type,
                                                                                 max_iter, inc, plot)
        else:
            test_errors = Eval.evaluate_testing_images(path + '/test/test', new_weights, 2, num_testing_images,
                                                       inference_type, max_iter, inc, plot)
        print ("Average Test Error rate: %f" % test_errors)


if __name__ == "__main__":
    main()
