try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
from ImageLoader import ImageLoader
from AutogradLearner import AutogradLearner
from AutogradLearner_new import AutogradLearner_new
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixTRBeliefPropagator import MatrixTRBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
from AutogradEvaluator import AutogradEvaluator
import os
import time
import matplotlib.pyplot as plt

class AutogradImageSegmentation(object):
    def __init__(self):
        self.d_unary = 65
        self.num_states = 2
        self.d_edge = 11
        self.max_height = 20
        self.max_width = 20
        self.num_training_images = 1
        self.num_testing_images = 1
        self.inc = True
        self.path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
        self.plot = False
        self.initialization_flag = True

        self.inference_type = ConvexBeliefPropagator
        self.max_iter = 4
        self.l2regularization = 1.0
        self.models = []
        self.labels = []
        self.names = []
        self.images = []
        self.dimensions = []
        self.length = []


        n = 1000
        self.training_error_list = np.zeros(n)
        self.testing_error_list = np.zeros(n)
        self.i = 0



    def set_up(self):

        loader = ImageLoader(self.max_height, self.max_width)

        self.images, self.models, self.labels, self.names = loader.load_all_images_and_labels(self.path+'/test/train', 2, self.num_training_images)

        self.learner = AutogradLearner(self.inference_type, self)

        self.learner.set_regularization(0.0, 1.0)

        self.learner._set_initialization_flag(self.initialization_flag)

        for model, states in zip(self.models, self.labels):
         self.learner.add_data(states, model)

        for bp in self.learner.belief_propagators_q:
         bp.set_max_iter(self.max_iter)
        for bp in self.learner.belief_propagators:
         bp.set_max_iter(self.max_iter)

        self.weights = np.zeros(self.d_unary * self.num_states + self.d_edge * self.num_states ** 2)


    def set_up2(self):

        self.learner = AutogradLearner(self.inference_type, self)

        self.learner.set_regularization(0.0, 1.0)

        self.learner._set_initialization_flag(self.initialization_flag)

        for model, states in zip(self.models, self.labels):
         self.learner.add_data(states, model)

        for bp in self.learner.belief_propagators_q:
         bp.set_max_iter(self.max_iter)
        for bp in self.learner.belief_propagators:
         bp.set_max_iter(self.max_iter)

        self.weights = np.zeros(self.d_unary * self.num_states + self.d_edge * self.num_states ** 2)


    def learn_primal(self):
        start = time.time()
        print "\n------------Primal-------------"
        new_weights = self.learner.learn(self.weights)

        # print("Dual Objective: %f" % self.learner.dual_obj(new_weights))
        # print("Primal Objective: %f" % self.learner.subgrad_obj(new_weights, self))
        elapsed = time.time() - start

        print ("Time elapsed: %f" % elapsed)
        return new_weights


    def learn_dual(self):
        start = time.time()
        print "\n------------Dual-------------"
        new_weights = self.learner.learn_dual(self.weights)

        # print("Dual Objective: %f" % self.learner.dual_obj(new_weights))
        # print("Primal Objective: %f" % self.learner.subgrad_obj(new_weights, self))

        elapsed = time.time() - start
        print ("Time elapsed: %f" % elapsed)
        return new_weights

    def evaluating(self, weights):
        Eval = AutogradEvaluator(self.max_height, self.max_width)
        if self.num_training_images > 0:
            print("Training:")
            train_errors, train_total_inconsistency = Eval.evaluate_training_images(self.images, self.models, self.labels, self.names, weights, self.num_training_images, self.inference_type, self.max_iter, self.inc, self.plot)
            print("Overall inconsistency: %f" % train_total_inconsistency)
            print ("Average Train Error rate: %f" % train_errors)

        if self.num_testing_images > 0:
            print("Testing:")
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images(self.path+'/test/test', weights, self.num_states, self.num_testing_images, self.inference_type, self.max_iter, self.inc, self.plot)
            print("Overall inconsistency: %f" % test_total_inconsistency)
            print ("Average Test Error rate: %f" % test_errors)

    def evaluating2(self, weights):
        Eval = AutogradEvaluator(self.max_height, self.max_width)
        if self.num_training_images > 0:
            train_errors, train_total_inconsistency = Eval.evaluate_training_images(self.images, self.models, self.labels, self.names, weights, self.num_training_images, self.inference_type, self.max_iter, self.inc, self.plot)

        if self.num_testing_images > 0:
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images(self.path+'/test/test', weights, self.num_states, self.num_testing_images, self.inference_type, self.max_iter, self.inc, self.plot)
        return train_errors, test_errors


    def evaluating3(self, weights):
        Eval = AutogradEvaluator(self.max_height, self.max_width)
        if self.num_training_images > 0:
            print("Training:")
            train_errors, train_total_inconsistency = Eval.evaluate_training_images2(self.dimensions, self.models, self.labels, self.names, weights, self.num_training_images, self.inference_type, self.max_iter, self.inc, self.plot)
            print("Overall inconsistency: %f" % train_total_inconsistency)
            print ("Average Train Error rate: %f" % train_errors)

        if self.num_testing_images > 0:
            print("Testing:")
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images2(self.path+'/test/test', weights, self.num_states, self.num_testing_images, self.inference_type, self.max_iter, self.inc, self.plot)
            print("Overall inconsistency: %f" % test_total_inconsistency)
            print ("Average Test Error rate: %f" % test_errors)


    def evaluating4(self, weights):
        Eval = AutogradEvaluator(self.max_height, self.max_width)
        if self.num_training_images > 0:
            print("Training:")
            train_errors, train_total_inconsistency = Eval.evaluate_training_images3(self.length, self.models,
                                                                                     self.labels, self.names, weights,
                                                                                     self.num_training_images,
                                                                                     self.inference_type, self.max_iter,
                                                                                     self.inc, self.plot)
            print("Overall inconsistency: %f" % train_total_inconsistency)
            print ("Average Train Error rate: %f" % train_errors)

        if self.num_testing_images > 0:
            print("Testing:")
            test_errors, test_total_inconsistency = Eval.evaluate_testing_images3(self.path + '/test/test', weights,
                                                                                  self.num_states, self.num_testing_images,
                                                                                  self.inference_type, self.max_iter,
                                                                                  self.inc, self.plot)
            print("Overall inconsistency: %f" % test_total_inconsistency)
            print ("Average Test Error rate: %f" % test_errors)

def main():

    ais = AutogradImageSegmentation()

    ais.set_up()

    primal_weights = ais.learn_primal()
    ais.evaluating(primal_weights)
    # #
    ais.learner.tau_q = None
    dual_weights = ais.learn_dual()
    ais.evaluating(dual_weights)




if __name__ == "__main__":
    main()
