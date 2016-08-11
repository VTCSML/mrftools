try:
    import autograd.numpy as np
    from autograd import value_and_grad, grad
except ImportError:
    import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from opt import *


class AutogradLearner(Learner):
    def __init__(self, inference_type):
        super(AutogradLearner, self).__init__(inference_type)

        self.bp_list = []
        self.message_start = []

    def add_data(self, labels, model):
        super(AutogradLearner, self).add_data(labels, model)

        if self.num_examples == 1:
            self.message_start.append(self.weight_dim)

        for bp in (self.belief_propagators[-1], self.belief_propagators_q[-1]):
            self.bp_list.append(bp)
            self.message_start.append(self.message_start[-1] + bp.message_mat.size)

    def learn(self, weights, callback_f=None):

        gradient = grad(self.subgrad_obj)
        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

    def learn_dual(self, weights, callback_f=None):

        gradient = grad(self.subgrad_obj_dual)
        res = minimize(self.subgrad_obj_dual, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

