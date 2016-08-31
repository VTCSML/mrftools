try:
    import autograd.numpy as np
    from autograd import value_and_grad, grad
except ImportError:
    import numpy as np
from scipy.optimize import minimize, check_grad
from Learner import Learner
from opt import *


class AutogradLearner(Learner):

    def __init__(self, inference_type):
        super(AutogradLearner, self).__init__(inference_type)

        self.initialization_flag = True

    def learn(self, weights, callback_f=None):

        gradient = grad(self.subgrad_obj)
        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

    def dual_obj(self, weights, options=None):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)

        term_p = sum([x.compute_dual_objective() for x in self.belief_propagators]) / len(self.belief_propagators)
        term_q = sum([x.compute_dual_objective() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * np.dot(weights, weights)
        objec += self.term_q_p

        return objec

    def learn_dual(self, weights, callback_f=None):

        gradient = grad(self.dual_obj)
        res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

