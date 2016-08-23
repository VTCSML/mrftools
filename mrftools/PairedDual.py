import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from opt import *


class PairedDual(Learner):
    def __init__(self, inference_type, bp_iter=1):
        super(PairedDual, self).__init__(inference_type)
        self.bp_iter = bp_iter

    def learn(self, weights, callback_f=None):
        self.set_inference_truncation(self.bp_iter)
        res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=self.subgrad_grad, callback=callback_f)
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
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        return objec