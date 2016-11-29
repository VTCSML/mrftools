try:
    import autograd.numpy as np
except ImportError:
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
        for bp in self.belief_propagators + self.belief_propagators_q:
            bp.set_max_iter(self.bp_iter)
        # res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=self.subgrad_grad, callback=callback_f)
        # new_weights = res.x

        res = ada_grad(self.dual_obj, self.subgrad_grad, weights, args= None, callback= callback_f)
        new_weights = res
        return new_weights


    def dual_obj(self, weights, options=None):
        # if self.tau_q is None or not self.fully_observed:
        #     self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)

        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)

        term_p = sum([x.compute_dual_objective() for x in self.belief_propagators]) / len(self.belief_propagators)
        term_q = sum([x.compute_dual_objective() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        # print term_q
        # print np.dot(self.tau_q, weights)
        # term_q = np.dot(self.tau_q, weights)
        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * np.dot(weights, weights)
        objec += self.term_q_p

        return objec
