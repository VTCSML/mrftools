import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from opt import *


class GradientInferenceLearner(Learner):
    def __init__(self, inference_type):
        super(GradientInferenceLearner, self).__init__(inference_type)

        self.bp_list = []
        self.message_start = []

    def add_data(self, labels, model):
        super(GradientInferenceLearner, self).add_data(labels, model)

        if self.num_examples == 1:
            self.message_start.append(self.weight_dim)

        for bp in (self.belief_propagators[-1], self.belief_propagators_q[-1]):
            self.bp_list.append(bp)
            self.message_start.append(self.message_start[-1] + bp.message_mat.size)

    def learn(self, weights, callback_f=None):
        weights_messages = np.zeros(self.message_start[-1])
        weights_messages[:self.weight_dim] = weights

        for index, bp in enumerate(self.bp_list):
            weights_messages[self.message_start[index]:self.message_start[index+1]] = bp.message_mat.ravel()

        new_weights_messages = ada_grad(self.dual_obj, self.dual_grad, weights_messages, None, callback_f)

        return new_weights_messages[:self.weight_dim]

    def dual_obj(self, weights_messages, options=None):
        weights = weights_messages[:self.weight_dim]
        for index, bp in enumerate(self.bp_list):
            messages = weights_messages[self.message_start[index]:self.message_start[index+1]]
            bp.set_messages(messages.reshape(bp.message_mat.shape))

        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, False)
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, False)

        term_p = sum([x.compute_dual_objective() for x in self.belief_propagators]) / self.num_examples
        term_q = sum([x.compute_dual_objective() for x in self.belief_propagators_q]) / self.num_examples
        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        return objec

    def dual_grad(self, weights_messages, options=None):
        weights = weights_messages[:self.weight_dim]
        for index, bp in enumerate(self.bp_list):
            messages = weights_messages[self.message_start[index]:self.message_start[index+1]]
            bp.set_messages(messages.reshape(bp.message_mat.shape))

        grad = np.zeros(weights_messages.shape)
        grad[:self.weight_dim] = self.subgrad_grad(weights, options)

        for index, bp in enumerate(self.bp_list):
            message_grad = bp._compute_inconsistency_vector().ravel()
            # if index % 2 == 1:
            #     message_grad = -message_grad
            grad[self.message_start[index]:self.message_start[index+1]] = message_grad

        return grad