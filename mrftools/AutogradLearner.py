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
        print("objective")
        print(self.subgrad_obj(weights))
        gradient = grad(self.subgrad_obj)
        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=gradient)
        print("optimized objective")
        print(self.subgrad_obj(res.x))
        print "\n\nGradient check at optimized solution:"
        print check_grad(self.subgrad_obj, gradient, res.x * (1 + 1e-9))

        #training_loss_and_grad = value_and_grad(self.subgrad_obj)
        #gradient = grad(self.subgrad_obj)
        #res = minimize(self.subgrad_obj, weights, method='L-BFGS-b', jac=gradient)
        #res = minimize(training_loss_and_grad, weights, args='autograd', method='L-BFGS-b', jac=True)
        new_weights = res.x

        return new_weights

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
        objec += 0.5 * self.l2_regularization * np.dot(weights, weights)
        objec += self.term_q_p

        return objec
