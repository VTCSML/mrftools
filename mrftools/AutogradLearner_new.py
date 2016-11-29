try:
    import autograd.numpy as np
    from autograd import value_and_grad, grad
except ImportError:
    import numpy as np
from scipy.optimize import minimize, check_grad
from Learner import Learner
from opt import *
# from AutogradEvaluator import AutogradEvaluator

class AutogradLearner_new(Learner):

    def __init__(self, inference_type, ais):
        super(AutogradLearner_new, self).__init__(inference_type)

        self.initialization_flag = True
        n = 500
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.training_error_list = np.zeros(n)
        self.testing_error_list = np.zeros(n)
        self.i = 0
        self.ais = ais

    def objective(self, weights, options=None):
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)

        term_p = sum([x.compute_univariate_logistic_loss() for x in self.belief_propagators]) / len(self.belief_propagators)


        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * np.dot(weights, weights)
        objec += term_p

        return objec

    def learn(self, weights, callback_f=None):
        n = 500
        self.objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.training_error_list = np.zeros(n)
        self.testing_error_list = np.zeros(n)
        self.i = 0

        gradient = grad(self.objective)

        res = minimize(self.objective, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

    def objective_anytime(self, weights, options=None):

        term_p = sum([x.compute_univariate_logistic_loss() for x in self.belief_propagators]) / len(self.belief_propagators)
        term_q = sum([x.compute_univariate_logistic_loss() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * np.dot(weights, weights)
        objec += self.term_q_p
        # objec += term_p

        return objec

    # def learn_anytime(self, weights, callback_f=None):
    #     n = 500
    #     self.objective_list = np.zeros(n)
    #     self.inconsistency_list = np.zeros(n)
    #     self.training_error_list = np.zeros(n)
    #     self.testing_error_list = np.zeros(n)
    #     self.i = 0
    #
    #     gradient = grad(self.objective)
    #
    #     res = minimize(self.objective_anytime, weights, method='L-BFGS-B', jac=gradient)
    #     new_weights = res.x
    #
    #     return new_weights

    def f(self, weights):
        a = self.objective(weights)
        b = sum([x.compute_inconsistency() for x in self.belief_propagators]) / len(self.belief_propagators)

        self.primal_objective_list[self.i] = a
        self.inconsistency_list[self.i] = b

        train_errors, test_errors = self.ais.evaluating2(weights)
        self.training_error_list[self.i] = train_errors
        self.testing_error_list[self.i] = test_errors
        self.i += 1


