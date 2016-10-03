try:
    import autograd.numpy as np
    from autograd import value_and_grad, grad
except ImportError:
    import numpy as np
from scipy.optimize import minimize, check_grad
from Learner import Learner
from opt import *
# from AutogradEvaluator import AutogradEvaluator

class AutogradLearner(Learner):

    def __init__(self, inference_type):
        super(AutogradLearner, self).__init__(inference_type)

        self.initialization_flag = True
        n = 1000
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.i = 0
        self.calculate_dual_gradient()
        self.calculate_primal_gradient()

    def calculate_dual_gradient(self):
        self.dual_gradient = grad(self.dual_obj)

    def calculate_primal_gradient(self):
        self.primal_gradient = grad(self.subgrad_obj)

    def learn(self, weights, callback_f=None):

        n = 1000
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.i = 0
        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=self.primal_gradient, callback=self.f)
        # res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=self.primal_gradient)
        new_weights = res.x

        plt.subplot(311)
        plt.plot(self.dual_objective_list)
        plt.ylabel('dual objective')
        plt.xlabel('number of minimization iterations')
        plt.subplot(312)
        plt.plot(self.primal_objective_list)
        plt.ylabel('primal objective')
        plt.xlabel('number of minimization iterations')
        plt.subplot(313)
        plt.plot(self.inconsistency_list)
        plt.ylabel('inconsistency')
        plt.xlabel('number of minimization iterations')
        plt.show()

        return new_weights


    def learn_dual(self, weights, callback_f=None):

        n = 1000
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.i = 0

        res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=self.dual_gradient, callback=self.f)
        # res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=self.dual_gradient)
        new_weights = res.x

        plt.subplot(311)
        plt.plot(self.dual_objective_list)
        plt.ylabel('dual objective')
        plt.xlabel('number of minimization iterations')
        plt.subplot(312)
        plt.plot(self.primal_objective_list)
        plt.ylabel('primal objective')
        plt.xlabel('number of minimization iterations')
        plt.subplot(313)
        plt.plot(self.inconsistency_list)
        plt.ylabel('inconsistency')
        plt.xlabel('number of minimization iterations')
        plt.show()



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


    def f(self, weights):

        a = self.dual_obj(weights)
        b = self.subgrad_obj(weights)
        c = sum([x.compute_inconsistency() for x in self.belief_propagators]) / len(self.belief_propagators)


        self.dual_objective_list[self.i] = a
        self.primal_objective_list[self.i] = b
        self.inconsistency_list[self.i] = c
        self.i += 1




