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

    def __init__(self, inference_type, ais = None):
        super(AutogradLearner, self).__init__(inference_type)

        self.initialization_flag = True
        n = 800
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.training_error_list = np.zeros(n)
        self.testing_error_list = np.zeros(n)
        self.i = 0
        self.ais = ais

    def learn(self, weights, callback_f=None):

        n = 800
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.training_error_list = np.zeros(n)
        self.testing_error_list = np.zeros(n)
        self.i = 0
        primal_gradient = grad(self.subgrad_obj)
        # res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=primal_gradient, callback=self.f)
        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=primal_gradient)
        # new_weights = ada_grad(self.subgrad_obj, primal_gradient, weights, 0, callback=self.f)
        new_weights = res.x

        # plt.subplot(411)
        # plt.plot(self.dual_objective_list)
        # plt.ylabel('dual objective')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(412)
        # plt.plot(self.primal_objective_list)
        # plt.ylabel('primal objective')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(413)
        # plt.plot(self.training_error_list)
        # plt.ylabel('training error')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(414)
        # plt.plot(self.testing_error_list)
        # plt.ylabel('testing error')
        # plt.xlabel('number of minimization iterations')
        # plt.show()

        return new_weights


    def learn_dual(self, weights, callback_f=None):

        n = 1000
        self.dual_objective_list = np.zeros(n)
        self.primal_objective_list = np.zeros(n)
        self.inconsistency_list = np.zeros(n)
        self.training_error_list = np.zeros(n)
        self.testing_error_list = np.zeros(n)
        self.i = 0

        dual_gradient = grad(self.dual_obj)
        # res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=dual_gradient, callback=self.f)
        res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=dual_gradient)
        new_weights = res.x

        # plt.subplot(511)
        # plt.plot(self.dual_objective_list)
        # plt.ylabel('dual objective')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(512)
        # plt.plot(self.primal_objective_list)
        # plt.ylabel('primal objective')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(513)
        # plt.plot(self.inconsistency_list)
        # plt.ylabel('inconsistency')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(514)
        # plt.plot(self.training_error_list)
        # plt.ylabel('training error')
        # plt.xlabel('number of minimization iterations')
        # plt.subplot(515)
        # plt.plot(self.testing_error_list)
        # plt.ylabel('testing error')
        # plt.xlabel('number of minimization iterations')
        # plt.show()

        return new_weights

    def f(self, weights):
        a = self.dual_obj(weights)
        b = self.subgrad_obj(weights)
        # c = sum([x.compute_inconsistency() for x in self.belief_propagators]) / len(self.belief_propagators)

        self.dual_objective_list[self.i] = a
        self.primal_objective_list[self.i] = b
        # self.inconsistency_list[self.i] = c

        train_errors, test_errors = self.ais.evaluating2(weights)
        self.training_error_list[self.i] = train_errors
        self.testing_error_list[self.i] = test_errors
        self.i += 1





