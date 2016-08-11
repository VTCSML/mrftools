try:
    import autograd.numpy as np
    from autograd import value_and_grad, grad
except ImportError:
    import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from PairedDual import PairedDual
from opt import *


class AutogradLearner(PairedDual):
    def learn(self, weights, callback_f=None):

        gradient = grad(self.subgrad_obj)
        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

    def learn_dual(self, weights, callback_f=None):

        gradient = grad(self.dual_obj)
        res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=gradient)
        new_weights = res.x

        return new_weights

