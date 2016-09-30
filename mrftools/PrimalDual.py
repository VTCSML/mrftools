import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from PairedDual import PairedDual
from opt import *


class PrimalDual(PairedDual):
    def __init__(self, inference_type):
        super(PrimalDual, self).__init__(inference_type)
        self.bp_iter = 300

    def learn(self, weights, callback_f=None):
        for bp in self.belief_propagators_q:
            bp.set_max_iter(self.bp_iter)

        for bp in self.belief_propagators:
            bp.set_max_iter(1)

        # res = minimize(self.dual_obj, weights, method='L-BFGS-B', jac=self.subgrad_grad, callback=callback_f)
        # new_weights = res.x
        self.start = time.time ( )
        res = ada_grad(self.dual_obj, self.subgrad_grad, weights, args= None, callback= callback_f)
        new_weights = res
        return new_weights
