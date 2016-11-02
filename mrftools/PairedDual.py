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
    def __init__(self, inference_type, bp_iter=2, warm_up=5):
        super(PairedDual, self).__init__(inference_type)
        self.bp_iter = bp_iter
        self.warm_up = warm_up

    def learn(self, weights, callback_f=None):
        for bp in self.belief_propagators + self.belief_propagators_q:
            bp.set_max_iter(self.bp_iter)
            for i in range(self.warm_up):
                    bp.update_messages()

        res = rms_prop(self.dual_obj, self.subgrad_grad, weights, args= None, callback= callback_f)
        new_weights = res
        return new_weights