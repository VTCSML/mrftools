import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from opt import *
# from PIL.ImageGrab import grab

class EM(Learner):
    
    def __init__(self, inference_type):
        super(EM, self).__init__( inference_type)
        
    def learn(self, weights, optimzer, callback_f = None):
        old_weights = np.inf
        new_weights = weights
        self.start = time.time ( )
        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            self.e_step(new_weights)
            new_weights = self.m_step(new_weights, optimzer, callback_f)

        return new_weights

    def e_step(self, weights):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)

    def m_step(self, weights,optimzer, callback_f=None):
        # res = rms_prop ( self.objective, self.gradient, weights, args=None, callback=callback_f )
        # res = ada_grad ( self.objective, self.gradient, weights, args=None, callback=callback_f )
        res = optimzer ( self.objective, self.gradient, weights, args=None, callback=callback_f )
        return res
        # res = minimize(self.objective, weights, None, method='L-BFGS-B', jac = self.gradient, callback=callback_f)
        # return res.x


    def leanr_repeated(self,weights, callback_f = None):
        old_weights = np.inf
        new_weights = weights
        self.start = time.time ( )
        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            self.e_step(new_weights)
            new_weights = self.m_step_repeated(new_weights, callback_f)

        return new_weights


    def m_step_repeated(self, weights, callback_f):
        res = ada_grad ( self.subgrad_obj, self.subgrad_grad, weights, args=None, callback=callback_f )
        # res = adam ( self.objective, self.gradient, weights, args=None, callback=callback_f )
        return res
        # res = minimize(self.objective, weights, None, method='L-BFGS-B', jac = self.gradient, callback=callback_f)
        # return res.x