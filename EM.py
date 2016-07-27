import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
# from PIL.ImageGrab import grab

class EM(Learner):
    
    def __init__(self, inference_type):
        super(EM, self).__init__( inference_type)
        
    def learn(self,weights, callback_f):
        old_weights = np.inf
        new_weights = weights
        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            self.e_step(new_weights)
            new_weights = self.m_step(new_weights, callback_f)
            
        return new_weights

    def e_step(self, weights):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)

    def m_step(self, weights, callback_f):
        res = minimize(self.objective, weights, None, method='L-BFGS-B', jac = self.gradient, callback=callback_f)
        return res.x
    