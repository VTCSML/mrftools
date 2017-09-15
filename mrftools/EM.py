from Learner import Learner
from opt import *


# from PIL.ImageGrab import grab


class EM(Learner):
    def __init__(self, inference_type):
        super(EM, self).__init__(inference_type)

    def learn(self, weights, optimizer=ada_grad, callback=None):
        old_weights = np.inf
        new_weights = weights
        self.start = time.time()
        while not np.allclose(old_weights, new_weights, rtol=1e-4, atol=1e-5):
            old_weights = new_weights
            self.e_step(new_weights)
            new_weights = self.m_step(new_weights, optimizer, callback)

        return new_weights

    def e_step(self, weights):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)

    def m_step(self, weights, optimizer=ada_grad, callback=None):
        res = optimizer(self.objective, self.gradient, weights, args=None, callback=callback)
        return res
