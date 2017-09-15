from PairedDual import PairedDual
from opt import *


class PrimalDual(PairedDual):
    def __init__(self, inference_type):
        super(PrimalDual, self).__init__(inference_type)
        self.bp_iter = 300

    def learn(self, weights, optimizer, callback=None):
        for bp in self.belief_propagators_q:
            bp.set_max_iter(self.bp_iter)

        for bp in self.belief_propagators:
            bp.set_max_iter(1)

        self.start = time.time()
        res = optimizer(self.dual_obj, self.subgrad_grad, weights, args=None, callback=callback)
        new_weights = res
        return new_weights
