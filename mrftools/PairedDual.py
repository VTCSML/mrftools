from Learner import Learner
from opt import *


class PairedDual(Learner):
    def __init__(self, inference_type, bp_iter=2, warm_up=5):
        super(PairedDual, self).__init__(inference_type)
        self.bp_iter = bp_iter
        self.warm_up = warm_up

    def learn(self, weights, optimizer=ada_grad, callback=None, opt_args=None):
        for bp in self.belief_propagators + self.belief_propagators_q:
            bp.set_max_iter(self.bp_iter)
            for i in range(self.warm_up):
                bp.update_messages()

        self.start = time.time()
        new_weights = optimizer(self.dual_obj, self.subgrad_grad, weights, args=opt_args, callback=callback)

        return new_weights
