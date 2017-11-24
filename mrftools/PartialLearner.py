import copy
from mrftools import *
from .PartialMatrixBP import PartialMatrixBP


class PartialLearner(Learner):
    def __init__(self, N, inference_type = PartialMatrixBP):
        super(PartialLearner, self).__init__(inference_type)
        self._N = N

    def do_inference(self, belief_propagators):
        """
        Perform inference on all stored models.

        :param belief_propagators: iterable of inference objects
        :return: None
        """
        for bp in belief_propagators:
            # if self.initialization_flag:
            #     bp.initialize_messages()
            bp.partial_infer(N=self._N)









