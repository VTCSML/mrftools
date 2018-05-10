import copy
from mrftools import *
from .PartialConvexBP_CyclicBlock import PartialConvexBP_CyclicBolck
import time


class PartialLearner_CyclicBlock(Learner):
    def __init__(self, num_R, num_C, inference_type = PartialConvexBP_CyclicBolck):
        super(PartialLearner_CyclicBlock, self).__init__(inference_type)
        self._num_R = num_R
        self._num_C = num_C
        self._t = 0

    def do_inference(self, belief_propagators):
        """
        Perform inference on all stored models.

        :param belief_propagators: iterable of inference objects
        :return: None

        """

        self._t = self._t + 1
        for bp in belief_propagators:
            # if self.initialization_flag:
            #     bp.initialize_messages()
            if bp._update_nodes_list == None:
                bp.separate_graph(self._num_R, self._num_C)

            if self._t <= 3:
                bp.infer(display=self.display)
            else:
                bp.partial_infer()
            #bp.partial_infer()



