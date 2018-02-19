import copy
from mrftools import *
from .PartialConvexBP_CyclicBlock import PartialConvexBP_CyclicBolck


class PartialLearner_CyclicBlock(Learner):
    def __init__(self, num_R, num_C, inference_type = PartialConvexBP_CyclicBolck):
        super(PartialLearner_CyclicBlock, self).__init__(inference_type)
        self._num_R = num_R
        self._num_C = num_C

    def do_inference(self, belief_propagators):
        """
        Perform inference on all stored models.

        :param belief_propagators: iterable of inference objects
        :return: None
        """
        for bp in belief_propagators:
            # if self.initialization_flag:
            #     bp.initialize_messages()
            if bp._update_nodes_list == None:
                bp.separate_graph(self._num_R, self._num_C)

            bp.partial_infer()
