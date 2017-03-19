"""Class to do generative learning directly on MRF parameters."""
from LogLinearModel import LogLinearModel
from Learner import Learner
from PairedDual import PairedDual
from MatrixBeliefPropagator import MatrixBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
import numpy as np
from itertools import combinations
from scipy.sparse import csc_matrix
from copy import deepcopy

class ApproxMaxLikelihood(Learner):
    """Object that runs approximate maximum likelihood parameter training."""

    # def __init__(self, markov_net, inference_type=MatrixBeliefPropagator):
    def __init__(self, markov_net, inference_type=ConvexBeliefPropagator):
        super(ApproxMaxLikelihood, self).__init__(inference_type)
        self.base_model = LogLinearModel()
        self.base_model.create_indicator_model(markov_net)

    def add_data(self, labels):
        model = deepcopy(self.base_model)
        super(ApproxMaxLikelihood, self).add_data(labels, model)

        # as a hack to save time, since these models don't condition on anything, make all belief propagators equal
        self.belief_propagators = [self.belief_propagators[0]]


    def init_grafting(self):
        model = deepcopy(self.base_model)
        super(ApproxMaxLikelihood, self).init_model(model)

        self.belief_propagators = [self.belief_propagators[0]]