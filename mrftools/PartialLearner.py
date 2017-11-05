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

    # def calculate_expectations(self, weights, belief_propagators, should_infer=True):
    #     """
    #     Calculate the feature expectations given the provided model weights.
    #
    #     :param weights: weight vector containing weights for all potentials
    #     :param belief_propagators: iterable of belief propagators whose models should be updated with the weights
    #     :param should_infer: Boolean value of whether to run inference. This value should usually only be False when
    #                         inference has already been run for this particular weight vector, i.e., if this function
    #                         is being called immediately after it has been called with the same weights.
    #     :return: feature expectation vector
    #     """
    #     self.set_weights(weights, belief_propagators)
    #     if should_infer:
    #         self.do_inference(belief_propagators)
    #
    #     return self.get_feature_expectations(belief_propagators)
    #
    #
    #
    # def learn(self, weights, optimizer=ada_grad, callback=None, opt_args=None):
    #     self.start_time = time.time()
    #     res = optimizer(self.subgrad_obj, self.subgrad_grad, weights, opt_args, callback=callback)
    #     new_weights = res
    #
    #
    # def subgrad_obj(self, weights, options=None, do_inference=True):
    #     """
    #     Compute the variational negative log likelihood. Performs inference on latent variables in the labeled
    #     inference objects before calling the EM objective
    #
    #     :param weights: Weight vector containing the same number of entries as all weights for this model
    #     :param do_inference: Boolean value indicating whether or not to run inference. Defaults to True.
    #     :return: objective value (float)
    #     """
    #     if self.label_expectations is None or not self.fully_observed:
    #         self.label_expectations = self.calculate_expectations(weights, self.conditioned_belief_propagators,
    #                                                               do_inference)
    #     return self.objective(weights)
    #
    # def subgrad_grad(self, weights, options=None, do_inference=False):
    #     """
    #     Compute the gradient of the variational negative log likelihood.
    #
    #     :param weights: Weight vector containing the same number of entries as all weights for this model
    #     :param do_inference: Boolean value indicating whether or not to run inference. Defaults to False because
    #                         typically the objective function was called immediately before, which does inference.
    #     :return: gradient with respect to weights
    #     """
    #     if self.label_expectations is None or not self.fully_observed:
    #         self.label_expectations = self.calculate_expectations(weights, self.conditioned_belief_propagators,
    #                                                               do_inference)
    #     return self.gradient(weights)









