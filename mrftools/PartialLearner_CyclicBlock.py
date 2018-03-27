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


    def objective1(self, weights, options=None):
        """
        Return the primal regularized negative variational log likelihood
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: objective value
        """

        self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, False)

        term_p = sum([np.true_divide(x.compute_energy_functional(), len(x.mn.variables)) for x in
                      self.belief_propagators]) / len(self.belief_propagators)

        if not self.fully_observed:
            # recompute energy functional for label distributions only in latent variable case
            self.set_weights(weights, self.conditioned_belief_propagators)
            term_q = sum([np.true_divide(x.compute_energy_functional(), len(x.mn.variables)) for x in
                          self.conditioned_belief_propagators]) / len(self.conditioned_belief_propagators)
        else:
            term_q = np.dot(self.label_expectations, weights)

        self.term_q_p = term_p - term_q

        objective = 0.0
        # add regularization penalties
        objective += self.l1_regularization * np.sum(np.abs(weights))
        objective += self.l2_regularization * weights.dot(weights)
        objective += self.term_q_p

        return objective
