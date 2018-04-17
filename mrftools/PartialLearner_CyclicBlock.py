import copy
from mrftools import *
from .PartialConvexBP_CyclicBlock import PartialConvexBP_CyclicBolck
import time


class PartialLearner_CyclicBlock(Learner):
    def __init__(self, num_R, num_C, inference_type = PartialConvexBP_CyclicBolck):
        super(PartialLearner_CyclicBlock, self).__init__(inference_type)
        self._num_R = num_R
        self._num_C = num_C
        self._t = 1

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

            if self._t < 5:
                bp.infer(display=self.display)
            else:
                bp.partial_infer()
            #bp.partial_infer()


    def objective1(self, weights, options=None):
        """
        Return the primal regularized negative variational log likelihood
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: objective value
        """

        self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, True)

        #self.update_summed_features(self.belief_propagators)

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


    def objective(self, weights, options=None):
        """
        Return the primal regularized negative variational log likelihood
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: objective value
        """



        #self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, True)
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


    def gradient(self, weights, options=None):
        """
        Return the gradient of the regularized negative variational log likelihood
        :param weights: weight vector containing weights for all potentials
        :param options: Unused (for now) options for objective function
        :return: gradient vector
        """

        if self.start_time != 0 and time.time() - self.start_time > self.max_time:
            if self.display == 'full':
                print('more than %d seconds...' % self.max_time)
            grad = np.zeros(len(weights))
            return grad
        else:
            #self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, False)
            self.inferred_expectations = self.calculate_expectations(weights, self.belief_propagators, True)

            grad = np.zeros(len(weights))

            # add regularization penalties
            grad += self.l1_regularization * np.sign(weights)
            grad += self.l2_regularization * weights

            grad -= np.squeeze(self.label_expectations)
            grad += np.squeeze(self.inferred_expectations)

            return grad


    def update_summed_features(self, belief_propagators):

        for bp in belief_propagators:
            bp.update_summed_features()


