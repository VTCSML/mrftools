try:
    import autograd.numpy as np
    from autograd.core import primitive
except ImportError:
    import numpy as np

from MatrixBeliefPropagator import MatrixBeliefPropagator, logsumexp, sparse_dot

class ConvexBeliefPropagator(MatrixBeliefPropagator):

    def __init__(self, markov_net, counting_numbers=None, labels=None):
        super(ConvexBeliefPropagator, self).__init__(markov_net, labels)

        self.unary_counting_numbers = np.ones(len(self.mn.variables))
        self.edge_counting_numbers = np.ones(2 * self.mn.num_edges)

        default_counting_numbers = dict()

        for var in markov_net.variables:
            default_counting_numbers[var] = 1
            for neighbor in markov_net.neighbors[var]:
                if var < neighbor:
                    default_counting_numbers[(var, neighbor)] = 1

        if counting_numbers:
            self._set_counting_numbers(counting_numbers)
        else:
            self._set_counting_numbers(default_counting_numbers)


    def _set_counting_numbers(self, counting_numbers):


        self.edge_counting_numbers = np.zeros(2 * self.mn.num_edges)

        for edge, i in self.mn.edge_index.items():
            reversed_edge = edge[::-1]
            if edge in counting_numbers:
                self.edge_counting_numbers[i] = counting_numbers[edge]
                self.edge_counting_numbers[i + self.mn.num_edges] = counting_numbers[edge]
            elif reversed_edge in counting_numbers:
                self.edge_counting_numbers[i] = counting_numbers[reversed_edge]
                self.edge_counting_numbers[i + self.mn.num_edges] = counting_numbers[reversed_edge]
            else:
                raise KeyError('Edge %s was not assigned a counting number.' % repr(edge))

        self.unary_counting_numbers = np.zeros((len(self.mn.variables), 1))

        for var, i in self.mn.var_index.items():
            self.unary_counting_numbers[i] = counting_numbers[var]

        self.unary_coefficients = self.unary_counting_numbers.copy()

        for edge, i in self.mn.edge_index.items():

            self.unary_coefficients[self.mn.var_index[edge[0]]] += self.edge_counting_numbers[i]
            self.unary_coefficients[self.mn.var_index[edge[1]]] += self.edge_counting_numbers[i]

    def compute_bethe_entropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        if self.fully_conditioned:
            entropy = 0
        else:
            entropy = - np.sum(self.edge_counting_numbers[:self.mn.num_edges] *
                               (np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor))) \
                      - np.sum(self.unary_counting_numbers.T *
                               (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))
        return entropy

    def update_messages(self, unary_mat, edge_pot_tensor, message_mat):
        """Update all messages between variables using belief division.
        Return the change in messages from previous iteration."""
        belief_mat = self.compute_beliefs(unary_mat, message_mat)
        message_mat_reverse = sparse_dot(message_mat, self.mn.reverse_mat)

        adjusted_message_prod = edge_pot_tensor - message_mat_reverse
        adjusted_message_prod /= self.edge_counting_numbers
        adjusted_message_prod += sparse_dot(belief_mat, self.mn.message_from_map.T)

        messages = np.squeeze(logsumexp(adjusted_message_prod, 1)) * self.edge_counting_numbers
        messages = np.nan_to_num(messages - messages.max(0))

        change = np.sum(np.abs(messages - message_mat))

        self.message_mat = messages

        return change, self.message_mat

    def _compute_dual_penalty(self):
        numerator = self.edge_counting_numbers
        denominator = self.unary_coefficients[self.mn.message_to].T

        coefficients = numerator / denominator

        raw_beliefs = self.mn.unary_mat + self.conditioning_mat + sparse_dot(self.message_mat, self.mn.message_to_map)

        dual_vars = self.message_mat - np.nan_to_num(coefficients * raw_beliefs[:, self.mn.message_to])

        return np.sum(dual_vars * self._compute_inconsistency_vector())

    def compute_dual_objective(self):
        return self.compute_energy_functional() + self._compute_dual_penalty()

    def compute_beliefs(self, unary_mat, message_mat):
        """Compute unary beliefs based on current messages."""
        if not self.fully_conditioned:
            self.belief_mat = unary_mat + self.conditioning_mat
            self.belief_mat += sparse_dot(message_mat, self.mn.message_to_map)

            self.belief_mat /= self.unary_coefficients.T
            log_z = logsumexp(self.belief_mat, 0)

            self.belief_mat = self.belief_mat - log_z
        return self.belief_mat


    def compute_pairwise_beliefs(self, belief_mat, message_mat, edge_pot_tensor):
        """Compute pairwise beliefs based on current messages."""
        if not self.fully_conditioned:

            adjusted_message_prod = sparse_dot(belief_mat, self.mn.message_from_map.T)
            message_mat_reverse = sparse_dot(message_mat, self.mn.reverse_mat)
            adjusted_message_prod -= message_mat_reverse

            adjusted_message_prod /= self.edge_counting_numbers

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape(
                (self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape(
                (1, self.mn.max_states, self.mn.num_edges))

            beliefs = edge_pot_tensor[:, :, self.mn.num_edges:] / \
                      self.edge_counting_numbers[self.mn.num_edges:] + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs

        return self.pair_belief_tensor

    def compute_univariate_logistic_loss(self):
        self.compute_beliefs()
        loss = - np.sum(np.nan_to_num(self.belief_mat) * self.lables_mat)
        return loss

np.reshape.defgrad(lambda ans, x, shape, order=None: lambda g: np.reshape(g, np.shape(x), order=order))

    # def compute_univariate_logistic_loss_anytime(self, tolerance=1e-8, display='iter'):
    #     """Run belief propagation until messages change less than tolerance."""
    #     change = np.inf
    #     iteration = 0
    #     loss = 0
    #     while change > tolerance and iteration < self.max_iter:
    #         change = self.update_messages()
    #         if display == "full":
    #             energy_func = self.compute_energy_functional()
    #             disagreement = self.compute_inconsistency()
    #             dual_obj = self.compute_dual_objective()
    #             if self.temp == 1:
    #                 print(
    #                 "Iteration %d, change in messages %f. Calibration disagreement: %f, energy functional: %f, dual obj: %f" % (
    #                 iteration, change, disagreement, energy_func, dual_obj))
    #                 self.temp += 1
    #             else:
    #                 print(
    #                     "Iteration %d, change in messages %s. Calibration disagreement: %s, energy functional: %s, dual obj: %s" % (
    #                         iteration, change, disagreement, energy_func, dual_obj))
    #
    #         elif display == "iter":
    #             print("Iteration %d, change in messages %f." % (iteration, change))
    #         iteration += 1
    #         loss += self.compute_univariate_logistic_loss()
    #     if display == 'final' or display == 'full' or display == 'iter':
    #         print("Belief propagation finished in %d iterations." % iteration)
    #
    #     return loss