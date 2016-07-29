import numpy as np

from MatrixBeliefPropagator import MatrixBeliefPropagator, logsumexp

class ConvexBeliefPropagator(MatrixBeliefPropagator):

    def __init__(self, markov_net, counting_numbers=None):
        super(ConvexBeliefPropagator, self).__init__(markov_net)

        self.unary_counting_numbers = np.ones(len(self.mn.variables))
        self.edge_counting_numbers = np.ones(2 * self.mn.num_edges)

        if counting_numbers:
            self._set_counting_numbers(counting_numbers)

    def _set_counting_numbers(self, counting_numbers):
        self.edge_counting_numbers = np.zeros(2 * self.mn.num_edges)

        for i, edge in enumerate(self.mn.edges):
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

        for i, edge in enumerate(self.mn.edges):
            self.unary_coefficients[edge[0]] += self.edge_counting_numbers[i]
            self.unary_coefficients[edge[1]] += self.edge_counting_numbers[i]

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

    def update_messages(self):
        """Update all messages between variables using belief division.
        Return the change in messages from previous iteration."""
        self.compute_beliefs()

        adjusted_message_prod = self.mn.edge_pot_tensor - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                                     self.message_mat[:, :self.mn.num_edges]))
        adjusted_message_prod /= self.edge_counting_numbers
        adjusted_message_prod += \
            self.mn.message_from_index.dot(np.nan_to_num(self.belief_mat.T)).T

        messages = np.squeeze(logsumexp(adjusted_message_prod, 1)) * self.edge_counting_numbers
        messages = np.nan_to_num(messages - messages.max(0))

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

    def compute_beliefs(self):
        """Compute unary beliefs based on current messages."""
        if not self.fully_conditioned:
            self.belief_mat = self.mn.unary_mat + self.conditioning_mat + \
                              self.mn.message_to_index.T.dot(self.message_mat.T).T
            self.belief_mat /= self.unary_coefficients.T
            log_z = logsumexp(self.belief_mat, 0)

            self.belief_mat = self.belief_mat - log_z

    def compute_pairwise_beliefs(self):
        """Compute pairwise beliefs based on current messages."""
        if not self.fully_conditioned:
            adjusted_message_prod = self.mn.message_from_index.dot(self.belief_mat.T).T \
                                    - np.nan_to_num(np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                    self.message_mat[:, :self.mn.num_edges])) /
                                                    self.edge_counting_numbers)

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape(
                (self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape(
                (1, self.mn.max_states, self.mn.num_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] / \
                      self.edge_counting_numbers[self.mn.num_edges:] + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs