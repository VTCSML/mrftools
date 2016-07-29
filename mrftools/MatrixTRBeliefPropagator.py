import numpy as np

from MatrixBeliefPropagator import MatrixBeliefPropagator, logsumexp

class MatrixTRBeliefPropagator(MatrixBeliefPropagator):

    def __init__(self, markov_net, tree_probabilities=None):
        super(MatrixTRBeliefPropagator, self).__init__(markov_net)

        if tree_probabilities:
            self._set_tree_probabilities(tree_probabilities)

    def _set_tree_probabilities(self, tree_probabilities):
        self.tree_probabilities = np.zeros(2 * self.mn.num_edges)

        for i, edge in enumerate(self.mn.edges):
            reversed_edge = edge[::-1]
            if edge in tree_probabilities:
                self.tree_probabilities[i] = tree_probabilities[edge]
                self.tree_probabilities[i + self.mn.num_edges] = tree_probabilities[edge]
            elif reversed_edge in tree_probabilities:
                self.tree_probabilities[i] = tree_probabilities[reversed_edge]
                self.tree_probabilities[i + self.mn.num_edges] = tree_probabilities[reversed_edge]
            else:
                raise KeyError('Edge %s was not assigned a probability.' % repr(edge))

        self.expected_degrees = self.mn.message_to_index.T.dot(self.tree_probabilities)

    def compute_bethe_entropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        if self.fully_conditioned:
            entropy = 0
        else:
            entropy = - np.sum(self.tree_probabilities[:self.mn.num_edges] *
                               np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor)) \
                      - np.sum((1 - self.expected_degrees) * (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))

        return entropy

    def update_messages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        self.compute_beliefs()

        adjusted_message_prod = self.mn.message_from_index.dot(self.belief_mat.T).T \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                             self.message_mat[:, :self.mn.num_edges]))

        messages = np.squeeze(logsumexp(self.mn.edge_pot_tensor / self.tree_probabilities + adjusted_message_prod, 1))
        messages = np.nan_to_num(messages - messages.max(0))

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

    def compute_beliefs(self):
        """Compute unary beliefs based on current messages."""
        if not self.fully_conditioned:
            self.belief_mat = self.mn.unary_mat + self.conditioning_mat + self.mn.message_to_index.T.dot(
                (self.message_mat * self.tree_probabilities).T).T
            log_z = logsumexp(self.belief_mat, 0)

            self.belief_mat = self.belief_mat - log_z

    def compute_pairwise_beliefs(self):
        """Compute pairwise beliefs based on current messages."""
        if not self.fully_conditioned:
            adjusted_message_prod = self.mn.message_from_index.dot(self.belief_mat.T).T \
                                    - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                 self.message_mat[:, :self.mn.num_edges]))

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape((self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape((1, self.mn.max_states, self.mn.num_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] / self.tree_probabilities[self.mn.num_edges:] \
                      + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs