import numpy as np
from MaxProductBeliefPropagator import MaxProductBeliefPropagator

from MatrixBeliefPropagator import MatrixBeliefPropagator, sparse_dot, logsumexp


class MaxProductLinearProgramming(MaxProductBeliefPropagator):
    def __init__(self, markov_net):
        super(MaxProductLinearProgramming, self).__init__(markov_net)

    def update_messages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        message_sum = sparse_dot(self.message_mat, self.mn.message_to_map)

        belief_mat = self.mn.unary_mat + self.augmented_mat
        belief_mat += message_sum

        adjusted_message_prod = self.mn.edge_pot_tensor - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                                     self.message_mat[:, :self.mn.num_edges]))
        adjusted_message_prod += belief_mat[:, self.mn.message_from]

        incoming_messages = np.squeeze(adjusted_message_prod.max(1))

        outgoing_messages = message_sum[:, self.mn.message_to] - self.message_mat
        messages = 0.5 * np.nan_to_num(incoming_messages - outgoing_messages)

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

