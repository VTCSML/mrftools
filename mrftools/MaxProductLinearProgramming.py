import numpy as np
from mrftools import *

from MatrixBeliefPropagator import MatrixBeliefPropagator, sparse_dot, logsumexp


class MaxProductLinearProgramming(MaxProductBeliefPropagator):
    def __init__(self, markov_net):
        super(MaxProductLinearProgramming, self).__init__(markov_net)


    def update_messages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        belief_mat = self.mn.unary_mat + self.conditioning_mat
        belief_mat += sparse_dot(self.message_mat, self.mn.message_to_map)

        belief_mat -= logsumexp(belief_mat, 0)

        adjusted_message_prod = self.mn.edge_pot_tensor - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                                     self.message_mat[:, :self.mn.num_edges]))
        adjusted_message_prod += belief_mat[:, self.mn.message_from]

        messages = np.squeeze(adjusted_message_prod.max(1))

        messages = 0.5 * messages
        term_2 = -0.5 * (belief_mat[:, self.mn.message_to] - self.message_mat)
        messages = messages + term_2

        messages = np.nan_to_num(messages - messages.max(0))

        # print messages

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change


def create_chain_model():
    """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet ( )

    np.random.seed ( 1 )

    k = [3, 3, 3, 3, 3]

    mn.set_unary_factor ( 0, np.random.randn ( k[0] ) )
    mn.set_unary_factor ( 1, np.random.randn ( k[1] ) )
    mn.set_unary_factor ( 2, np.random.randn ( k[2] ) )
    mn.set_unary_factor ( 3, np.random.randn ( k[3] ) )

    factor4 = np.random.randn ( k[4] )
    factor4[2] = -float ( 'inf' )

    mn.set_unary_factor ( 4, factor4 )

    mn.set_edge_factor ( (0, 1), np.random.randn ( k[0], k[1] ) )
    mn.set_edge_factor ( (1, 2), np.random.randn ( k[1], k[2] ) )
    mn.set_edge_factor ( (2, 3), np.random.randn ( k[2], k[3] ) )
    mn.set_edge_factor ( (3, 4), np.random.randn ( k[3], k[4] ) )
    mn.create_matrices ( )

    return mn


def main():
    mn = create_chain_model ( )
    bp = MaxProductLinearProgramming( mn )
    bp.infer ( display='full' )
    bp.load_beliefs ( )




if __name__ == "__main__":
    main()