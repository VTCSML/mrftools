"""BeliefPropagator class."""
import numpy as np

from MarkovNet import MarkovNet
from Inference import Inference


class MatrixBeliefPropagator(Inference):
    """Object that can run belief propagation on a MarkovNet."""

    def __init__(self, markov_net):
        """Initialize belief propagator for markov_net."""
        self.mn = markov_net
        self.var_beliefs = dict()
        self.pair_beliefs = dict()

        if not self.mn.matrix_mode:
            self.mn.create_matrices()

        self.previously_initialized = False
        self.initialize_messages()

        self.belief_mat = np.zeros((self.mn.max_states, len(self.mn.variables)))
        self.pair_belief_tensor = np.zeros((self.mn.max_states, self.mn.max_states, self.mn.num_edges))
        self.augmented_mat = np.zeros( (self.mn.max_states, len( self.mn.variables )) )
        self.max_iter = 300
        self.fully_conditioned = False
        self.conditioned = np.zeros(len(self.mn.variables), dtype=bool)

        self.disallow_impossible_states()

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def initialize_messages(self):
        self.message_mat = np.zeros((self.mn.max_states, 2 * self.mn.num_edges))

    def augment_loss(self, var, state):
        i = self.mn.var_index[var]
        self.augmented_mat[:, i] = 1
        self.augmented_mat[state, i] = -1

    def condition(self, var, state):
        i = self.mn.var_index[var]
        self.augmented_mat[:, i] = -np.inf
        self.augmented_mat[state, i] = 0
        if isinstance(state, int):
            self.conditioned[i] = True

        if np.all(self.conditioned):
            # compute beliefs and set flag to never recompute them
            self.compute_beliefs()
            self.compute_pairwise_beliefs()
            self.fully_conditioned = True

    def disallow_impossible_states(self):
        """ force variables to only allow nonzero probability on their possible states """
        for var, num_states in self.mn.num_states.items():
            self.condition(var, range(num_states))

    def compute_beliefs(self):
        """Compute unary beliefs based on current messages."""
        if not self.fully_conditioned:
            self.belief_mat = self.mn.unary_mat + self.augmented_mat
            self.belief_mat += sparse_dot(self.message_mat, self.mn.message_to_map)

            self.belief_mat -= logsumexp(self.belief_mat, 0)

    def compute_pairwise_beliefs(self):
        """Compute pairwise beliefs based on current messages."""
        if not self.fully_conditioned:
            adjusted_message_prod = self.belief_mat[:, self.mn.message_from] \
                                    - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                               self.message_mat[:, :self.mn.num_edges]))

            to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape((self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape((1, self.mn.max_states, self.mn.num_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs

    def update_messages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        self.compute_beliefs()

        adjusted_message_prod = self.mn.edge_pot_tensor - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                                     self.message_mat[:, :self.mn.num_edges]))
        adjusted_message_prod += self.belief_mat[:, self.mn.message_from]

        messages = np.squeeze(logsumexp(adjusted_message_prod, 1))
        messages = np.nan_to_num(messages - messages.max(0))

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

    def _compute_inconsistency_vector(self):
        expanded_beliefs = np.exp(self.belief_mat[:, self.mn.message_to])

        pairwise_beliefs = np.hstack((np.sum(np.exp(self.pair_belief_tensor), axis = 0),
                                      np.sum(np.exp(self.pair_belief_tensor), axis = 1)))

        return expanded_beliefs - pairwise_beliefs

    def compute_inconsistency(self):
        """Return the total disagreement between each unary belief and its pairwise beliefs."""
        disagreement = np.sum(np.abs(self._compute_inconsistency_vector()))

        return disagreement

    def infer(self, tolerance = 1e-8, display = 'iter'):
        """Run belief propagation until messages change less than tolerance."""
        change = np.inf
        iteration = 0
        while change > tolerance and iteration < self.max_iter:
            change = self.update_messages()
            if display == "full":
                energy_func = self.compute_energy_functional()
                disagreement = self.compute_inconsistency()
                dual_obj = self.compute_dual_objective()
                print("Iteration %d, change in messages %f. Calibration disagreement: %f, energy functional: %f, dual obj: %f" % (iteration, change, disagreement, energy_func, dual_obj))
            elif display == "iter":
                print("Iteration %d, change in messages %f." % (iteration, change))
            iteration += 1
        if display == 'final' or display == 'full' or display == 'iter':
            print("Belief propagation finished in %d iterations." % iteration)

    def load_beliefs(self):
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        for (var, i) in self.mn.var_index.items():
            self.var_beliefs[var] = self.belief_mat[:len(self.mn.unary_potentials[var]), i]

        for edge, i in self.mn.edge_index.items():
            (var, neighbor) = edge

            belief = self.pair_belief_tensor[:len(self.mn.unary_potentials[var]),
                     :len(self.mn.unary_potentials[neighbor]), i]

            self.pair_beliefs[(var, neighbor)] = belief

            self.pair_beliefs[(neighbor, var)] = belief.T

    def compute_bethe_entropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        if self.fully_conditioned:
            entropy = 0
        else:
            entropy = - np.sum(np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor)) \
                      - np.sum((1 - self.mn.degrees) * (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))

        return entropy

    def compute_energy(self):
        """Compute the log-linear energy. Assume that the beliefs have been computed and are fresh."""
        energy = np.sum(np.nan_to_num(self.mn.edge_pot_tensor[:, :, self.mn.num_edges:]) * np.exp(self.pair_belief_tensor)) + \
                 np.sum(np.nan_to_num(self.mn.unary_mat) * np.exp(self.belief_mat))

        return energy

    def get_feature_expectations(self):
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        summed_features = self.mn.feature_mat.dot(np.exp(self.belief_mat).T)

        summed_pair_features = self.mn.edge_feature_mat.dot(np.exp(self.pair_belief_tensor).reshape(
            (self.mn.max_states**2, self.mn.num_edges)).T)

        marginals = np.append(summed_features.reshape(-1), summed_pair_features.reshape(-1))

        return marginals

    def compute_energy_functional(self):
        """Compute the energy functional."""
        self.compute_beliefs()
        self.compute_pairwise_beliefs()
        return self.compute_energy() + self.compute_bethe_entropy()

    def compute_dual_objective(self):
        """Compute the value of the BP Lagrangian."""
        objective = self.compute_energy_functional() + \
                    np.sum(self.message_mat * self._compute_inconsistency_vector())

        return objective

    def set_messages(self, messages):
        assert(np.all(self.message_mat.shape == messages.shape))
        self.message_mat = messages

def logsumexp(matrix, dim = None):
    """Compute log(sum(exp(matrix), dim)) in a numerically stable way."""
    try:
        return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except FloatingPointError:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val

def sparse_dot(full_matrix, sparse_matrix):
    return sparse_matrix.T.dot(full_matrix.T).T

def make_grad_logsumexp(ans, matrix, dim):
    def gradient_product(g):
        return np.full(matrix.shape, g) * np.exp(matrix - np.full(matrix.shape, ans))
    return gradient_product

def make_grad_sparse_dot(ans, full_matrix, sparse_matrix):
    def gradient_product(g):
        #todo make this work
        return
    return gradient_product