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
        self.conditioning_mat = np.zeros((self.mn.max_states, len(self.mn.variables)))

    def initialize_messages(self):
        self.message_mat = np.zeros((self.mn.max_states, 2 * self.mn.num_edges))

    def condition(self,states):
        """computer condition_mat for mode q"""
        label_mask = -float('inf') * np.ones((self.mn.max_states, len(self.mn.variables)))
        for (var, i) in self.mn.var_index.items():
            if states[var] != -100:
                label_mask[states[var], i] = 0
            else:
                label_mask[:, i] = 0
        
        self.conditioning_mat = label_mask

    def condition_state(self):
        self.mn.unary_mat += self.conditioning_mat

    def compute_beliefs(self):
        """Compute unary beliefs based on current messages."""
        self.condition_state()
        self.belief_mat = self.mn.unary_mat + self.mn.message_to_index.T.dot(self.message_mat.T).T
        log_z = logsumexp(self.belief_mat, 0)

        self.belief_mat = self.belief_mat - log_z


    def compute_pairwise_beliefs(self):
        """Compute pairwise beliefs based on current messages."""

        adjusted_message_prod = self.mn.message_from_index.dot(self.belief_mat.T).T \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                           self.message_mat[:, :self.mn.num_edges]))

        to_messages = adjusted_message_prod[:, :self.mn.num_edges].reshape((self.mn.max_states, 1, self.mn.num_edges))
        from_messages = adjusted_message_prod[:, self.mn.num_edges:].reshape((1, self.mn.max_states, self.mn.num_edges))

        beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] + to_messages + from_messages

        max_val = beliefs.max(axis = (0, 1), keepdims=True)

        log_partitions = np.log(np.sum(np.exp(beliefs - max_val), axis = (0, 1), keepdims=True)) + max_val

        beliefs -= log_partitions

        self.pair_belief_tensor = beliefs


    def update_messages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        self.compute_beliefs()

        adjusted_message_prod = self.mn.message_from_index.dot(self.belief_mat.T).T \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                           self.message_mat[:, :self.mn.num_edges]))

        messages = np.squeeze(logsumexp(self.mn.edge_pot_tensor + adjusted_message_prod, 1))
        messages = np.nan_to_num(messages - messages.max(0))

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = messages

        return change

    def _compute_inconsistency_vector(self):

        expanded_beliefs = np.exp(self.mn.message_to_index.dot(self.belief_mat.T).T)

        pairwise_beliefs = np.hstack((np.sum(np.exp(self.pair_belief_tensor), axis = 0),
                                      np.sum(np.exp(self.pair_belief_tensor), axis = 1)))

        return expanded_beliefs - pairwise_beliefs


    def compute_inconsistency(self):
        """Return the total disagreement between each unary belief and its pairwise beliefs."""
        disagreement = np.sum(np.abs(self._compute_inconsistency_vector()))

        return disagreement

    def infer(self, tolerance = 1e-8, display = 'iter', max_iter = 300):
        """Run belief propagation until messages change less than tolerance."""
        change = np.inf
        iteration = 0
        while change > tolerance and iteration < max_iter:
            change = self.update_messages()
            if display == "full":
                disagreement = self.compute_inconsistency()
                energy_func = self.compute_energy_functional()
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
            self.var_beliefs[var] = self.belief_mat[:len(self.mn.unaryPotentials[var]), i]

        for i in range(self.mn.num_edges):
            (var, neighbor) = self.mn.edges[i]

            belief = self.pair_belief_tensor[:len(self.mn.unaryPotentials[var]),
                     :len(self.mn.unaryPotentials[neighbor]), i]

            self.pair_beliefs[(var, neighbor)] = belief

            self.pair_beliefs[(neighbor, var)] = belief.T


    def compute_bethe_entropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        entropy = - np.sum(np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor)) \
                  - np.sum((1 - self.mn.degrees) * (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))

        return entropy

    def compute_energy(self):
        """Compute the log-linear energy. Assume that the beliefs have been computed and are fresh."""
        energy = np.sum(np.nan_to_num(self.mn.edge_pot_tensor[:, :, self.mn.num_edges:]) * np.exp(self.pair_belief_tensor)) + \
                 np.sum(np.nan_to_num(self.mn.unary_mat) * np.exp(self.belief_mat))

        return energy

    def get_feature_expectations(self):
        self.infer(display='off')
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        summed_features = np.inner(np.exp(self.belief_mat), self.mn.feature_mat).T

        summed_pair_features = np.sum(np.exp(self.pair_belief_tensor), 2).T

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

    if matrix.size <= 1:
        return matrix

    max_val = np.nan_to_num(matrix.max())
    with np.errstate(divide='ignore'):
        return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val

def main():
    """Test basic functionality of BeliefPropagator."""
    from BeliefPropagator import BeliefPropagator

    mn = MarkovNet()

    np.random.seed(1)

    np.set_printoptions(precision=5)
    # np.seterr(all='raise')

    k = [2, 2, 3, 3, 3]

    mn.set_unary_factor(0, np.random.randn(k[0]))
    mn.set_unary_factor(1, np.random.randn(k[1]))
    mn.set_unary_factor(2, np.random.randn(k[2]))
    mn.set_unary_factor(3, np.random.randn(k[3]))
    mn.set_unary_factor(4, np.random.randn(k[3]))

    # factor4 = np.random.randn(k[4])
    # factor4[2] = -float('inf')

    # mn.set_unary_factor(4, factor4)

    mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
    mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
    mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
    mn.set_edge_factor((3, 4), np.random.randn(k[3], k[4]))
    # mn.set_edge_factor((1,4), np.random.randn(k[1], k[4]))
    # mn.set_edge_factor((3,0), np.random.randn(k[3], k[0])) # uncomment this to make loopy

    print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
    print("Neighbors of 1: " + repr(mn.get_neighbors(1)))

    bp = MatrixBeliefPropagator(mn)

    # for t in range(15):
    #     change = bp.update_messages()
    #     disagreement = bp.compute_inconsistency()
    #     print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))

    bp.infer(display='full')

    bp.compute_pairwise_beliefs()

    bp.load_beliefs()

    from BruteForce import BruteForce

    bf = BruteForce(mn)

    unary_error = 0

    for i in mn.variables:
        bf_marg = bf.unary_marginal(i)
        bp_marg = np.exp(bp.var_beliefs[i])

        unary_error += np.sum(np.abs(bf_marg - bp_marg))

        print ("Brute force unary marginal of %d: %s" % (i, repr(bf_marg)))
        print ("Belief prop unary marginal of %d: %s" % (i, repr(bp_marg)))

    pairwise_error = 0.0

    for var in mn.variables:
        for neighbor in mn.get_neighbors(var):
            edge = (var, neighbor)
            bf_marg = bf.pairwise_marginal(var, neighbor)
            bp_marg = np.exp(bp.pair_beliefs[edge])

            pairwise_error += np.sum(np.abs(bf_marg - bp_marg))

            print ("Brute force pairwise marginal of %s: %s" % (repr(edge), repr(bf_marg)))
            print ("Belief prop pairwise marginal of %s: %s" % (repr(edge), repr(bp_marg)))

    print ("Unary error %s, pairwise error %s" % (unary_error, pairwise_error))

    print ("Bethe energy functional: %f" % bp.compute_energy_functional())

    print ("Brute force log partition function: %f" % np.log(bf.compute_z()))


    print ("Brute force entropy: %f" % np.log(bf.entropy()))
    print ("Bethe entropy: %f" % bp.compute_bethe_entropy())
    print ("Belief prop energy: %f" % bp.compute_energy())

    bp_old = BeliefPropagator(mn)

    bp_old.runInference(display='full')

    print ("Old Bethe energy functional: %f" % bp_old.computeEnergyFunctional())
    print ("Old Bethe entropy: %f" % bp_old.compute_bethe_entropy())
    print ("Old belief prop energy: %f" % bp_old.compute_energy())

    print("Running grid timing comparison to loop BP")

    mn = MarkovNet()

    length = 32

    k = 8

    for x in range(length):
        for y in range(length):
            mn.set_unary_factor((x, y), np.random.random(k))

    for x in range(length - 1):
        for y in range(length):
            mn.set_edge_factor(((x, y), (x + 1, y)), np.random.random((k, k)))
            mn.set_edge_factor(((y, x), (y, x + 1)), np.random.random((k, k)))

    log_bp = BeliefPropagator(mn)

    bp = MatrixBeliefPropagator(mn)

    import time

    t0 = time.time()
    bp.infer(display='final', max_iter=30000)
    t1 = time.time()

    bp_time = t1 - t0

    t0 = time.time()
    log_bp.runInference(display='final', maxIter=30000)
    t1 = time.time()

    log_bp_time = t1 - t0

    print("Matrix BP took %f, loop-based BP took %f. Speedup was %f" % (bp_time, log_bp_time, log_bp_time / bp_time))




if  __name__ =='__main__':
    main()
