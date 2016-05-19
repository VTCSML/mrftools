"""BeliefPropagator class."""
import numpy as np

from MarkovNet import MarkovNet


class MatrixBeliefPropagator(object):
    """Object that can run belief propagation on a MarkovNet."""

    def __init__(self, markovNet):
        """Initialize belief propagator for markovNet."""
        self.mn = markovNet
        self.varBeliefs = dict()
        self.pairBeliefs = dict()

        if not self.mn.matrix_mode:
            self.mn.create_matrices()

        self.previously_initialized = False
        self.initialize_messages()

        self.belief_mat = np.zeros((self.mn.max_states, len(self.mn.variables)))
        self.pair_belief_tensor = np.zeros((self.mn.max_states, self.mn.max_states, self.mn.num_edges))

    def initialize_messages(self):
        self.message_mat = np.zeros((self.mn.max_states, 2 * self.mn.num_edges))

    def computeBeliefs(self):
        """Compute unary beliefs based on current messages."""
        self.belief_mat = self.mn.unary_mat + self.mn.message_to_index.T.dot(self.message_mat.T).T
        logZ = logsumexp(self.belief_mat, 0)

        self.belief_mat = self.belief_mat - logZ


    def computePairwiseBeliefs(self):
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


    def updateMessages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        self.computeBeliefs()

        adjusted_message_prod = self.mn.message_from_index.dot(self.belief_mat.T).T \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                           self.message_mat[:, :self.mn.num_edges]))

        messages = np.squeeze(logsumexp(np.nan_to_num(self.mn.edge_pot_tensor + adjusted_message_prod), 1))
        messages -= messages.max(0)

        change = np.sum(np.abs(messages - self.message_mat))

        self.message_mat = np.nan_to_num(messages)

        return change

    def _compute_inconsistency_vector(self):

        expanded_beliefs = np.exp(self.mn.message_to_index.dot(self.belief_mat.T).T)

        pairwise_beliefs = np.hstack((np.sum(np.exp(self.pair_belief_tensor), axis = 0),
                                      np.sum(np.exp(self.pair_belief_tensor), axis = 1)))

        return expanded_beliefs - pairwise_beliefs


    def computeInconsistency(self):
        """Return the total disagreement between each unary belief and its pairwise beliefs."""
        disagreement = np.sum(np.abs(self._compute_inconsistency_vector()))

        return disagreement

    def runInference(self, tolerance = 1e-8, display = 'iter', maxIter = 300):
        """Run belief propagation until messages change less than tolerance."""
        change = np.inf
        iteration = 0
        while change > tolerance and iteration < maxIter:
            change = self.updateMessages()
            if display == "full":
                disagreement = self.computeInconsistency()
                energyFunc = self.computeEnergyFunctional()
                dualObj = self.computeDualObjective()
                print("Iteration %d, change in messages %f. Calibration disagreement: %f, energy functional: %f, dual obj: %f" % (iteration, change, disagreement, energyFunc, dualObj))
            elif display == "iter":
                print("Iteration %d, change in messages %f." % (iteration, change))
            iteration += 1
        if display == 'final' or display == 'full' or display == 'iter':
            print("Belief propagation finished in %d iterations." % iteration)

    def load_beliefs(self):

        for (var, i) in self.mn.var_index.items():
            self.varBeliefs[var] = self.belief_mat[:len(self.mn.unaryPotentials[var]), i]

        for i in range(self.mn.num_edges):
            (var, neighbor) = self.mn.edges[i]

            belief = self.pair_belief_tensor[:len(self.mn.unaryPotentials[var]),
                     :len(self.mn.unaryPotentials[neighbor]), i]

            self.pairBeliefs[(var, neighbor)] = belief

            self.pairBeliefs[(neighbor, var)] = belief.T

    def computeBetheEntropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        entropy = - np.sum(np.nan_to_num(self.pair_belief_tensor) * np.exp(self.pair_belief_tensor)) \
                  - np.sum((1 - self.mn.degrees) * (np.nan_to_num(self.belief_mat) * np.exp(self.belief_mat)))

        return entropy

    def computeEnergy(self):
        """Compute the log-linear energy. Assume that the beliefs have been computed and are fresh."""
        energy = np.sum(np.nan_to_num(self.mn.edge_pot_tensor[:, :, self.mn.num_edges:]) * np.exp(self.pair_belief_tensor)) + \
                 np.sum(np.nan_to_num(self.mn.unary_mat) * np.exp(self.belief_mat))

        return energy

    def computeEnergyFunctional(self):
        """Compute the energy functional."""
        self.computeBeliefs()
        self.computePairwiseBeliefs()
        return self.computeEnergy() + self.computeBetheEntropy()

    def computeDualObjective(self):
        """Compute the value of the BP Lagrangian."""
        objective = self.computeEnergyFunctional() + \
                    np.sum(self.message_mat * self._compute_inconsistency_vector())

        return objective

def logsumexp(matrix, dim = None):
    """Compute log(sum(exp(matrix), dim)) in a numerically stable way."""

    if matrix.size <= 1:
        return matrix

    maxVal = matrix.max(dim, keepdims=True)
    return np.log(np.sum(np.exp(matrix - maxVal), dim, keepdims=True)) + maxVal

def main():
    """Test basic functionality of BeliefPropagator."""
    from BeliefPropagator import BeliefPropagator

    mn = MarkovNet()

    np.random.seed(1)

    np.set_printoptions(precision=5)
    # np.seterr(all='raise')

    k = [2, 2, 3, 3, 3]

    mn.setUnaryFactor(0, np.random.randn(k[0]))
    mn.setUnaryFactor(1, np.random.randn(k[1]))
    mn.setUnaryFactor(2, np.random.randn(k[2]))
    mn.setUnaryFactor(3, np.random.randn(k[3]))
    mn.setUnaryFactor(4, np.random.randn(k[3]))

    # factor4 = np.random.randn(k[4])
    # factor4[2] = -float('inf')

    # mn.setUnaryFactor(4, factor4)

    mn.setEdgeFactor((0,1), np.random.randn(k[0], k[1]))
    mn.setEdgeFactor((1,2), np.random.randn(k[1], k[2]))
    mn.setEdgeFactor((3,2), np.random.randn(k[3], k[2]))
    mn.setEdgeFactor((3,4), np.random.randn(k[3], k[4]))
    # mn.setEdgeFactor((1,4), np.random.randn(k[1], k[4]))
    # mn.setEdgeFactor((3,0), np.random.randn(k[3], k[0])) # uncomment this to make loopy

    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))

    bp = MatrixBeliefPropagator(mn)

    # for t in range(15):
    #     change = bp.updateMessages()
    #     disagreement = bp.computeInconsistency()
    #     print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))

    bp.runInference(display='full')

    bp.computePairwiseBeliefs()

    bp.load_beliefs()

    from BruteForce import BruteForce

    bf = BruteForce(mn)

    unary_error = 0

    for i in mn.variables:
        bf_marg = bf.unaryMarginal(i)
        bp_marg = np.exp(bp.varBeliefs[i])

        unary_error += np.sum(np.abs(bf_marg - bp_marg))

        print ("Brute force unary marginal of %d: %s" % (i, repr(bf_marg)))
        print ("Belief prop unary marginal of %d: %s" % (i, repr(bp_marg)))

    pairwise_error = 0.0

    for var in mn.variables:
        for neighbor in mn.getNeighbors(var):
            edge = (var, neighbor)
            bf_marg = bf.pairwiseMarginal(var, neighbor)
            bp_marg = np.exp(bp.pairBeliefs[edge])

            pairwise_error += np.sum(np.abs(bf_marg - bp_marg))

            print ("Brute force pairwise marginal of %s: %s" % (repr(edge), repr(bf_marg)))
            print ("Belief prop pairwise marginal of %s: %s" % (repr(edge), repr(bp_marg)))

    print ("Unary error %s, pairwise error %s" % (unary_error, pairwise_error))

    print ("Bethe energy functional: %f" % bp.computeEnergyFunctional())

    print ("Brute force log partition function: %f" % np.log(bf.computeZ()))


    print ("Brute force entropy: %f" % np.log(bf.entropy()))
    print ("Bethe entropy: %f" % bp.computeBetheEntropy())
    print ("Belief prop energy: %f" % bp.computeEnergy())

    bp_old = BeliefPropagator(mn)

    bp_old.runInference(display='full')

    print ("Old Bethe energy functional: %f" % bp_old.computeEnergyFunctional())
    print ("Old Bethe entropy: %f" % bp_old.computeBetheEntropy())
    print ("Old belief prop energy: %f" % bp_old.computeEnergy())

    print("Running grid timing comparison to loop BP")

    mn = MarkovNet()

    length = 32

    k = 8

    for x in range(length):
        for y in range(length):
            mn.setUnaryFactor((x, y), np.random.random(k))

    for x in range(length - 1):
        for y in range(length):
            mn.setEdgeFactor(((x, y), (x + 1, y)), np.random.random((k, k)))
            mn.setEdgeFactor(((y, x), (y, x + 1)), np.random.random((k, k)))

    log_bp = BeliefPropagator(mn)

    bp = MatrixBeliefPropagator(mn)

    import time

    t0 = time.time()
    bp.runInference(display='final', maxIter=30000)
    t1 = time.time()

    bp_time = t1 - t0

    t0 = time.time()
    log_bp.runInference(display='final', maxIter=30000)
    t1 = time.time()

    log_bp_time = t1 - t0

    print("Matrix BP took %f, loop-based BP took %f. Speedup was %f" % (bp_time, log_bp_time, log_bp_time / bp_time))




if  __name__ =='__main__':
    main()
