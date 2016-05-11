"""BeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet
from scipy.sparse import dok_matrix, csc_matrix

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

        if self.previously_initialized:
            self.message_mat = np.zeros((self.mn.max_states, 2 * self.mn.num_edges))
            self.message_mat[self.message_mask.nonzero()] = -np.inf
        else:
            self.message_mat = -np.inf * np.ones((self.mn.max_states, 2 * self.mn.num_edges))

            i = 0
            for var in self.mn.variables:
                for neighbor in self.mn.neighbors[var]:
                    if var < neighbor:
                        dims = self.mn.getPotential((var, neighbor)).shape

                        self.message_mat[:dims[1], i] = 0
                        self.message_mat[:dims[0], i + self.mn.num_edges] = 0

                        i += 1

            self.message_mask = csc_matrix(np.isinf(self.message_mat))
            self.previously_initialized = True

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

        max_val = beliefs.max(axis = (0, 1), keepdims = True)

        log_partitions = np.log(np.sum(np.exp(beliefs - max_val), axis = (0, 1), keepdims = True)) + max_val

        beliefs -= log_partitions

        self.pair_belief_tensor = beliefs


    def updateMessages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        self.computeBeliefs()

        adjusted_message_prod = np.nan_to_num(self.mn.message_from_index.dot(self.belief_mat.T).T) \
                                - np.nan_to_num(np.hstack((self.message_mat[:, self.mn.num_edges:],
                                                           self.message_mat[:, :self.mn.num_edges])))

        messages = logsumexp(self.mn.edge_pot_tensor + adjusted_message_prod, 1)
        messages -= messages.max(0, keepdims = True)

        change = np.sum(np.abs(np.nan_to_num(messages) - np.nan_to_num(self.message_mat)))

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

    maxVal = matrix.max()
    with np.errstate(divide = 'ignore'):
        return np.log(np.sum(np.exp(matrix - maxVal), dim)) + maxVal

def main():
    """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet()

    np.random.seed(1)

    np.set_printoptions(precision=3)
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

    bp.initialize_messages()

    from BruteForce import BruteForce

    bf = BruteForce(mn)

    for i in mn.variables:
        print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unaryMarginal(i))))
        print ("Belief prop unary marginal of %d: %s" % (i, repr(np.exp(bp.varBeliefs[i]))))

    print ("Brute force pairwise marginal: " + repr(bf.pairwiseMarginal(0,1)))
    print ("Belief prop pairwise marginal: " + repr(np.exp(bp.pairBeliefs[(0,1)])))

    print ("Bethe energy functional: %f" % bp.computeEnergyFunctional())

    print ("Brute force log partition function: %f" % np.log(bf.computeZ()))


    from BeliefPropagator import BeliefPropagator

    bp_old = BeliefPropagator(mn)

    bp_old.runInference(display='full')

    print ("Bethe energy functional: %f" % bp_old.computeEnergyFunctional())


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
