"""BeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet
from util import logsumexp

class BeliefPropagator(object):
    """Object that can run belief propagation on a MarkovNet."""

    def __init__(self, markovNet):
        """Initialize belief propagator for markovNet."""
        self.mn = markovNet
        self.varBeliefs = dict()
        self.pairBeliefs = dict()
        self.messages = dict()
        self.initMessages()
        self.initBeliefs()

    def initMessages(self):
        """Initialize messages to default initialization (set to zeros)."""
        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                self.messages[(var, neighbor)] = - np.log(self.mn.numStates[neighbor])

    def initBeliefs(self):
        """Initialize beliefs."""
        for var in self.mn.variables:
            belief = self.mn.unaryPotentials[var]
            logZ = logsumexp(belief)
            belief = belief - logZ
            self.varBeliefs[var] = belief

        # Initialize pairwise beliefs

        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                belief = self.mn.getPotential((var, neighbor))
                logZ = logsumexp(np.sum(belief))
                belief = belief - logZ
                self.pairBeliefs[(var, neighbor)] = belief

    def computeBeliefs(self):
        """Compute unary beliefs based on current messages."""
        for var in self.mn.variables:
            belief = self.mn.unaryPotentials[var]
            for neighbor in self.mn.getNeighbors(var):
                belief = belief + self.messages[(neighbor, var)]
            logZ = logsumexp(belief)
            belief = belief - logZ
            self.varBeliefs[var] = belief

    def computePairwiseBeliefs(self):
        """Compute pairwise beliefs based on current messages."""
        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                if var < neighbor:
                    belief = self.mn.getPotential((var, neighbor))

                    # compute product of all messages to var except from neighbor
                    varMessageProduct = self.varBeliefs[var] - self.messages[(neighbor, var)]
                    belief = (belief.T + varMessageProduct).T

                    # compute product of all messages to neighbor except from var
                    neighborMessageProduct = self.varBeliefs[neighbor] - self.messages[(var, neighbor)]
                    belief = belief + neighborMessageProduct

                    logZ = logsumexp(belief)
                    belief = belief - logZ
                    self.pairBeliefs[(var, neighbor)] = belief
                    self.pairBeliefs[(neighbor, var)] = belief.T


    def computeMessage(self, var, neighbor):
        """Compute the message from var to factor."""
        # compute the product of all messages coming into var except the one from neighbor
        adjustedMessageProduct = self.varBeliefs[var] - self.messages[(neighbor, var)]

        # sum over all states of var
        message = logsumexp((self.mn.getPotential((neighbor, var)) + adjustedMessageProduct).T, 0).T

        # normalize message
        message = message - logsumexp(message)

        return message

    def updateMessages(self):
        """Update all messages between variables using belief division. Return the change in messages from previous iteration."""
        change = 0.0
        self.computeBeliefs()
        newMessages = dict()
        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                newMessages[(var, neighbor)] = self.computeMessage(var, neighbor)
                change += np.sum(np.abs(newMessages[(var, neighbor)] - self.messages[(var, neighbor)]))
        self.messages = newMessages

        return change

    def computeInconsistency(self):
        """Return the total disagreement between each unary belief and its pairwise beliefs."""
        disagreement = 0.0
        self.computeBeliefs()
        self.computePairwiseBeliefs()
        for var in self.mn.variables:
            unaryBelief = np.exp(self.varBeliefs[var])
            for neighbor in self.mn.getNeighbors(var):
                pairBelief = np.sum(np.exp(self.pairBeliefs[(var, neighbor)]), 1)
                disagreement += np.sum(np.abs(unaryBelief - pairBelief))
        return disagreement

    def runInference(self, tolerance = 1e-8, display = 'iter'):
        """Run belief propagation until messages change less than tolerance."""
        change = np.inf

        iteration = 0
        while change > tolerance:
            change = self.updateMessages()
            if display == "iter":
                disagreement = self.computeInconsistency()
                energyFunc = self.computeEnergyFunctional()
                dualObj = self.computeDualObjective()
                print("Iteration %d, change in messages %f. Calibration disagreement: %f, energy functional: %f, dual obj: %f" % (iteration, change, disagreement, energyFunc, dualObj))
            iteration += 1

    def computeBetheEntropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        entropy = 0.0

        for var in self.mn.variables:
            neighbors = self.mn.getNeighbors(var)
            entropy += -(1 - len(neighbors)) * np.sum(np.exp(self.varBeliefs[var]) * self.varBeliefs[var])
            for neighbor in neighbors:
                if var < neighbor:
                    entropy += -np.sum(np.exp(self.pairBeliefs[(var, neighbor)]) * self.pairBeliefs[(var, neighbor)])
        return entropy

    def computeEnergy(self):
        """Compute the log-linear energy. Assume that the beliefs have been computed and are fresh."""
        energy = 0.0

        for var in self.mn.variables:
            neighbors = self.mn.getNeighbors(var)
            energy += np.sum(self.mn.unaryPotentials[var] * self.varBeliefs[var])
            for neighbor in neighbors:
                if var < neighbor:
                    energy += np.sum(self.mn.getPotential((var, neighbor)) * self.pairBeliefs[(var, neighbor)])
        return energy

    def computeEnergyFunctional(self):
        """Compute the energy functional."""
        self.computeBeliefs()
        self.computePairwiseBeliefs()
        return self.computeEnergy() + self.computeBetheEntropy()

    def computeDualObjective(self):
        """Compute the value of the BP Lagrangian."""
        objective = self.computeEnergyFunctional()
        for var in self.mn.variables:
            unaryBelief = np.exp(self.varBeliefs[var])
            for neighbor in self.mn.getNeighbors(var):
                pairBelief = np.sum(np.exp(self.pairBeliefs[(var, neighbor)]), 1)
                objective += self.messages[(neighbor, var)].dot(pairBelief - unaryBelief)
        return objective

def main():
    """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet()

    np.random.seed(1)

    mn.setUnaryFactor(0, np.random.randn(4))
    mn.setUnaryFactor(1, np.random.randn(3))
    mn.setUnaryFactor(2, np.random.randn(6))
    mn.setUnaryFactor(3, np.random.randn(2))

    mn.setEdgeFactor((0,1), np.random.randn(4,3))
    mn.setEdgeFactor((1,2), np.random.randn(3,6))
    mn.setEdgeFactor((3,2), np.random.randn(2,6))
    # mn.setEdgeFactor((3,0), np.random.randn(2,4)) # uncomment this to make loopy

    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))

    bp = BeliefPropagator(mn)

    # for t in range(15):
    #     change = bp.updateMessages()
    #     disagreement = bp.computeInconsistency()
    #     print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))

    bp.runInference()


    bp.computePairwiseBeliefs()

    from BruteForce import BruteForce

    bf = BruteForce(mn)

    for i in range(2):
        print "Brute force unary marginal of %d: %s" % (i, repr(bf.unaryMarginal(i)))
        print "Belief prop unary marginal of %d: %s" % (i, repr(bf.unaryMarginal(i)))

    print "Brute force pairwise marginal: " + repr(bf.pairwiseMarginal(0,1))
    print "Belief prop pairwise marginal: " + repr(np.exp(bp.pairBeliefs[(0,1)]))

    print "Bethe energy functional: %f" % bp.computeEnergyFunctional()




if  __name__ =='__main__':
    main()
