"""BeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet

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
                self.messages[(var, neighbor)] = np.ones(( self.mn.numStates[neighbor]))

    def initBeliefs(self):
        """Initialize beliefs."""
        for var in self.mn.variables:
            belief = self.mn.unaryPotentials[var]
            Z = np.sum(belief)
            belief = belief / Z
            self.varBeliefs[var] = belief

        # Initialize pairwise beliefs

        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                belief = self.mn.getPotential((var, neighbor))
                Z = np.sum(np.sum(belief))
                belief = belief / Z
                self.pairBeliefs[(var, neighbor)] = belief

    def computeBeliefs(self):
        """Compute unary beliefs based on current messages."""
        for var in self.mn.variables:
            belief = self.mn.unaryPotentials[var]
            for neighbor in self.mn.getNeighbors(var):
                belief = belief * self.messages[(neighbor, var)]
            Z = np.sum(belief)
            belief = belief / Z
            self.varBeliefs[var] = belief

    def computePairwiseBeliefs(self):
        """Compute pairwise beliefs based on current messages."""
        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                if var < neighbor:
                    belief = self.mn.getPotential((var, neighbor))
                    varMessages = self.varBeliefs[var] / self.messages[(neighbor, var)]
                    belief = (belief.T * varMessages).T
                    neighborMessages = self.varBeliefs[neighbor] / self.messages[(var, neighbor)]
                    belief = belief * neighborMessages

                    Z = np.sum(np.sum(belief))
                    belief = belief / Z
                    self.pairBeliefs[(var, neighbor)] = belief
                    self.pairBeliefs[(neighbor, var)] = belief.T


    def computeMessage(self, var, neighbor):
        """Compute the message from var to factor."""
        adjustedMessageProduct = self.varBeliefs[var] / self.messages[(neighbor, var)]

        message = self.mn.getPotential((neighbor, var)).dot(adjustedMessageProduct)
        # normalize message
        message = message / np.sum(message)

        # assert np.shape(self.messages[(var, neighbor)]) == np.shape(message)

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
        for var in self.mn.variables:
            unaryBelief = self.varBeliefs[var]
            for neighbor in self.mn.getNeighbors(var):
                pairBelief = np.sum(self.pairBeliefs[(var, neighbor)], 1)
                disagreement += np.sum(np.abs(unaryBelief - pairBelief))
        return disagreement

def main():
    """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet()

    mn.setUnaryFactor(0, np.random.rand(4))
    mn.setUnaryFactor(1, np.random.rand(3))
    # mn.setUnaryFactor(2, np.random.rand(5))
    # mn.setUnaryFactor(3, np.random.rand(6))

    mn.setEdgeFactor((0,1), np.random.rand(4,3))
    # mn.setEdgeFactor((1,2), np.random.rand(3,5))
    # mn.setEdgeFactor((3,2), np.random.rand(6,5))

    # close the loop
    # mn.setEdgeFactor((3,0), np.random.rand(6,4))

    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))
    # print("Neighbors of 2: " + repr(mn.getNeighbors(2)))

    bp = BeliefPropagator(mn)

    for t in range(10):
        change = bp.updateMessages()
        disagreement = bp.computeInconsistency()
        print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))
    bp.computePairwiseBeliefs()


if  __name__ =='__main__':
    main()
