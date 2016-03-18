"""Markov network class for storing potential functions and structure."""
import numpy as np

class MarkovNet(object):
    """Object containing the definition of a Markov net."""

    def __init__(self):
        """Initialize a Markov net."""
        self.edgePotentials = dict()
        self.unaryPotentials = dict()
        self.neighbors = dict()
        self.variables = set()
        self.numStates = dict()

    def setUnaryFactor(self, variable, potential):
        """Set the potential function for the unary factor. Implicitly declare variable. Must be called before setting edge factors."""
        self.unaryPotentials[variable] = potential
        self.variables.add(variable)
        self.neighbors[variable] = set()
        self.numStates[variable] = np.size(potential)

    def setEdgeFactor(self, edge, potential):
        """Set a factor by inputting the involved variables then the potential function. The potential function should be a np matrix."""
        assert np.shape(potential) == (len(self.unaryPotentials[edge[0]]), len(self.unaryPotentials[edge[1]])), "potential size %d, %d incompatible with unary sizes %d, %d" % (np.shape(potential)[0], np.shape(potential)[1], len(self.unaryPotentials[edge[0]]), len(self.unaryPotentials[edge[1]]))

        self.edgePotentials[edge] = potential

        self.neighbors[edge[0]].add(edge[1])
        self.neighbors[edge[1]].add(edge[0])

    def getPotential(self, variables):
        """Return the potential between pair[0] and pair[1]. If (pair[1], pair[0]) is in our dictionary instead, return the transposed potential."""
        if len(variables) == 1:
            return self.unaryPotentials[variables]
        else:
            if variables in self.edgePotentials:
                return self.edgePotentials[variables]
            else:
                return self.edgePotentials[(variables[1], variables[0])].T

    def getNeighbors(self, variable):
        """Return the neighbors of variable."""
        return self.neighbors[variable]

def main():
    """Test function for MarkovNet."""
    mn = MarkovNet()

    mn.setUnaryFactor(0, np.random.rand(4))
    mn.setUnaryFactor(1, np.random.rand(3))
    mn.setUnaryFactor(2, np.random.rand(5))

    mn.setEdgeFactor((0,1), np.random.rand(4,3))
    mn.setEdgeFactor((1,2), np.random.rand(3,5))

    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))
    print("Neighbors of 2: " + repr(mn.getNeighbors(2)))

if  __name__ =='__main__':
    main()
