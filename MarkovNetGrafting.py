"""Markov network class for storing potential functions and structure."""
import numpy as np
from MarkovNet import MarkovNet


class MarkovNetGrafting(MarkovNet):
    """Object containing the definition of a pairwise Markov net."""

    def __init__(self):
        """Initialize a Markov net for grafting."""
        MarkovNet.__init__(self)
        self.numWeights = 0
        self.mapFeaturesToSpace = []
        self.space = dict()
        self.searchSpace = []
        self.numWeightsSearchSpace = 0


    def initCandidateEdges(self):
        """Initialize the set of all possible Edges"""
        for var1 in self.variables:
            self.space[var1] = self.unaryPotentials[var1]
            for var2 in self.variables:
                if var1 < var2:
                    self.edgePotentials[(var1, var2)] = np.zeros(shape=(len(self.unaryPotentials[var1]),len(self.unaryPotentials[var2])))
                    self.space[(var1, var2)] = np.zeros(shape=(len(self.unaryPotentials[var1]),len(self.unaryPotentials[var2])))
                    self.searchSpace.append((var1, var2))
                    #self.numWeights += len(self.unaryPotentials[var1]) * len(self.unaryPotentials[var2])
                if var1 != var2:
                    self.neighbors[var1].add(var2)
        #print(self.edgePotentials)

    def initSearchSpace(self):
        """Initialize the set of all possible Edges"""
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 < var2:
                    self.searchSpace.append((var1, var2))
                    self.numWeightsSearchSpace += len(self.unaryPotentials[var1]) * len(self.unaryPotentials[var2])

    def updateSearchSpace(self, edge):
        self.searchSpace.remove(edge)
        self.numWeightsSearchSpace -= len(self.unaryPotentials[edge[0]]) * len(self.unaryPotentials[edge[1]])


    def initWeights(self):
        """Initialize the set of all possible Weights"""
        for x in self.space:
            pot = self.space[x]
            if isinstance(x, tuple):
                curr_num_weights = np.prod(pot.shape)
            else:
                curr_num_weights = len(pot)
            self.numWeights += curr_num_weights
            for t in range(curr_num_weights):
                self.mapFeaturesToSpace.append(x)
        return(np.zeros(self.numWeights))

def main():
    """Test function for MarkovNet."""
    mn = MarkovNetGrafting()

    mn.setUnaryFactor(0, np.random.randn(4))
    mn.setUnaryFactor(1, np.random.randn(3))
    mn.setUnaryFactor(2, np.random.randn(5))

    #mn.setEdgeFactor((0,1), np.random.randn(4,3))
    #mn.setEdgeFactor((1,2), np.random.randn(3,5))
    mn.initCandidateEdges()

    print(mn.edgePotentials)
    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))
    print("Neighbors of 2: " + repr(mn.getNeighbors(2)))

    print(mn.evaluateState([0,0,0]))

    print(dir(mn))

if  __name__ =='__main__':
    main()