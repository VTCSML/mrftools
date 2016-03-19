"""Class to do generative learning directly on MRF parameters."""

from MarkovNet import MarkovNet
from BeliefPropagator import BeliefPropagator
import numpy as np
from scipy.optimize import minimize

class ApproxMaxLikelihood(object):
    """Object that runs approximate maximum likelihood parameter training."""

    def __init__(self, markovNet):
        """Initialize learner. Pass in Markov net."""
        self.mn = markovNet
        self.data = []
        self.l1Regularization = 1
        self.l2Regularization = 1
        self.bp = BeliefPropagator(self.mn)
        self.dataSum = 0

        # set up order of potentials
        self.potentials = []
        for var in self.mn.variables:
            self.potentials.append(var)
            for neighbor in self.mn.getNeighbors(var):
                if var < neighbor:
                    self.potentials.append((var, neighbor))
                    assert (var, neighbor) in self.mn.edgePotentials

    def setRegularization(self, l1, l2):
        """Set the regularization parameters."""
        self.l1Regularization = l1
        self.l2Regularization = l2

    def addData(self, states):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables."""
        example = []

        # create vector representation of data using the same order as self.potentials
        for i in range(len(self.potentials)):
            if isinstance(self.potentials[i], tuple):
                # set pairwise feature
                pair = self.potentials[i]
                table = np.zeros((self.mn.numStates[pair[0]], self.mn.numStates[pair[1]]))
                table[states[pair[0]], states[pair[1]]] = 1
            else:
                # set unary feature
                var = self.potentials[i]
                table = np.zeros(self.mn.numStates[var])
                table[states[var]] = 1

            # flatten table and append
            example.extend(table.reshape((-1, 1)).tolist())

        self.data.append(np.array(example))
        self.dataSum += np.array(example)

    def setWeights(self, weightVector):
        """Set weights of Markov net from vector using the order in self.potentials."""
        j = 0
        for i in range(len(self.potentials)):
            if isinstance(self.potentials[i], tuple):
                # set pairwise potential
                pair = self.potentials[i]
                size = (self.mn.numStates[pair[0]], self.mn.numStates[pair[1]])
                self.mn.setEdgeFactor(pair, weightVector[j:j + np.prod(size)].reshape(size))
                j += np.prod(size)
            else:
                # set unary potential
                var = self.potentials[i]
                size = self.mn.numStates[var]
                self.mn.setUnaryFactor(var, weightVector[j:j+size])
                j += size

        assert j == len(weightVector)


    def getMarginalVector(self):
        """Run inference and return the marginal in vector form using the order of self.potentials."""
        self.bp.runInference(display = 'off')
        self.bp.computeBeliefs()
        self.bp.computePairwiseBeliefs()

        # make vector form of marginals
        marginals = []
        for i in range(len(self.potentials)):
            if isinstance(self.potentials[i], tuple):
                # get pairwise belief
                table = self.bp.pairBeliefs[self.potentials[i]]
            else:
                # get unary belief
                table = self.bp.varBeliefs[self.potentials[i]]

            # flatten table and append
            marginals.extend(table.reshape((-1, 1)).tolist())
        marginals = np.array(marginals)

        return marginals


    def objective(self, weightVector):
        """Compute the learning objective with the provided weight vector."""
        self.setWeights(weightVector)
        marginals = self.getMarginalVector()

        objective = 0.0

        # add regularization penalties
        objective += self.l1Regularization * np.sum(np.abs(weightVector))
        objective += 0.5 * self.l2Regularization * weightVector.dot(weightVector)

        # add likelihood penalty
        objective += weightVector.dot(marginals - self.dataSum / len(self.data))

        return objective

    def gradient(self, weightVector):
        """Compute the gradient for the provided weight vector."""
        self.setWeights(weightVector)
        marginals = self.getMarginalVector()

        gradient = np.zeros(len(weightVector))

        # add regularization penalties
        gradient += self.l1Regularization * np.sign(weightVector)
        gradient += self.l2Regularization * weightVector

        # add likelihood penalty
        gradient += (marginals - self.dataSum / len(self.data)).squeeze()

        return gradient



def main():
    """Simple test function for maximum likelihood."""
    mn = MarkovNet()

    mn.setUnaryFactor(0, np.zeros(4))
    mn.setUnaryFactor(1, np.zeros(3))
    mn.setUnaryFactor(2, np.zeros(2))

    mn.setEdgeFactor((0,1), np.zeros((4,3)))
    mn.setEdgeFactor((1,2), np.zeros((3,2)))

    aml = ApproxMaxLikelihood(mn)

    aml.addData({0:0, 1:0, 2:0})
    aml.addData({0:1, 1:1, 2:1})
    aml.addData({0:2, 1:2, 2:1})
    aml.addData({0:3, 1:2, 2:1})
    # print aml.data

    weights = np.random.randn(4 + 3 + 2 + 4*3 + 3*2)

    print aml.objective(weights)
    print aml.gradient(weights)

    res = minimize(aml.objective, weights, method='L-BFGS-B', jac = aml.gradient)

    print res

if  __name__ =='__main__':
    main()
