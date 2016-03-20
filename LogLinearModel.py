"""Class to convert from log linear model to MRF"""

from MarkovNet import MarkovNet
import numpy as np


class LogLinearModel(MarkovNet):
    """Log linear model class. Able to convert from log linear features to pairwise MRF. For now, only allows indicator features for pairwise features."""

    def __init__(self):
        """Initialize a LogLinearModel. Create a Markov net."""
        super(LogLinearModel, self).__init__()
        self.unaryFeatures = dict()
        self.unaryFeatureWeights = dict()
        self.numFeatures = dict()

    def setUnaryWeights(self, var, weights):
        """Set the log-linear weights for the unary features of var.
        :type weights: np.ndarray
        """
        assert isinstance(weights, np.ndarray)
        assert np.shape(weights)[0] == self.numStates[var]
        self.unaryFeatureWeights[var] = weights

    def setUnaryFeatures(self, var, values):
        """
        Set the log-linear features for a particular variable
        :rtype: None
        :type values: np.ndarray
        """
        assert isinstance(values, np.ndarray)
        self.unaryFeatures[var] = values

        self.numFeatures[var] = len(values)

    def setAllUnaryFactors(self):
        for var in self.variables:
            self.setUnaryFactor(var, self.unaryFeatureWeights[var].dot(self.unaryFeatures[var]))


def main():
    """Test function for MarkovNet."""
    model = LogLinearModel()

    model.declareVariable(0, 4)
    model.declareVariable(1, 3)
    model.declareVariable(2, 5)

    model.setUnaryWeights(0, np.random.randn(4, 3))
    model.setUnaryWeights(1, np.random.randn(3, 3))
    model.setUnaryWeights(2, np.random.randn(5, 3))

    model.setUnaryFeatures(0, np.random.randn(3))
    model.setUnaryFeatures(1, np.random.randn(3))
    model.setUnaryFeatures(2, np.random.randn(3))

    model.setAllUnaryFactors()

    model.setEdgeFactor((0, 1), np.random.randn(4, 3))
    model.setEdgeFactor((1, 2), np.random.randn(3, 5))

    print("Neighbors of 0: " + repr(model.getNeighbors(0)))
    print("Neighbors of 1: " + repr(model.getNeighbors(1)))
    print("Neighbors of 2: " + repr(model.getNeighbors(2)))

    print(model.evaluateState([0, 0, 0]))


if __name__ == '__main__':
    main()
