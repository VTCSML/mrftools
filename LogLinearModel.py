"""Class to convert from log linear model to MRF"""

from MarkovNet import MarkovNet
import numpy as np


class LogLinearModel(MarkovNet):
    """Log linear model class. Able to convert from log linear features to pairwise MRF. For now, only allows indicator features for pairwise features."""

    def __init__(self):
        """Initialize a LogLinearModel. Create a Markov net."""
        super(LogLinearModel, self).__init__()
        self.unaryFeatures = dict()
        self.unary_feature_weights = dict()
        self.numFeatures = dict()

    def setUnaryWeights(self, var, weights):
        """Set the log-linear weights for the unary features of var.
        :type weights: np.ndarray
        """
        assert isinstance(weights, np.ndarray)
        assert np.shape(weights)[0] == self.numStates[var]
        self.unary_feature_weights[var] = weights

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
            self.setUnaryFactor(var, self.unary_feature_weights[var].dot(self.unaryFeatures[var]))

    def set_feature_matrix(self, feature_mat):
        assert (np.array_equal(self.feature_mat.shape, feature_mat.shape))

        self.feature_mat[:, :] = feature_mat

    def set_weight_matrix(self, weight_mat):
        assert (np.array_equal(self.weight_mat.shape, weight_mat.shape))

        self.weight_mat[:, :] = weight_mat

    def set_unary_matrix(self):
        self.set_unary_mat(self.weight_mat.T.dot(self.feature_mat))

    def create_matrices(self):
        super(LogLinearModel, self).create_matrices()

        self.max_features = max([x for x in self.numFeatures.values()])

        self.weight_mat = np.zeros((self.max_features, self.max_states))

        self.feature_mat = np.zeros((self.max_features, len(self.variables)))

        for var in self.variables:
            index = self.var_index[var]
            self.feature_mat[:, index] = self.unaryFeatures[var]


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
