"""Class to convert from log linear model to MRF"""

from MarkovNet import MarkovNet
try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
from scipy.sparse import csr_matrix


class LogLinearModel(MarkovNet):
    """Log linear model class. Able to convert from log linear features to pairwise MRF. For now, only allows indicator features for pairwise features."""

    def __init__(self):
        """Initialize a LogLinearModel. Create a Markov net."""
        super(LogLinearModel, self).__init__()
        self.unary_features = dict()
        self.unary_feature_weights = dict()
        self.edge_features = dict()
        self.num_features = dict()
        self.num_edge_features = dict()
        self.weight_dim = None

    def set_edge_factor(self, edge, potential):
        super(LogLinearModel, self).set_edge_factor(edge, potential)
        if edge not in self.edge_features:
            # set default edge feature
            self.set_edge_features(edge, np.array([1.0]))

    def set_unary_weights(self, var, weights):
        """Set the log-linear weights for the unary features of var.
        :type weights: np.ndarray
        """
        assert isinstance(weights, np.ndarray)
        assert np.shape(weights)[0] == self.num_states[var]
        self.unary_feature_weights[var] = weights

    def set_unary_features(self, var, values):
        """
        Set the log-linear features for a particular variable
        :rtype: None
        :type values: np.ndarray
        """
        assert isinstance(values, np.ndarray)
        self.unary_features[var] = values

        self.num_features[var] = len(values)

    def set_edge_features(self, edge, values):
        reversed_edge = (edge[1], edge[0])
        self.edge_features[edge] = values
        self.num_edge_features[edge] = len(values)

        self.edge_features[reversed_edge] = values
        self.num_edge_features[reversed_edge] = len(values)

    def set_all_unary_factors(self):
        for var in self.variables:
            self.set_unary_factor(var, np.dot(self.unary_feature_weights[var], self.unary_features[var]))
 
    def set_feature_matrix(self, feature_mat):
        assert (np.array_equal(self.feature_mat.shape, feature_mat.shape))

        self.feature_mat = feature_mat

    def set_weights(self, weight_vector, feature_mat, edge_feature_mat):
        num_vars = len(self.variables)

        feature_size = self.max_features * self.max_states
        feature_weights = weight_vector[:feature_size].reshape((self.max_features, self.max_states))

        pairwise_weights = weight_vector[feature_size:].reshape((self.max_edge_features, self.max_states ** 2))

        weight_mat = self.set_weight_matrix(feature_weights)
        edge_weight_mat = self.set_edge_weight_matrix(pairwise_weights)
        unary_mat = self.update_unary_matrix(feature_mat, weight_mat)
        edge_pot_tensor = self.update_edge_tensor(edge_feature_mat, edge_weight_mat)

        return unary_mat, edge_pot_tensor

    def set_weight_matrix(self, weight_mat):
        assert (np.array_equal(self.weight_mat.shape, weight_mat.shape))
        self.weight_mat = weight_mat
        return weight_mat

    def set_edge_weight_matrix(self, edge_weight_mat):
        assert (np.array_equal(self.edge_weight_mat.shape, edge_weight_mat.shape))
        self.edge_weight_mat = edge_weight_mat
        return edge_weight_mat

    def update_unary_matrix(self, feature_mat, weight_mat):
        unary_mat = self.set_unary_mat(np.dot(feature_mat.T, weight_mat).T)
        return unary_mat

    def update_edge_tensor(self,edge_feature_mat, edge_weight_mat):
        half_edge_tensor = np.dot(edge_feature_mat.T, edge_weight_mat).T.reshape(
            (self.max_states, self.max_states, self.num_edges))

        self.edge_pot_tensor = np.concatenate((np.transpose(half_edge_tensor, (1, 0, 2)), half_edge_tensor), axis=2)
        return self.edge_pot_tensor

    def create_matrices(self):
        super(LogLinearModel, self).create_matrices()

        # create unary matrices
        self.max_features = max([x for x in self.num_features.values()])
        self.weight_mat = np.zeros((self.max_features, self.max_states))
        self.feature_mat = np.zeros((self.max_features, len(self.variables)))

        i = 0
        for var in self.variables:
            self.feature_mat[:, i] = self.unary_features[var]
            i += 1

        # create edge matrices
        self.max_edge_features = max([x for x in self.num_edge_features.values()])
        self.edge_weight_mat = np.zeros((self.max_edge_features, self.max_states**2))
        self.edge_feature_mat = np.zeros((self.max_edge_features, self.num_edges))

        for edge, i in self.edge_index.items():
            self.edge_feature_mat[:, i] = self.edge_features[edge]

        self.weight_dim = self.max_states * self.max_features + self.max_edge_features * self.max_states ** 2

    def create_indicator_model(self, markov_net):
        n = len(markov_net.variables)

        # set unary variables
        for i, var in enumerate(markov_net.variables):
            self.declare_variable(var, num_states=markov_net.num_states[var])
            self.set_unary_factor(var, markov_net.unary_potentials[var])
            indicator_features = np.zeros(n)
            indicator_features[i] = 1.0
            self.set_unary_features(var, indicator_features)

        # count edges
        num_edges = 0
        for var in markov_net.variables:
            for neighbor in markov_net.get_neighbors(var):
                num_edges += 1

        # create edge indicator features
        i = 0
        for var in markov_net.variables:
            for neighbor in markov_net.get_neighbors(var):
                edge = (var, neighbor)
                self.set_edge_factor(edge, markov_net.get_potential(edge))
                indicator_features = np.zeros(num_edges)
                indicator_features[i] = 1.0
                self.set_edge_features(edge, indicator_features)
                i += 1

        self.create_matrices()

        #self.feature_mat = csr_matrix(self.feature_mat)
        #self.edge_feature_mat = csr_matrix(self.edge_feature_mat)

        for (var, i) in self.var_index.items():
            self.weight_mat[i, :] = -np.inf
            potential = self.unary_potentials[var]
            self.weight_mat[i, :len(potential)] = potential

        for (edge, i) in self.edge_index.items():
            padded_potential = -np.inf * np.ones((self.max_states, self.max_states))
            potential = self.get_potential(edge)
            padded_potential[:self.num_states[edge[0]], :self.num_states[edge[1]]] = potential
            self.edge_weight_mat[i, :] = padded_potential.ravel()

    def load_factors_from_matrices(self):
        self.update_unary_matrix()
        self.update_edge_tensor()

        for (var, i) in self.var_index.items():
            self.set_unary_factor(var, self.unary_mat[:self.num_states[var], i].ravel())

        for edge, i in self.edge_index.items():
            self.set_edge_factor(edge,
                         self.edge_pot_tensor[:self.num_states[edge[1]], :self.num_states[edge[0]], i].squeeze().T)
