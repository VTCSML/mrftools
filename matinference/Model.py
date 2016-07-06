import numpy as np
import logging
from scipy.sparse import dok_matrix, csc_matrix

logger = logging.getLogger(__name__)


class Model(object):
    """
    Contains the Markov random field structure and observed features.
    """

    def __init__(self, num_states=2, unary_dim=1, pair_dim=1):
        self.vars = []
        self.var_index = dict()
        self.edges = []
        self.edge_index = dict()
        self.neighbors = dict()
        self.features = dict()
        self.unary_dim = unary_dim
        self.pair_dim = pair_dim
        self.num_states = num_states
        self.weight_dim = self.unary_dim * self.num_states + self.pair_dim * self.num_states ** 2
        self.weights = np.zeros((self.weight_dim, 1))
        self.template_mat = None
        self.factor_mat = None
        self.num_marginals = 0
        self.belief_indices = dict()

    def add_var(self, name, feature_vec=np.array([1.0])):
        """
        Adds a variable to the model.
        :param name: unique identifier for the variable (can be any hashable value)
        :param feature_vec: feature vector for the unary potential of this variable
        :return: None
        """
        assert name not in self.var_index, "Cannot add %s. %s is already in our variable set." % (name, name)

        self.var_index[name] = len(self.vars)
        self.vars.append(name)

        self.neighbors[name] = set()

        assert self.unary_dim == feature_vec.size, "Expected dimensionality-%d features for %s, but received %d." \
                                                   % (self.unary_dim, name, feature_vec.size)

        self.features[name] = feature_vec

        self.num_marginals = len(self.vars) * self.num_states + len(self.edges) * self.num_states ** 2

    def add_edge(self, pair, feature_vec=np.array([1.0])):
        """
        Adds an edge to the model.
        :param pair: Length-2 tuple of the two variables involved in the edge.
        :param feature_vec: Feature vector for the pairwise potential of this variable
        :return: None
        """
        (left, right) = pair
        rev_pair = (right, left)

        assert left in self.var_index and right in self.var_index, \
            "Attempted to add edge when nodes are not known variables."
        assert pair not in self.features and rev_pair not in self.features, "Attempted to add edge that already exists."
        assert self.pair_dim == feature_vec.size, "Expected dimensionality-%d features for %s, but received %d." % \
                                                  (self.pair_dim, pair, feature_vec.size)

        self.edge_index[pair] = len(self.edges)
        self.edges.append(pair)
        self.neighbors[left].add(right)
        self.neighbors[right].add(left)
        self.features[pair] = feature_vec

        self.num_marginals = len(self.vars) * self.num_states + len(self.edges) * self.num_states ** 2

    def set_weights(self, new_weights):
        assert new_weights.size == self.weight_dim, "Weights are the wrong dimensionality"
        self.weights = new_weights.reshape((self.weight_dim, 1))

    def create_matrices(self):
        """
        Creates templating matrix such that self.weights.T.dot(self.template_mat) is the log potential vector
        :return: None
        """
        self.template_mat = dok_matrix((self.weight_dim, self.num_marginals))
        self.factor_mat = dok_matrix((self.num_marginals, len(self.vars) + len(self.edges)))

        column = 0
        factor_index = 0
        for var in self.vars:
            indices = []
            for i in range(self.num_states):
                index_start = i * self.unary_dim
                self.template_mat[index_start:index_start + self.unary_dim, column] = self.features[var]
                indices.append(column)
                column += 1
            self.belief_indices[var] = indices
            self.factor_mat[indices, factor_index] = 1
            factor_index += 1

        for edge in self.edges:
            indices = []
            for i in range(self.num_states):
                for j in range(self.num_states):
                    index_start = (i * self.num_states + j) * self.pair_dim
                    self.template_mat[index_start:index_start + self.pair_dim, column] = self.features[edge]
                    indices.append(column)
                    column += 1
            self.belief_indices[edge] = indices
            self.factor_mat[indices, factor_index] = 1
            factor_index += 1

        self.template_mat = csc_matrix(self.template_mat)
        self.factor_mat = csc_matrix(self.factor_mat)
