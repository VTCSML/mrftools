"""Markov network class for storing potential functions and structure."""
import numpy as np
from scipy.sparse import csc_matrix, dok_matrix
import time

class MarkovNet(object):
    """Object containing the definition of a pairwise Markov net."""

    def __init__(self):
        """Initialize a Markov net."""
        self.edge_potentials = dict()
        self.unary_potentials = dict()
        self.neighbors = dict()
        self.variables = set()
        self.num_states = dict()
        self.matrix_mode = False
        self.tree_probabilities = dict()
        self.num_weights = 0
        self.map_features_to_space = []
        self.space = dict()
        self.search_space = []
        self.num_weights_search_space = 0
        self.has_edges = False

    def set_unary_factor(self, variable, potential):
        """Set the potential function for the unary factor. Implicitly declare variable. Must be called before setting edge factors."""
        self.unary_potentials[variable] = potential
        if variable not in self.variables:
            self.declare_variable(variable, np.size(potential))

    def declare_variable(self, variable, num_states):
        if variable not in self.variables:
            self.variables.add(variable)
            self.neighbors[variable] = set()
            self.num_states[variable] = num_states
        else:
            print("Warning: declaring a variable %s that was previously declared." % repr(variable))

    def set_edge_factor(self, edge, potential):
        """Set a factor by inputting the involved variables then the potential function. The potential function should be a np matrix."""
        assert np.shape(potential) == (len(self.unary_potentials[edge[0]]), len(self.unary_potentials[edge[1]])), "potential size %d, %d incompatible with unary sizes %d, %d" % (np.shape(potential)[0], np.shape(potential)[1], len(self.unary_potentials[edge[0]]), len(self.unary_potentials[edge[1]]))
        self.has_edges = True
        if edge[0] < edge[1]:
            self.edge_potentials[edge] = potential
        else:
            self.edge_potentials[(edge[1], edge[0])] = potential.T

        self.neighbors[edge[0]].add(edge[1])
        self.neighbors[edge[1]].add(edge[0])


    def get_potential(self, pair):
        """Return the potential between pair[0] and pair[1]. If (pair[1], pair[0]) is in our dictionary instead, return the transposed potential."""
        if pair in self.edge_potentials:
            return self.edge_potentials[pair]
        else:
            return self.edge_potentials[(pair[1], pair[0])].T

    def get_neighbors(self, variable):
        """Return the neighbors of variable."""
        return self.neighbors[variable]

    def evaluate_state(self, states):
        """Evaluate the energy of a state. states should be a dictionary of variable: state (int) pairs."""
        energy = 0.0
        for var in self.variables:
            energy += self.unary_potentials[var][states[var]]

            for neighbor in self.neighbors[var]:
                if var < neighbor:
                    energy += self.get_potential((var, neighbor))[states[var], states[neighbor]]

        return energy


    def set_unary_mat(self, unary_mat):
        assert np.array_equal(self.unary_mat.shape, unary_mat.shape)
        self.unary_mat[:, :] = unary_mat


    def set_edge_tensor(self, edge_tensor):
        if np.array_equal(self.edge_pot_tensor.shape, edge_tensor.shape):
            self.edge_pot_tensor[:,:,:] = edge_tensor
        else:
            mirrored_edge_tensor = np.concatenate((edge_tensor, edge_tensor.transpose((1, 0, 2))), 2)
            assert np.array_equal(self.edge_pot_tensor.shape, mirrored_edge_tensor.shape)

            self.edge_pot_tensor[:, :, :] = mirrored_edge_tensor

    def create_matrices(self):
        self.matrix_mode = True

        self.max_states = max([len(x) for x in self.unary_potentials.values()])
        self.unary_mat = -np.inf * np.ones((self.max_states, len(self.variables)))
        self.var_index = dict()
        self.var_list = []
        self.degrees = np.zeros(len(self.variables))

        i = 0
        for var in self.variables:
            potential = self.unary_potentials[var]
            self.unary_mat[0:len(potential), i] = potential
            self.var_index[var] = i
            self.var_list.append(var)
            self.degrees[i] = len(self.neighbors[var])
            i += 1

        # set up pairwise tensor

        self.num_edges = 0
        for var in self.variables:
            for neighbor in self.neighbors[var]:
                if var < neighbor:
                    self.num_edges += 1

        self.edge_pot_tensor = -np.inf * np.ones((self.max_states, self.max_states, 2 * self.num_edges))
        self.edge_index = {}

        from_rows = []
        from_cols = []
        to_rows = []
        to_cols = []

        i = 0
        for var in self.variables:
            for neighbor in self.neighbors[var]:
                if var < neighbor:
                    potential = self.get_potential((var, neighbor))
                    dims = potential.shape

                    self.edge_pot_tensor[0:dims[1], 0:dims[0], i] = potential.T
                    self.edge_pot_tensor[0:dims[0], 0:dims[1], i + self.num_edges] = potential

                    var_i = self.var_index[var]
                    neighbor_i = self.var_index[neighbor]

                    from_rows.append(i)
                    from_cols.append(var_i)
                    from_rows.append(i + self.num_edges)
                    from_cols.append(neighbor_i)

                    to_rows.append(i)
                    to_cols.append(neighbor_i)
                    to_rows.append(i + self.num_edges)
                    to_cols.append(var_i)

                    self.edge_index[(var, neighbor)] = i

                    i += 1

        self.message_to_map = csc_matrix((np.ones(len(to_rows)), (to_rows, to_cols)),
                                         (2 * self.num_edges, len(self.variables)))

        self.message_to = np.zeros(2 * self.num_edges, dtype=np.intp)
        self.message_to[to_rows] = to_cols

        self.message_from = np.zeros(2 * self.num_edges, dtype=np.intp)
        self.message_from[from_rows] = from_cols

    def init_candidate_edges(self):
        """Initialize the set of all possible Edges"""
        for var1 in self.variables:
            self.space[var1] = copy.deepcopy(self.unaryPotentials[var1])
            for var2 in self.variables:
                if var1 < var2:
                    self.edgePotentials[(var1, var2)] = np.zeros(
                        shape=(len(self.unaryPotentials[var1]), len(self.unaryPotentials[var2])))
                    self.space[(var1, var2)] = np.zeros(
                        shape=(len(self.unaryPotentials[var1]), len(self.unaryPotentials[var2])))
                    self.search_space.append((var1, var2))
                if var1 != var2:
                    self.neighbors[var1].add(var2)

    def init_search_space(self):
        """Initialize the set of all possible Edges"""
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 < var2:
                    self.search_space.append((var1, var2))
                    self.num_weights_search_space += len(self.unary_potentials[var1]) * len(self.unary_potentials[var2])

    def update_search_space(self, edge):
        self.search_space.remove(edge)
        self.num_weights_search_space -= len(self.unary_potentials[edge[0]]) * len(self.unary_potentials[edge[1]])

    def init_weights(self):
        """Initialize the set of all possible Weights"""
        for x in self.space:
            pot = self.space[x]
            if isinstance(x, tuple):
                curr_num_weights = np.prod(pot.shape)
            else:
                curr_num_weights = len(pot)
            self.num_weights += curr_num_weights
            for t in range(curr_num_weights):
                self.map_features_to_space.append(x)
        return (np.zeros(self.num_weights))

    def get_edges_data_sum(self, data):
        """Compute joint states reoccurrences in the data"""
        edgesDataSum = dict()
        for edge in self.search_space:
            edgesDataSum[edge] = np.asarray(
                np.zeros((len(self.unary_potentials[edge[0]]), (len(self.unary_potentials[edge[1]])))).reshape((-1, 1)))
        for states in data:
            for edge in self.search_space:
                table = np.zeros((len(self.unary_potentials[edge[0]]), (len(self.unary_potentials[edge[1]]))))
                table[states[edge[0]], states[edge[1]]] = 1
                tmp = np.asarray(table.reshape((-1, 1)))
                edgesDataSum[edge] = edgesDataSum[edge] + tmp
        return edgesDataSum

