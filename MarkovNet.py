"""Markov network class for storing potential functions and structure."""
import numpy as np
from scipy.sparse import csc_matrix, dok_matrix

class MarkovNet(object):
    """Object containing the definition of a pairwise Markov net."""

    def __init__(self):
        """Initialize a Markov net."""
        self.edgePotentials = dict()
        self.unaryPotentials = dict()
        self.neighbors = dict()
        self.variables = set()
        self.numStates = dict()
        self.matrix_mode = False

    def setUnaryFactor(self, variable, potential):
        """Set the potential function for the unary factor. Implicitly declare variable. Must be called before setting edge factors."""
        self.unaryPotentials[variable] = potential
        if variable not in self.variables:
            self.declareVariable(variable, np.size(potential))

    def declareVariable(self, variable, numStates):
        if variable not in self.variables:
            self.variables.add(variable)
            self.neighbors[variable] = set()
            self.numStates[variable] = numStates
        else:
            print("Warning: declaring a variable %s that was previously declared." % repr(variable))

    def setEdgeFactor(self, edge, potential):
        """Set a factor by inputting the involved variables then the potential function. The potential function should be a np matrix."""
        assert np.shape(potential) == (len(self.unaryPotentials[edge[0]]), len(self.unaryPotentials[edge[1]])), "potential size %d, %d incompatible with unary sizes %d, %d" % (np.shape(potential)[0], np.shape(potential)[1], len(self.unaryPotentials[edge[0]]), len(self.unaryPotentials[edge[1]]))

        if edge[0] < edge[1]:
            self.edgePotentials[edge] = potential
        else:
            self.edgePotentials[(edge[1], edge[0])] = potential.T

        self.neighbors[edge[0]].add(edge[1])
        self.neighbors[edge[1]].add(edge[0])

    def getPotential(self, pair):
        """Return the potential between pair[0] and pair[1]. If (pair[1], pair[0]) is in our dictionary instead, return the transposed potential."""
        if pair in self.edgePotentials:
            return self.edgePotentials[pair]
        else:
            return self.edgePotentials[(pair[1], pair[0])].T

    def getNeighbors(self, variable):
        """Return the neighbors of variable."""
        return self.neighbors[variable]

    def evaluateState(self, states):
        """Evaluate the energy of a state. states should be a dictionary of variable: state (int) pairs."""
        energy = 0.0
        for var in self.variables:
            energy += self.unaryPotentials[var][states[var]]

            for neighbor in self.neighbors[var]:
                if var < neighbor:
                    energy += self.getPotential((var, neighbor))[states[var], states[neighbor]]

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

        self.max_states = max([len(x) for x in self.unaryPotentials.values()])
        self.unary_mat = -np.inf * np.ones((self.max_states, len(self.variables)))
        self.var_index = dict()
        self.var_list = []
        self.degrees = np.zeros(len(self.variables))

        i = 0
        for var in self.variables:
            potential = self.unaryPotentials[var]
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
        self.edges = []

        from_rows = []
        from_cols = []
        to_rows = []
        to_cols = []

        i = 0
        for var in self.variables:
            for neighbor in self.neighbors[var]:
                if var < neighbor:
                    potential = self.getPotential((var, neighbor))
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

                    self.edges.append((var, neighbor))

                    i += 1

        self.message_to_index = csc_matrix((np.ones(len(to_rows)), (to_rows, to_cols)),
                                      (2 * self.num_edges, len(self.variables)))
        self.message_from_index = csc_matrix((np.ones(len(from_rows)), (from_rows, from_cols)),
                                      (2 * self.num_edges, len(self.variables)))


def main():
    """Test function for MarkovNet."""
    mn = MarkovNet()

    mn.setUnaryFactor(0, np.random.randn(4))
    mn.setUnaryFactor(1, np.random.randn(3))
    mn.setUnaryFactor(2, np.random.randn(5))

    mn.setEdgeFactor((0,1), np.random.randn(4,3))
    mn.setEdgeFactor((1,2), np.random.randn(3,5))

    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))
    print("Neighbors of 2: " + repr(mn.getNeighbors(2)))

    print(mn.evaluateState([0,0,0]))

if  __name__ =='__main__':
    main()
