"""Class to do generative learning directly on MRF parameters."""

from MarkovNet import MarkovNet
from BeliefPropagator import BeliefPropagator
import numpy as np
from scipy.optimize import minimize, check_grad


class ApproxMaxLikelihood(object):
    """Object that runs approximate maximum likelihood parameter training."""

    def __init__(self, markov_net):
        """Initialize learner. Pass in Markov net."""
        self.mn = markov_net
        self.data = []
        self.l1Regularization = 1
        self.l2Regularization = 1
        self.bp = BeliefPropagator(self.mn)
        self.dataSum = 0

        # set up order of potentials
        self.potentials = []
        for var in self.mn.variables:
            self.potentials.append(var)
            for neighbor in self.mn.get_neighbors(var):
                if var < neighbor:
                    self.potentials.append((var, neighbor))
                    assert (var, neighbor) in self.mn.edgePotentials

    def set_regularization(self, l1, l2):
        """Set the regularization parameters."""
        self.l1Regularization = l1
        self.l2Regularization = l2

    def add_data(self, states):
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

    def set_weights(self, weight_vector):
        """Set weights of Markov net from vector using the order in self.potentials."""
        j = 0
        for i in range(len(self.potentials)):
            if isinstance(self.potentials[i], tuple):
                # set pairwise potential
                pair = self.potentials[i]
                size = (self.mn.numStates[pair[0]], self.mn.numStates[pair[1]])
                self.mn.set_edge_factor(pair, weight_vector[j:j + np.prod(size)].reshape(size))
                j += np.prod(size)
            else:
                # set unary potential
                var = self.potentials[i]
                size = self.mn.numStates[var]
                self.mn.set_unary_factor(var, weight_vector[j:j + size])
                j += size

        assert j == len(weight_vector)


    def get_marginal_vector(self):
        """Run inference and return the marginal in vector form using the order of self.potentials."""
        self.bp.runInference(display = 'off')
        self.bp.compute_beliefs()
        self.bp.compute_pairwise_beliefs()

        # make vector form of marginals
        marginals = []
        for i in range(len(self.potentials)):
            if isinstance(self.potentials[i], tuple):
                # get pairwise belief
                table = np.exp(self.bp.pair_beliefs[self.potentials[i]])
            else:
                # get unary belief
                table = np.exp(self.bp.var_beliefs[self.potentials[i]])

            # flatten table and append
            marginals.extend(table.reshape((-1, 1)).tolist())
        marginals = np.array(marginals)

        return marginals


    def objective(self, weight_vector):
        """Compute the learning objective with the provided weight vector. Approximate negative log-likelihood."""
        self.set_weights(weight_vector)
        marginals = self.get_marginal_vector()

        objective = 0.0

        # add regularization penalties
        objective += self.l1Regularization * np.sum(np.abs(weight_vector))
        objective += 0.5 * self.l2Regularization * weight_vector.dot(weight_vector)

        # add likelihood penalty log Z - labelEnergy
        objective += self.bp.computeEnergyFunctional()
        objective -= weight_vector.dot(self.dataSum / len(self.data))


        return objective

    def gradient(self, weight_vector):
        """Compute the gradient for the provided weight vector."""
        self.set_weights(weight_vector)
        marginals = self.get_marginal_vector()

        gradient = np.zeros(len(weight_vector))

        # add regularization penalties
        gradient += self.l1Regularization * np.sign(weight_vector)
        gradient += self.l2Regularization * weight_vector

        # add likelihood penalty
        gradient += (marginals - self.dataSum / len(self.data)).squeeze()

        return gradient



def main():
    """Simple test function for maximum likelihood."""

    np.random.seed(0)

    mn = MarkovNet()

    mn.set_unary_factor(0, np.zeros(4))
    mn.set_unary_factor(1, np.zeros(3))
    mn.set_unary_factor(2, np.zeros(2))

    mn.set_edge_factor((0, 1), np.zeros((4, 3)))
    mn.set_edge_factor((1, 2), np.zeros((3, 2)))

    aml = ApproxMaxLikelihood(mn)

    aml.set_regularization(0, 1)

    aml.add_data({0:0, 1:0, 2:0})
    aml.add_data({0:1, 1:1, 2:1})
    aml.add_data({0:2, 1:2, 2:1})
    aml.add_data({0:3, 1:2, 2:1})
    # print aml.data

    weights = np.random.randn(4 + 3 + 2 + 4*3 + 3*2)

    print "\n\nGradient check:"
    print check_grad(aml.objective, aml.gradient, weights)

    print aml.objective(weights)
    print aml.gradient(weights)

    res = minimize(aml.objective, weights, method='L-BFGS-B', jac = aml.gradient)

    print res

if  __name__ =='__main__':
    main()
