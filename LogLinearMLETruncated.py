from LogLinearModel import LogLinearModel
import math as math
import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check
import copy
from scipy.optimize import minimize, check_grad
from BeliefPropagatorTruncated import BeliefPropagator
from MatrixCache import MatrixCache

class LogLinearMLE(object):
    """Object that runs approximate maximum likelihood parameter training."""

    def __init__(self, baseModel):
        """Initialize learner. Pass in Markov net."""
        assert isinstance(baseModel, LogLinearModel)
        self.baseModel = baseModel
        self.models = []
        self.beliefPropagators = []
        self.labels = []
        self.l1Regularization = 1
        self.l2Regularization = 1
        self.featureSum = 0
        self.prevWeights = 0
        self.needInference = True

        # set up order of potentials
        self.potentials = []
        for var in baseModel.variables:
            self.potentials.append(var)
            for neighbor in baseModel.getNeighbors(var):
                if var < neighbor:
                    self.potentials.append((var, neighbor))
                    assert (var, neighbor) in baseModel.edgePotentials

    def setRegularization(self, l1, l2):
        """Set the regularization parameters."""
        self.l1Regularization = l1
        self.l2Regularization = l2

    def addData(self, states, features):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""
        example = []

        model = copy.deepcopy(self.baseModel)

        # create vector representation of data using the same order as self.potentials
        for i in range(len(self.potentials)):
            if self.potentials[i] not in self.baseModel.variables:
                # set pairwise state
                pair = self.potentials[i]
                table = np.zeros((model.numStates[pair[0]], model.numStates[pair[1]]))
                table[states[pair[0]], states[pair[1]]] = 1
            else:
                # set unary data
                var = self.potentials[i]

                table = np.zeros((model.numStates[var], len(features[var])))
                table[states[var],:] = features[var]

                # set model features
                model.setUnaryFeatures(var, features[var])

            # flatten table and append
            example.extend(table.reshape((-1, 1)).tolist())

        self.models.append(model)
        self.beliefPropagators.append(BeliefPropagator(model))
        self.labels.append(np.array(example))
        self.featureSum += np.array(example)

    def setWeights(self, weightVector):
        """Set weights of Markov net from vector using the order in self.potentials."""
        if np.array_equal(weightVector, self.prevWeights):
            # if using the same weight vector as previously, there is no need to rerun inference
            # this often happens when computing the objective and the gradient with the same weights
            self.needInference = False
            return

        self.prevWeights = weightVector
        self.needInference = True

        weightCache = MatrixCache()
        for model in self.models:
            j = 0
            for i in range(len(self.potentials)):
                if isinstance(self.potentials[i], tuple):
                    # set pairwise potential
                    pair = self.potentials[i]
                    size = (model.numStates[pair[0]], model.numStates[pair[1]])
                    factorWeights = weightCache.getCached(weightVector[j:j + np.prod(size)].reshape(size))
                    model.setEdgeFactor(pair, factorWeights)
                    j += np.prod(size)
                else:
                    # set unary potential
                    var = self.potentials[i]
                    size = (model.numStates[var], model.numFeatures[var])
                    factorWeights = weightCache.getCached(weightVector[j:j + np.prod(size)].reshape(size))
                    model.setUnaryWeights(var, factorWeights)
                    j += np.prod(size)
            model.setAllUnaryFactors()

            assert j == len(weightVector)

    def getFeatureExpectationsTruncated(self):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        :rtype: numpy.ndarray
        """
        marginalSum = 0
        for i in range(len(self.labels)):
            bp = self.beliefPropagators[i]
            model = self.models[i]
            if self.needInference:
                bp.runTruncatedInference(display = 'off')
                # edit for autograd
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()


            # make vector form of marginals
            marginals = []
            for i in range(len(self.potentials)):
                if isinstance(self.potentials[i], tuple):
                    # get pairwise belief
                    table = np.exp(bp.pairBeliefs[self.potentials[i]])
                else:
                    # get unary belief and multiply by features
                    var = self.potentials[i]
                    table = np.outer(np.exp(bp.varBeliefs[var]), model.unaryFeatures[var])

                # flatten table and append
                marginals.extend(list(table.reshape((-1, 1))))
            marginalSum += np.array(marginals)

        return marginalSum / len(self.labels)


    def objective(self, weightVector):
        self.setWeights(weightVector)
        featureExpectations = self.getFeatureExpectationsTruncated()

        objective = 0.0
        # add regularization penalties
        #objective += self.l1Regularization * np.sum(np.abs(weightVector))
        objective += 0.5 * self.l2Regularization * np.dot(weightVector, weightVector)
        # add likelihood penalty
        objective -= np.dot(weightVector, (self.featureSum / len(self.labels)))

        for bp in self.beliefPropagators:
            objective += bp.computeEnergyFunctional() / len(self.labels)
        # print "Finished one inference"
        
        return objective




def main():
    """Simple test function for maximum likelihood."""
    model = LogLinearModel()

    model.declareVariable(0, 4)
    model.declareVariable(1, 3)
    model.declareVariable(2, 2)

    d = 3

    model.setUnaryWeights(0, np.random.randn(4, d))
    model.setUnaryWeights(1, np.random.randn(3, d))
    model.setUnaryWeights(2, np.random.randn(2, d))

    model.setUnaryFeatures(0, np.random.randn(d))
    model.setUnaryFeatures(1, np.random.randn(d))
    model.setUnaryFeatures(2, np.random.randn(d))

    model.setAllUnaryFactors()

    model.setEdgeFactor((0,1), np.zeros((4,3)))
    model.setEdgeFactor((1,2), np.zeros((3,2)))

    learner = LogLinearMLE(model)

    learner.addData({0:0, 1:0, 2:0}, {0: np.random.randn(3), 1: np.random.randn(3), 2: np.random.randn(3)})
    learner.addData({0:1, 1:1, 2:0}, {0: np.random.randn(3), 1: np.random.randn(3), 2: np.random.randn(3)})
    learner.addData({0:2, 1:0, 2:1}, {0: np.random.randn(3), 1: np.random.randn(3), 2: np.random.randn(3)})
    learner.addData({0:3, 1:2, 2:0}, {0: np.random.randn(3), 1: np.random.randn(3), 2: np.random.randn(3)})

    # print "Training data"
    # print learner.labels

    # add unary weights
    weights = np.random.randn((4 + 3 + 2) * d)
    # add edge weights
    weights = np.append(weights, np.random.randn(4 * 3 + 3 * 2))

    learner.setRegularization(0, 1) # gradient checking doesn't work well with the l1 regularizer
    

    objective =  learner.objective(weights)
    gradient = grad(learner.objective)
    
    print "\n\nObjective:"
    print objective
    print "\n\nGradient:"
    print gradient


    print "\n\nGradient check:"
    print check_grad(objective, gradient, weights)

    print "\n\nOptimization:"
    res = minimize(objective, weights, method='L-BFGS-B', jac = gradient)

    print res

    print "\n\nGradient check at optimized solution:"
    print check_grad(objective, gradient, res.x)


if  __name__ =='__main__':
    main()
