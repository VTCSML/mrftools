from LogLinearModel import LogLinearModel
import numpy as np
import copy
from scipy.optimize import minimize, check_grad, approx_fprime
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixCache import MatrixCache

class MatrixLogLinearMLE(object):
    """Object that runs approximate maximum likelihood parameter training."""

    def __init__(self, baseModel, inference_type):
        """Initialize learner. Pass in Markov net."""
        assert isinstance(baseModel, LogLinearModel)
        self.baseModel = baseModel
        self.models = []
        self.beliefPropagators = []
        self.labels = []
        self.l1Regularization = 0.00
        self.l2Regularization = 1
        self.featureSum = 0
        self.prevWeights = 0
        self.baseModel.create_matrices()
        self.inference_type = inference_type

    def setRegularization(self, l1, l2):
        """Set the regularization parameters."""
        self.l1Regularization = l1
        self.l2Regularization = l2

    def addData(self, states, features):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""
        example = []

        model = copy.deepcopy(self.baseModel)

        labeled_feature_mat = np.zeros((model.max_states, model.max_features))
        feature_mat = np.zeros((model.max_features, len(model.variables)))

        for (var, i) in model.var_index.items():
            labeled_feature_mat[states[var], :] += features[var]
            feature_mat[:, i] = features[var]

        model.feature_mat = feature_mat

        pairwise = np.zeros((model.max_states, model.max_states))
        for k in range(model.num_edges):
            (var, neighbor) = model.edges[k]
            pairwise[states[var], states[neighbor]] += 1

        example = np.append(labeled_feature_mat.flatten(), pairwise.flatten())

        self.models.append(model)
        self.beliefPropagators.append(self.inference_type(model))
        self.labels.append(example)
        self.featureSum += example

    def setWeights(self, weight_vector, models):
        """Set weights of Markov net from vector using the order in self.potentials."""

        max_features = self.baseModel.max_features
        num_vars = len(self.baseModel.variables)
        max_states = self.baseModel.max_states
        num_edges = self.baseModel.num_edges

        feature_size = max_features * max_states

        feature_weights = weight_vector[:feature_size].reshape((max_features, max_states))

        pairwise_weights = weight_vector[feature_size:].reshape((max_states, max_states, 1)) * np.ones((1, 1, num_edges))

        for model in models:
            model.set_weight_matrix(feature_weights)
            model.set_edge_tensor(pairwise_weights)

            model.set_unary_matrix()


    def do_inference(self, belief_propagators):
        for i in range(len(self.labels)):
            bp = self.beliefPropagators[i]
            bp.infer(display='off')

    def getFeatureExpectations(self, beliefPropagators):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        """
        marginalSum = 0
        for i in range(len(self.labels)):
            bp = beliefPropagators[i]
            marginalSum += bp.get_feature_expectations()

        return marginalSum / len(self.labels)

    def objective(self, weightVector):
        """Compute the learning objective with the provided weight vector."""
        self.setWeights(weightVector, self.models)
        self.do_inference(self.beliefPropagators)
        featureExpectations = self.getFeatureExpectations(self.beliefPropagators)

        objective = 0.0

        # add regularization penalties
        objective += self.l1Regularization * np.sum(np.abs(weightVector))
        objective += 0.5 * self.l2Regularization * weightVector.dot(weightVector)

        # add likelihood penalty
        objective -= weightVector.dot(self.featureSum / len(self.labels))

        for bp in self.beliefPropagators:
            objective += bp.compute_energy_functional() / len(self.labels)
            # print "Finished one inference"

        return objective

    def gradient(self, weightVector):
        """Compute the gradient for the provided weight vector."""
        self.setWeights(weightVector, self.models)
        inferredExpectations = self.getFeatureExpectations(self.beliefPropagators)

        gradient = np.zeros(len(weightVector))

        # add regularization penalties
        gradient += self.l1Regularization * np.sign(weightVector)
        gradient += self.l2Regularization * weightVector

        # add likelihood penalty
        gradient += inferredExpectations - self.featureSum / len(self.labels)

        return gradient

def main():
    """Simple test function for maximum likelihood."""

    np.set_printoptions(precision=3)

    model = LogLinearModel()

    np.random.seed(1)

    model.declareVariable(0, 4)
    model.declareVariable(1, 4)
    model.declareVariable(2, 4)

    d = 2

    model.setUnaryWeights(0, np.random.randn(4, d))
    model.setUnaryWeights(1, np.random.randn(4, d))
    model.setUnaryWeights(2, np.random.randn(4, d))

    model.setUnaryFeatures(0, np.random.randn(d))
    model.setUnaryFeatures(1, np.random.randn(d))
    model.setUnaryFeatures(2, np.random.randn(d))

    model.setAllUnaryFactors()

    model.setEdgeFactor((0,1), np.zeros((4, 4)))
    model.setEdgeFactor((1,2), np.zeros((4, 4)))

    from TemplatedLogLinearMLE import TemplatedLogLinearMLE

    learner = MatrixLogLinearMLE(model, MatrixBeliefPropagator)

    data = [({0: 0, 1: 0, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
            ({0: 1, 1: 1, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
            ({0: 2, 1: 0, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
            ({0: 3, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

    # add unary weights
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))

    import time

    print(learner)

    for (states, features) in data:
        learner.addData(states, features)

    learner.setRegularization(.2, 1)

    print "\n\nObjective:"
    print learner.objective(weights)
    print "\n\nGradient:"
    print learner.gradient(weights)


    print "\n\nGradient check:"
    print check_grad(learner.objective, learner.gradient, weights)

    t0 = time.time()

    print "\n\nOptimization:"


    res = minimize(learner.objective, weights, method='L-BFGS-b', jac = learner.gradient)

    t1 = time.time()

    print("Optimization took %f seconds" % (t1 - t0))

    print res

    print "\n\nGradient check at optimized solution:"
    print check_grad(learner.objective, learner.gradient, res.x * (1 + 1e-9))


if  __name__ =='__main__':
    main()
