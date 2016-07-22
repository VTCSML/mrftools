import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator


class Learner(object):
    def __init__(self,inference_type):
        self.tau_q = []
        self.tau_p = []
        self.models_q = []
        self.bpIter = 1
        self.weight_record = np.array([])
        self.time_record = np.array([])
        self.inference_type = inference_type
        self.num_examples = 0
        self.models = []
        self.models_q = []
        self.belief_propagators_q = []
        self.belief_propagators = []
        self.l1_regularization = 0.00
        self.l2_regularization = 1


    def set_regularization(self, l1, l2):
        """Set the regularization parameters."""
        self.l1_regularization = l1
        self.l2_regularization = l2
        
    def add_data(self, labels, model):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the
         unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""

        self.models.append(model)
        self.belief_propagators.append(self.inference_type(model))

        model_q = copy.deepcopy(model)
        self.models_q.append(model_q)

        bp_q = self.inference_type(model_q)
        for (var, state) in labels.items():
            bp_q.condition(var, state)
        
        self.belief_propagators_q.append(bp_q)
        
        self.num_examples += 1
        
    def do_inference(self, belief_propagators):
        for bp in belief_propagators:
            bp.infer(display = 'off')
        
    def get_feature_expectations(self, belief_propagators):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        """
        marginal_sum = 0
        for i in range(self.num_examples):
            bp = belief_propagators[i]
            marginal_sum += bp.get_feature_expectations()
            
        return marginal_sum / self.num_examples
    
    def get_bethe_entropy(self, belief_propagators):
        bethe = 0
        for bp in belief_propagators:
            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()
            bethe += bp.compute_bethe_entropy()
            
        bethe = bethe / self.num_examples
        return bethe
    
    def subgrad_obj(self, weights, options=None):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        return self.objective(weights)
    
    def subgrad_grad(self, weights, options=None):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, False)
        return self.gradient(weights)
    
    def learn(self, weights,callback_f):
        old_weights = np.inf
        new_weights = weights

        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            res = minimize(self.subgrad_obj, new_weights, method='L-BFGS-B', jac=self.subgrad_grad, callback=callback_f)
            new_weights = res.x

        return new_weights

    def reset(self):
        self.weight_record =  np.array([])
        self.time_record = np.array([])
        for bp in self.belief_propagators + self.belief_propagators_q:
            bp.initialize_messages()

    def set_weights(self, weight_vector, belief_propagators):
        """Set weights of Markov net from vector using the order in self.potentials."""
        for bp in belief_propagators:
            bp.mn.set_weights(weight_vector)

    def calculate_tau(self, weights, belief_propagators, should_infer=True):
        self.set_weights(weights, belief_propagators)
        if should_infer:
            self.do_inference(belief_propagators)

        return self.get_feature_expectations(belief_propagators)

    def objective(self, weights, options=None):
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)
        self.set_weights(weights, self.belief_propagators_q)

        term_p = sum([x.compute_energy_functional() for x in self.belief_propagators]) / self.num_examples
        term_q = sum([x.compute_energy_functional() for x in self.belief_propagators_q]) / self.num_examples
        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        return objec
        
    def gradient(self, weights, options=None):
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, False)
        self.set_weights(weights, self.belief_propagators_q)

        grad = np.zeros(len(weights))
      
        # add regularization penalties
        grad += self.l1_regularization * np.sign(weights)
        grad += self.l2_regularization * weights
      
        grad -= np.squeeze(self.tau_q)
        grad += np.squeeze(self.tau_p)
      
        return grad 
    
    
def main():
    """Simple test function for maximum likelihood."""

    np.set_printoptions(precision=3)

    model = LogLinearModel()

    np.random.seed(1)

    model.declare_variable(0, 4)
    model.declare_variable(1, 4)
    model.declare_variable(2, 4)

    d = 2

    model.set_unary_weights(0, np.random.randn(4, d))
    model.set_unary_weights(1, np.random.randn(4, d))
    model.set_unary_weights(2, np.random.randn(4, d))

    model.set_unary_features(0, np.random.randn(d))
    model.set_unary_features(1, np.random.randn(d))
    model.set_unary_features(2, np.random.randn(d))

    model.set_all_unary_factors()

    model.set_edge_factor((0, 1), np.zeros((4, 4)))
    model.set_edge_factor((1, 2), np.zeros((4, 4)))

#     from TemplatedLogLinearMLE import TemplatedLogLinearMLE

    learner = Learner(model,MatrixBeliefPropagator)
    
    data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

#    data = [({0: 0, 1: 0, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
#             ({0: 1, 1: 1, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
#             ({0: 2, 1: 0, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
#             ({0: 3, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

    # add unary weights
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))

#     print(learner)

    for (states, features) in data:
#         print 'dataaaaaaaa'
        learner.add_data(states, features)

    learner.set_regularization(.2, 1)
    
    
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))
    
    weights = learner.learn(weights)
    
    print weights
    


if  __name__ =='__main__':
    main()
    
    
    
        
        
        
    
