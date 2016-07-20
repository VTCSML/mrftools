import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
# from PIL.ImageGrab import grab

class EM(Learner):
    
    def __init__(self, inference_type):
        super(EM, self).__init__( inference_type)
        
    def learn(self,weights):
        old_weights = np.inf
        new_weights = weights
        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            self.e_step(new_weights)
            new_weights = self.m_step(new_weights)
            
        return new_weights

    def e_step(self, weights):
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)

    def m_step(self, weights):
        res = minimize(self.objective, weights, None, method='L-BFGS-B', jac = self.gradient, callback=self.callback_f)
        return res.x
    
    
    
def main():
    """Simple test_functions function for maximum likelihood."""

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

    learner = EM(model,MatrixBeliefPropagator)
    
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

    learner.setRegularization(.2, 1)
        
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))
    
    weights = learner.learn(weights)
    
    print  weights
    

if  __name__ =='__main__':
    main()
    
    