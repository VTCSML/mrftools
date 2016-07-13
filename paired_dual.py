import copy
import time
from _hashlib import new

import numpy as np
from scipy.optimize import minimize, check_grad

from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixLogLinearMLE import MatrixLogLinearMLE
from Learner import Learner
# import opt
# from PIL.ImageGrab import grab

class paired_dual(Learner):
    def __init__(self,baseModel,inference_type):
        super(paired_dual, self).__init__(baseModel,inference_type)
        


    def adagrad(self, x, args):
        t = 1
        tolerance = 1e-8
        max_iter = 500
        change = np.inf
    
        grad_sum = 0
        while change > tolerance and t < max_iter:
            self.subgrad_obj(x, 'paired')
            old_x = x
            g = self.subgrad_grad(x, args)
            grad_sum += g * g
            x = x - 1.0 * g / (np.sqrt(grad_sum) + 0.001)
            change = np.sum(np.abs(x - old_x))
            t += 1
#             callback(x)
            # print func(x, args)
            # print change
        return x


    def Learn(self, weights):
        return self.adagrad(weights, 'paired')
    
    
    #     def Learn(self,weights):
#         old_weights = np.inf
#         new_weights = weights
#  
#         while not np.allclose(old_weights, new_weights):
#             old_weights = new_weights
#             res = minimize(self.subgrad_obj, new_weights ,args = 'paired', method='L-BFGS-B', jac = self.subgrad_grad,callback=self.callbackF)
#             new_weights = res.x
#  
#         return new_weights




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

#     from TemplatedLogLinearMLE import TemplatedLogLinearMLE

    learner = paired_dual(model,MatrixBeliefPropagator)
    
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
        learner.addData(states, features)

    learner.setRegularization(.2, 1)
        
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))
    
    weights = learner.Learn(weights)
    
    print  weights
    

if  __name__ =='__main__':
    main()