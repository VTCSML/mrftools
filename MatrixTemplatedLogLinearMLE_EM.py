import copy
import time
from _hashlib import new

import numpy as np
from scipy.optimize import minimize, check_grad

from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from MatrixLogLinearMLE import MatrixLogLinearMLE


class MatrixTemplatedLogLinearMLE_EM(MatrixLogLinearMLE):
        
    def __init__(self,baseModel,inference_type):
        super(MatrixTemplatedLogLinearMLE_EM, self).__init__(baseModel,inference_type)
        self.tau_q = []
        self.tau_p = []
        self.H_p = 0
        self.H_q = 0
        self.beliefPropagators_q = []
        self.models_q = []
        self.term_q_p = 0
        self.bpIter = 1
        self.weight_record = np.array([])
        self.time_record = np.array([])
        self.inference_type = inference_type


    def addData(self, states, features):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""

        model = copy.deepcopy(self.baseModel)
        model_q = copy.deepcopy(self.baseModel)

        feature_mat = np.zeros((model.max_features, len(model.variables)))

        label_mask = -float('inf') * np.ones((model.max_states, len(model.variables)))
        for (var, i) in model.var_index.items():
            feature_mat[:, i] = features[var]
            if states[var] != -100:
                label_mask[states[var], i] = 0
            else:
                label_mask[:, i] = 0

        model.feature_mat = feature_mat
        model_q.feature_mat = feature_mat

        self.models.append(model)
        self.beliefPropagators.append(self.inference_type(model))

        self.models_q.append(model_q)
        self.beliefPropagators_q.append(self.inference_type(model_q))
        self.labels.append(label_mask)
        
    def do_inference(self, belief_propagators, method):
        for i in range(len(self.labels)):
            bp = self.beliefPropagators[i]
            if method == 'subgradient' or method == 'EM':
                bp.infer(display = 'off')
            elif method == 'paired':
                bp.infer(display = 'off', maxIter = self.bpIter)

    def getFeatureExpectations(self, beliefPropagators):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        """
        marginalSum = 0
        for i in range(len(self.labels)):
            bp = beliefPropagators[i]
            marginalSum += bp.get_feature_expectations()
            
        return marginalSum / len(self.labels)

    def getBetheEntropy(self,belief_propagator):
        bethe = 0
        for i in range(len(self.labels)):
            bp = belief_propagator[i]
            bp.computeBeliefs()
            bp.computePairwiseBeliefs()
            bethe += bp.compute_bethe_entropy()
            
        bethe = bethe / len(self.labels)
        return bethe

    def subgrad_obj(self,weights,method):
        self.E_step(weights,method,True)
        return self.Objective(weights,method)
    
    def subgrad_grad(self,weights,method):
        self.E_step(weights,method,False)
        return self.Gradient(weights,method)
    
    def paired_dual_learning(self, weights):
        old_weights = np.inf
        new_weights = weights

        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            res = minimize(self.subgrad_obj, new_weights ,args = 'paired', method='L-BFGS-B', jac = self.subgrad_grad,callback=self.callbackF)
            new_weights = res.x

        return new_weights
        
    def subGradient(self,weights):
        old_weights = np.inf
        new_weights = weights

        while not np.allclose(old_weights, new_weights):
            old_weights = new_weights
            res = minimize(self.subgrad_obj, new_weights, args='subgradient', method='L-BFGS-B', jac=self.subgrad_grad,
                           callback=self.callbackF)
            new_weights = res.x
            # print check_grad(self.subgrad_obj, self.subgrad_grad, new_weights, 'subgradient')

        return new_weights

    def reset(self):
        self.weight_record =  np.array([])
        self.time_record = np.array([])
        for bp in self.beliefPropagators + self.beliefPropagators_q:
            bp.initialize_messages()

    def callbackF(self,w):
        a = np.copy(w)
        if (self.weight_record.size) == 0:
            self.weight_record = a.reshape((1, a.size))
            self.time_record = np.array([int(round(time.time() * 1000))])
        else:
            self.weight_record = np.vstack((self.weight_record,a))
            self.time_record = np.vstack((self.time_record,int(round(time.time() * 1000))))

    def calculate_tau(self, weights, method, mode,should_infere):
        if mode == 'q':
            self.setWeights(weights,self.models_q)
            if should_infere == True:
                self.do_inference(self.beliefPropagators_q,method)

            for i in range(len(self.labels)):
                self.models_q[i].unary_mat += self.labels[i]

            self.tau_q = self.getFeatureExpectations(self.beliefPropagators_q)
            self.H_q = self.getBetheEntropy(self.beliefPropagators_q)
        elif mode == 'p':
            self.setWeights(weights, self.models)
            if should_infere == True:
                self.do_inference(self.beliefPropagators,method)
            self.tau_p = self.getFeatureExpectations(self.beliefPropagators)
            self.H_p = self.getBetheEntropy(self.beliefPropagators)

    def E_step(self,weights,method,should_infere):
#         print 'E_step................'
        self.calculate_tau(weights,method,'q',should_infere)

    def M_step(self,weights):
        # print check_grad(self.Objective, self.Gradient, weights, 'EM')
        res = minimize(self.Objective, weights ,args = 'EM', method='L-BFGS-B', jac = self.Gradient,callback=self.callbackF)
        return res.x
    
    def Objective(self, weights, method):
        self.calculate_tau(weights, method, 'p',True)

        term_p = weights.dot(self.tau_p) + self.H_p
        term_q = weights.dot(self.tau_q) + self.H_q
        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1Regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2Regularization * weights.dot(weights)
        objec += self.term_q_p

#         print objec
        return objec
        
    def Gradient(self,weights,method):
        self.calculate_tau(weights, method, 'p',False)

        grad = np.zeros(len(weights))
      
        # add regularization penalties
        grad += self.l1Regularization * np.sign(weights)
        grad += self.l2Regularization * weights
      
        grad -= np.squeeze(self.tau_q)
        grad += np.squeeze(self.tau_p)
      
        return grad
    
    
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

    learner = MatrixTemplatedLogLinearMLE_EM(model,MatrixBeliefPropagator)
    
#     data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

    data = [({0: 0, 1: 0, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
             ({0: 1, 1: 1, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
             ({0: 2, 1: 0, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
             ({0: 3, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

    # add unary weights
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))

#     print(learner)

    for (states, features) in data:
        learner.addData(states, features)

    learner.setRegularization(.2, 1)
    
    for i in range(10):
        old_weights  = weights
        # =====================================
        # E-step: inference
        # =====================================
        learner.E_step(weights,'EM',True)
              
                    # =====================================
                    # M-step: learning parameters
                    # =====================================
        weights = learner.M_step(weights)
        
    print weights
    
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))
    
    weights = learner.subGradient(weights)
    
    print weights
    
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))
    
    weights = learner.paired_dual_learning(weights)
    
    print  weights


if  __name__ =='__main__':
    main()
    
    
    
    
    