import numpy as np
from LogLinearMLE import LogLinearMLE
from TemplatedLogLinearMLE import TemplatedLogLinearMLE
import copy
from scipy.optimize import minimize, check_grad
from BeliefPropagator import BeliefPropagator
from MatrixCache import MatrixCache
import math
import time
from matplotlib.font_manager import weight_as_number
from MatrixLogLinearMLE import MatrixLogLinearMLE
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator



class MatrixTemplatedLogLinearMLE_EM(MatrixLogLinearMLE):
        
    def __init__(self,baseModel):
        super(MatrixTemplatedLogLinearMLE_EM, self).__init__(baseModel)
        self.tau_q = []
        self.tau_p = []
        self.H_p = 0
        self.H_q = 0
        self.beliefPropagators_p = []
        self.models_p = []
        self.term_q_p = 0
        self.bpIter = 1
        self.weight_record =  np.array([])
        self.time_record = np.array([])
        self.needInference = True
        self.varBelief_p = {}
        self.unary_mult_mat = np.array([])
        self.unary_mult_dic = {}

    def addData(self, states, features):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""

        example = []

        model = copy.deepcopy(self.baseModel)
        model_p = copy.deepcopy(self.baseModel)


        labeled_feature_mat = np.zeros((model.max_states, model.max_features))
        feature_mat = np.zeros((model.max_features, len(model.variables)))

        self.unary_mult_mat = np.ones((model.max_states,len(model.variables)))
        for (var, i) in model.var_index.items():
            feature_mat[:, i] = features[var]
            if states[var] != -100:
                factor = np.zeros(model.max_states) -float('Inf')
                indx = states[var]
                factor[indx] = 0
                self.unary_mult_mat[:,i] = factor
                self.unary_mult_dic[model] = self.unary_mult_mat
        

        model.feature_mat = feature_mat
        model_p.feature_mat = feature_mat

        self.models.append(model)
        self.beliefPropagators.append(MatrixBeliefPropagator(model))
        self.labels.append(0)
        
        self.models_p.append(model_p)
        self.beliefPropagators_p.append(MatrixBeliefPropagator(model_p))

    def calculate_bethe_entropy(self,bp,method):
        betheEntropy = 0
        if method == 'paired':
            bp.runInference(display = 'off', maxIter = self.bpIter)
        elif method == 'EM' or method == 'subgradient':
            bp.runInference(display = 'off')
        bp.computeBeliefs()
        bp.computePairwiseBeliefs()
        betheEntropy = bp.computeBetheEntropy()
        
        return betheEntropy
    
    def getBetheEntropy(self,belief_propagator,method):
        bethe = 0
        for i in range(len(self.labels)):
            bp = belief_propagator[i]
            bethe += self.calculate_bethe_entropy(bp,method)
            
        bethe = bethe / len(self.labels)
        return bethe
        
    def getFeatureExpectations(self, beliefPropagators):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        """
        marginalSum = 0
        for i in range(len(self.labels)):
            bp = beliefPropagators[i]
            model = bp.mn
            if self.needInference:
                bp.runInference(display = 'off')
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                
            summed_features = np.inner(np.exp(bp.belief_mat), model.feature_mat).T

            summed_pair_features = np.sum(np.exp(bp.pair_belief_tensor), 2).T

            marginals = np.append(summed_features.reshape(-1), summed_pair_features.reshape(-1))

            marginalSum += marginals

        return marginalSum / len(self.labels)

    def setWeights(self, weight_vector, models):
        """Set weights of Markov net from vector using the order in self.potentials."""
        if np.array_equal(weight_vector, self.prevWeights):
            # if using the same weight vector as previously, there is no need to rerun inference
            # this often happens when computing the objective and the gradient with the same weights
            self.needInference = False
            return

        self.prevWeights = weight_vector
        self.needInference = True

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
            if model in self.unary_mult_dic.keys():
                model.unary_mat = np.multiply(model.unary_mat,self.unary_mult_dic[model])
                model.unary_mat[model.unary_mat == float('Inf')] = -float('Inf')

    def subgrad_obj(self,weights,method):
        self.E_step(weights,method)
        return self.Objective(weights,method)
    
    def subgrad_grad(self,weights,method):
        self.E_step(weights,method)
        return self.Gradient(weights,method)
    
    def pairdDual_Learning(self,weights):
        res = minimize(self.subgrad_obj, weights ,args = 'paired', method='L-BFGS-B', jac = self.subgrad_grad,callback=self.callbackF)
        return res.x
        
    def subGradient(self,weights):
        res = minimize(self.subgrad_obj, weights ,args = 'subgradient', method='L-BFGS-B', jac = self.subgrad_grad,callback=self.callbackF)
        return res.x

    def clearRecord(self):
        self.weight_record =  np.array([])
        self.time_record = np.array([]) 

    def callbackF(self,w):
#         self.addWeights(w)
#         print w[0]
        a = np.copy(w)
        if (self.weight_record.size) == 0:
            self.weight_record = a
            self.time_record = int(round(time.time() * 1000))
        else:
            self.weight_record = np.vstack((self.weight_record,a))
            self.time_record = np.vstack((self.time_record,int(round(time.time() * 1000))))

    def calculate_tau(self, weights, method, mode):
#         fullWeightVector = self.createFullWeightVector(weights)
#         self.setWeights(fullWeightVector,mode)
        if mode == 'q':
            self.setWeights(weights,self.models)
            self.tau_q = self.getFeatureExpectations(self.beliefPropagators)
            self.H_q = self.getBetheEntropy(self.beliefPropagators,method)
        elif mode == 'p':
            self.setWeights(weights,self.models_p)
            self.tau_p = self.getFeatureExpectations(self.beliefPropagators_p)
            self.H_p = self.getBetheEntropy(self.beliefPropagators_p,method)

    def E_step(self,weights,method):
#         print 'E_step................'
        
        self.calculate_tau(weights,method,'q')

    def M_step(self,weights):
        res = minimize(self.Objective, weights ,args = 'EM', method='L-BFGS-B', jac = self.Gradient,callback=self.callbackF)
        return res.x
    
    def Objective(self,weights,p_method):
        self.calculate_tau(weights,p_method,'p')
        return self.calculate_objective(weights)

    def calculate_objective(self,weights):
#         fullWeightVector = self.createFullWeightVector(weights)
        fullWeightVector = weights
        term_p = fullWeightVector.dot(self.tau_p) + self.H_p
        term_q = -(fullWeightVector.dot(self.tau_q) + self.H_q)
        self.term_q_p = term_q+term_p
        
        objec = 0.0
        # add regularization penalties
        objec += self.l1Regularization * np.sum(np.abs(fullWeightVector))
        objec += 0.5 * self.l2Regularization * fullWeightVector.dot(fullWeightVector)
        objec += self.term_q_p

#         print objec
        return objec
        
    def Gradient(self,weights,method):
        #   Gradient
#         fullWeightVector = self.createFullWeightVector(weights)
        fullWeightVector = weights
        self.setWeights(weights,self.models_p)        
        
        grad = np.zeros(len(fullWeightVector))
      
      # add regularization penalties
        grad += self.l1Regularization * np.sign(fullWeightVector)
        grad += self.l2Regularization * fullWeightVector
      
        grad -=  np.squeeze(self.tau_q)
        grad +=  np.squeeze(self.tau_p)
      
        return grad  
                  
#         fullGradient =  grad
#         var = next(iter(self.baseModel.variables))
#         numStates = self.baseModel.numStates[var]
#         numFeatures = self.baseModel.numFeatures[var]
#       
#         unaryGradient = np.zeros(numStates * numFeatures)
#         pairwiseGradient = np.zeros(numStates * numStates)
#       
#       # aggregate gradient to templated dimensions
#         j = 0
#         for i in range(len(self.potentials)):
#             if self.potentials[i] in self.baseModel.variables:
#             # set unary potential
#                 var = self.potentials[i]
#                 size = (numStates, numFeatures)
#                 unaryGradient += fullGradient[j:j + np.prod(size)]
#                 j += np.prod(size)
#             else:
#                 # set pairwise potential
#                 pair = self.potentials[i]
#                 size = (numStates, numStates)
#                 pairwiseGradient += fullGradient[j:j + np.prod(size)]
#                 j += np.prod(size)
#               
#         grad = np.append(unaryGradient, pairwiseGradient)
# #         grad = unaryGradient
#         return grad
    
    
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

    learner = MatrixTemplatedLogLinearMLE_EM(model)
    
    data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

#     data = [({0: 0, 1: 0, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
#             ({0: 1, 1: 1, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
#             ({0: 2, 1: 0, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
#             ({0: 3, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

    # add unary weights
    weights = np.ones(4 * d)
    # add edge weights
    weights = np.append(weights, np.ones(4 * 4))

    import time

    print(learner)

    for (states, features) in data:
        learner.addData(states, features)

    learner.setRegularization(.2, 1)
    
    for i in range(10):
        old_weights  = weights
        # =====================================
        # E-step: inference
        # =====================================
        learner.E_step(weights,'EM')
              
                    # =====================================
                    # M-step: learning parameters
                    # =====================================
        weights = learner.M_step(weights)

#     print "\n\nObjective:"
#     print learner.objective(weights)
#     print "\n\nGradient:"
#     print learner.gradient(weights)
# 
# 
#     print "\n\nGradient check:"
#     print check_grad(learner.objective, learner.gradient, weights)
# 
#     t0 = time.time()
# 
#     print "\n\nOptimization:"
# 
# 
#     res = minimize(learner.objective, weights, method='L-BFGS-b', jac = learner.gradient)
# 
#     t1 = time.time()
# 
#     print("Optimization took %f seconds" % (t1 - t0))
# 
#     print res
# 
#     print "\n\nGradient check at optimized solution:"
#     print check_grad(learner.objective, learner.gradient, res.x * (1 + 1e-9))


if  __name__ =='__main__':
    main()
    
    
    
    
    