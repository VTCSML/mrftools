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


class TemplatedLogLinearMLE_EM(TemplatedLogLinearMLE):
    
    def __init__(self,baseModel):
        super(TemplatedLogLinearMLE, self).__init__(baseModel)
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
        
    def addData(self, states, features):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""
        example = []
        model = copy.deepcopy(self.baseModel)
        model_p = copy.deepcopy(self.baseModel)
        
        # create vector representation of data using the same order as self.potentials
        for i in range(len(self.potentials)):
            if self.potentials[i] in self.baseModel.variables:
                # set unary data
                var = self.potentials[i]
                
#                 table = np.zeros((model.numStates[var], len(features[var])))
                model_p.setUnaryFeatures(var, features[var])
                # set model features
                if states[var] ==  -100:
                    model.setUnaryFeatures(var, features[var])
                else:
#                     table[states[var],:] = features[var]
                    factor = np.zeros(model.numStates[var]) -float('Inf')
#                     factor = np.zeros(model.numStates[var]) 
                    indx = states[var]
                    factor[indx] = 0
#                     factor[indx] = 1    
                    model.setUnaryFactor(var,factor)
            
        
        
            # flatten table and append
#             example.extend(table.reshape((-1, 1)).tolist())
        
        self.models.append(model)
        bp = BeliefPropagator(model)
        self.beliefPropagators.append(bp)
#         self.labels.append(np.array(example))
#         self.featureSum += np.array(example)
        self.labels.append(0)

        self.models_p.append(model_p)
        bp = BeliefPropagator(model_p)
        self.beliefPropagators_p.append(bp)

    def getBetheEntropy(self,bp,method):
        betheEntropy = 0
        if self.needInference:
            if method == 'paired':
                bp.runInference(display = 'off', maxIter = self.bpIter)
            elif method == 'EM' or method == 'subgradient':
                bp.runInference(display = 'off')
            bp.computeBeliefs()
            bp.computePairwiseBeliefs()
            betheEntropy = bp.computeBetheEntropy()
        
        return betheEntropy

    def getFeatureExpectations(self,mode,method):
        """Run inference and return the marginal in vector form using the order of self.potentials.
            :rtype: numpy.ndarray
            """
        marginalSum = 0
        if mode == 'q':
            self.H_q = 0
        elif mode == 'p':
            self.H_p = 0
        for i in range(len(self.labels)):
            if mode == 'q':
                bp = self.beliefPropagators[i]
                model = self.models[i]
                self.H_q += self.getBetheEntropy(bp,method)
            elif mode == 'p':
                bp = self.beliefPropagators_p[i]
                model = self.models_p[i]
                self.H_p += self.getBetheEntropy(bp,method)
        
            # make vector form of marginals
            marginals = []
            for j in range(len(self.potentials)):
                if self.potentials[j] not in self.baseModel.variables:
                    # get pairwise belief
                    table = np.exp(bp.pairBeliefs[self.potentials[j]])
                else:
                    # get unary belief and multiply by features
                    var = self.potentials[j]
                    table = np.outer(np.exp(bp.varBeliefs[var]), model.unaryFeatures[var])
                
                # flatten table and append
                marginals.extend(table.reshape((-1, 1)).tolist())
            marginalSum += np.array(marginals)

        if mode =='q':
            self.H_q = self.H_q / len(self.labels)
        elif mode == 'p':
            self.H_p = self.H_p / len(self.labels)
        
        
        return marginalSum / len(self.labels)

    def setWeights(self, weightVector,mode):
        """Set weights of Markov net from vector using the order in self.potentials."""
#         if np.array_equal(weightVector, self.prevWeights):
#             # if using the same weight vector as previously, there is no need to rerun inference
#             # this often happens when computing the objective and the gradient with the same weights
#             self.needInference = False
#             return
#         
#         self.prevWeights = weightVector
#         self.needInference = True
        if mode == 'q':
            weightCache = MatrixCache()
            for model in self.models:
                j = 0
                for i in range(len(self.potentials)):
                    if self.potentials[i] not in self.baseModel.variables:
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
                        if -float('Inf') not in model.unaryPotentials[var]:
                            factorWeights = weightCache.getCached(weightVector[j:j + np.prod(size)].reshape(size))
                            #                    model.setUnaryWeights(var, factorWeights)
                            fac = factorWeights.dot(model.unaryFeatures[var])
                            model.setUnaryFactor(var,fac)
                        else:
                            factorWeights = weightCache.getCached(weightVector[j:j + np.prod(size)].reshape(size))
                            #                    model.setUnaryWeights(var, factorWeights)
                            fac = factorWeights.dot(model.unaryFeatures[var])
                            ind = np.where(model.unaryPotentials[var] != -float('Inf'))
                            a = np.zeros(model.numStates[var]) -float('Inf')
                            a[ind] = fac[ind]
                            model.setUnaryFactor(var,fac)
                            
                        j += np.prod(size)
            
            assert j == len(weightVector)
        
        elif mode == 'p':
            weightCache = MatrixCache()
            for model in self.models_p:
                j = 0
                for i in range(len(self.potentials)):
                    if self.potentials[i] not in self.baseModel.variables:
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
#                         fac = factorWeights.dot(model.unaryFeatures[var])
#                         model.setUnaryFactor(var,fac)
                        j += np.prod(size)
                #                    print model.unaryPotentials[var]
                model.setAllUnaryFactors()

                assert j == len(weightVector)

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
        fullWeightVector = self.createFullWeightVector(weights)
        self.setWeights(fullWeightVector,mode)
        tau = self.getFeatureExpectations(mode,method)
        if mode == 'q':
            self.tau_q = tau
        elif mode == 'p':
            self.tau_p = tau
        
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
        fullWeightVector = self.createFullWeightVector(weights)

        term_p = fullWeightVector.dot(self.tau_p) + self.H_p
        term_q = -(fullWeightVector.dot(self.tau_q) + self.H_q)
        self.term_q_p = term_q+term_p
        
        objec = 0.0
        # add regularization penalties
        objec += self.l1Regularization * np.sum(np.abs(fullWeightVector))
        objec += 0.5 * self.l2Regularization * fullWeightVector.dot(fullWeightVector)
        objec += self.term_q_p

        return objec
        
    def Gradient(self,weights,method):
        #   Gradient
        fullWeightVector = self.createFullWeightVector(weights)
        self.setWeights(fullWeightVector,'p')
        
        grad = np.zeros(len(fullWeightVector))
      
      # add regularization penalties
        grad += self.l1Regularization * np.sign(fullWeightVector)
        grad += self.l2Regularization * fullWeightVector
      
        grad -=  np.squeeze(self.tau_q)
        grad +=  np.squeeze(self.tau_p)
      
        fullGradient =  grad
                  
        
        var = next(iter(self.baseModel.variables))
        numStates = self.baseModel.numStates[var]
        numFeatures = self.baseModel.numFeatures[var]
      
        unaryGradient = np.zeros(numStates * numFeatures)
        pairwiseGradient = np.zeros(numStates * numStates)
      
      # aggregate gradient to templated dimensions
        j = 0
        for i in range(len(self.potentials)):
            if self.potentials[i] in self.baseModel.variables:
            # set unary potential
                var = self.potentials[i]
                size = (numStates, numFeatures)
                unaryGradient += fullGradient[j:j + np.prod(size)]
                j += np.prod(size)
            else:
                # set pairwise potential
                pair = self.potentials[i]
                size = (numStates, numStates)
                pairwiseGradient += fullGradient[j:j + np.prod(size)]
                j += np.prod(size)
              
        grad = np.append(unaryGradient, pairwiseGradient)
        return grad
    
    
    