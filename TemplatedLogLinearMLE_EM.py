import numpy as np
from LogLinearMLE import LogLinearMLE
from TemplatedLogLinearMLE import TemplatedLogLinearMLE
import copy
from scipy.optimize import minimize, check_grad
from BeliefPropagator import BeliefPropagator
from MatrixCache import MatrixCache

class TemplatedLogLinearMLE_EM(TemplatedLogLinearMLE):
    
    def __init__(self,baseModel):
        super(TemplatedLogLinearMLE, self).__init__(baseModel)
        self.tau_q = []
        self.tau_p = []
        self.beliefPropagators_p = []
        self.H_p = 0
        self.H_q = 0



    def createFullWeightVector(self, weightVector):
        var = next(iter(self.baseModel.variables))
        numStates = self.baseModel.numStates[var]
        numFeatures = self.baseModel.numFeatures[var]
        
        assert len(weightVector) == numStates * numStates + numStates * numFeatures
        
        unaryWeights = weightVector[0:numStates * numFeatures].tolist()
        pairWeights = weightVector[numStates * numFeatures:].tolist()
        
        fullWeightVector = []
        
        for i in range(len(self.potentials)):
            if self.potentials[i] in self.baseModel.variables:
                # set unary potential
                fullWeightVector.extend(unaryWeights)
            else:
                # set pairwise potential
                fullWeightVector.extend(pairWeights)
        
        return np.array(fullWeightVector)



    def addData(self, states, features):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""
        example = []
        H_q = 0
        model = copy.deepcopy(self.baseModel)
        
        # create vector representation of data using the same order as self.potentials
        for i in range(len(self.potentials)):
            if self.potentials[i] not in self.baseModel.variables:
                # set pairwise state
                pair = self.potentials[i]

                table = np.zeros((model.numStates[pair[0]], model.numStates[pair[1]]))
                if states[pair[0]] != -100 and states[pair[1]] != -100:
                    table[states[pair[0]], states[pair[1]]] = 1
            else:
                # set unary data
                var = self.potentials[i]
                
                table = np.zeros((model.numStates[var], len(features[var])))
                if states[var] != -100:
                    table[states[var],:] = features[var]
                
                # set model features
                if states[var] == -100:
                    model.setUnaryFeatures(var, features[var])
                else:
                    factor = np.zeros(model.numStates[var])
                    indx = states[var]
                    factor[indx] = 1
    #                print 'index: ' + str(indx)
    #                print 'Unary Factor: '+ str(factor_q)
                    model.setUnaryFactor(var,factor)
            
            
            # flatten table and append
            example.extend(table.reshape((-1, 1)).tolist())
        
        self.models.append(model)
        bp = BeliefPropagator(model)
        self.beliefPropagators.append(bp)
        self.labels.append(np.array(example))
        self.featureSum += np.array(example)
        
        self.tau_q = self.getFeatureExpectations()
        self.H_q += bp.computeBetheEntropy()
        
            
            
            
            
    def addData_p(self, states, features):
        example = []
        model = copy.deepcopy(self.baseModel)
        
        # create vector representation of data using the same order as self.potentials
        for i in range(len(self.potentials)):
            if self.potentials[i] not in self.baseModel.variables:
                # set pairwise state
                pair = self.potentials[i]
                
                table = np.zeros((model.numStates[pair[0]], model.numStates[pair[1]]))
                if states[pair[0]] != -100 and states[pair[1]] != -100:
                    table[states[pair[0]], states[pair[1]]] = 1
            else:
                # set unary data
                var = self.potentials[i]
                
                table = np.zeros((model.numStates[var], len(features[var])))
                if states[var] != -100:
                    table[states[var],:] = features[var]
                
                # set model features
                if states[var] != -100:
                    model.setUnaryFeatures(var, features[var])
                else:
                    factor = np.zeros(model.numStates[var])
                    indx = states[var]
                    factor[indx] = 1
                    #                print 'index: ' + str(indx)
                    #                print 'Unary Factor: '+ str(factor_q)
                    model.setUnaryFactor(var,factor)
            
            
            # flatten table and append
            example.extend(table.reshape((-1, 1)).tolist())
        
        self.beliefPropagators = []
        self.models = []
        self.labels = []
        self.featureSum = 0
        
        self.models.append(model)
        bp = BeliefPropagator(model)
        self.beliefPropagators.append(bp)
        self.labels.append(np.array(example))
        self.featureSum += np.array(example)


        self.tau_p = self.getFeatureExpectations()
        self.H_p += bp.computeBetheEntropy()


    




    def getlabels(self):
        """Run inference and return the labels.
            :rtype: numpy.ndarray
            """
        Z = []
        for i in range(len(self.labels)):
            bp = self.beliefPropagators[i]
            model = self.models[i]
            if self.needInference:
                bp.runInference(display = 'off')
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                Z1 = []
                # make vector form of labels
                Z1 = []
                
                for item in self.potentials:
                    if isinstance(item, tuple) == False:
                        Z1.append(np.argmax(bp.varBeliefs[item]))
#
                Z.append(Z1)

        
        return Z




    def objective(self, weightVector1):
        """Compute the learning objective with the provided weight vector."""


        fullWeightVector = self.createFullWeightVector(weightVector1)
        
#        return super(TemplatedLogLinearMLE, self).objective(fullWeightVector)
        weightVector = fullWeightVector
    
#        self.setWeights(weightVector)
#        featureExpectations = self.getFeatureExpectations()

        objective = 0.0
        
        # add regularization penalties
        objective += self.l1Regularization * np.sum(np.abs(weightVector))
        objective += 0.5 * self.l2Regularization * weightVector.dot(weightVector)
        
        objective += weightVector.dot(self.tau_q) + self.H_q
        
        objective -= weightVector.dot(self.tau_p) - self.H_p
        
        # add likelihood penalty
#        objective -= weightVector.dot(self.featureSum / len(self.labels))

#        for bp in self.beliefPropagators:
#            objective += bp.computeEnergyFunctional() / len(self.labels)
        # print "Finished one inference"
        
        
        return objective
    
    
    
    
    
    def gradient(self, weightVector1):
        
        fullWeightVector = self.createFullWeightVector(weightVector1)
        weightVector = fullWeightVector
        
#        fullGradient = super(TemplatedLogLinearMLE, self).gradient(fullWeightVector)


#        self.setWeights(weightVector)
#        inferredExpectations = self.getFeatureExpectations()

        gradient = np.zeros(len(weightVector))

        # add regularization penalties
        gradient += self.l1Regularization * np.sign(weightVector)
        gradient += self.l2Regularization * weightVector

        # add likelihood penalty
#        gradient += np.squeeze(inferredExpectations - self.featureSum / len(self.labels))
        gradient +=  np.squeeze(self.tau_q)
        gradient -=  np.squeeze(self.tau_p)

        fullGradient =   gradient
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
        
            return np.append(unaryGradient, pairwiseGradient)








