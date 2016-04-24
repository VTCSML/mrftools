import autograd.numpy as np
from LogLinearMLE import LogLinearMLE

class TemplatedLogLinearMLE(LogLinearMLE):
    
    def __init__(self,baseModel):
        super(TemplatedLogLinearMLE, self).__init__(baseModel)
    
    def createFullWeightVector(self, weightVector):
        var = next(iter(self.baseModel.variables))
        numStates = self.baseModel.numStates[var]
        numFeatures = self.baseModel.numFeatures[var]
        
        assert len(weightVector) == numStates * numStates + numStates * numFeatures
        
        unaryWeights = list(weightVector[0:numStates * numFeatures])
        pairWeights = list(weightVector[numStates * numFeatures:])
        
        fullWeightVector = []
        
        for i in range(len(self.potentials)):
            if self.potentials[i] in self.baseModel.variables:
                # set unary potential
                fullWeightVector.extend(unaryWeights)
            else:
                # set pairwise potential
                fullWeightVector.extend(pairWeights)
        
        return np.array(fullWeightVector)
    
    def objective(self, weightVector):
        
        fullWeightVector = self.createFullWeightVector(weightVector)
        
        return super(TemplatedLogLinearMLE, self).objective(fullWeightVector)
    
    def gradient(self, weightVector):
        
        fullWeightVector = self.createFullWeightVector(weightVector)
        
        fullGradient = super(TemplatedLogLinearMLE, self).gradient(fullWeightVector)
        
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

