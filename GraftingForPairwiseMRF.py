"""Class to do generative learning directly on MRF parameters."""

from BeliefPropagatorGrafting import BeliefPropagatorGrafting
from MarkovNetForGrafting import MarkovNetForGrafting
import numpy as np
from scipy.optimize import minimize, check_grad
from ApproxMaxLikelihood import ApproxMaxLikelihood
from BeliefPropagatorForGrafting import BeliefPropagatorForGrafting

class GraftingForPairwiseMRF(ApproxMaxLikelihood):
    """Object that runs approximate maximum likelihood parameter training."""

    def __init__(self,MarkovNetGrafting):
         ApproxMaxLikelihood.__init__(self,MarkovNetGrafting)
         self.bpGraft = BeliefPropagatorForGrafting(self.mn)
         self.candidateEdgeDataSum = []

    def getCandidateEdgeDataSum(self, states, mn, candidateEdgeDataSum):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the unary variables."""
        example = []
        # create vector representation of data using the same order as self.potentials
        for pair in mn.searchSpace:
            # set pairwise feature
            table = np.zeros((len(mn.unaryPotentials[pair[0]]), (len(mn.unaryPotentials[pair[1]]))))
            table[states[pair[0]], states[pair[1]]] = 1
            # flatten table and append
            example.extend(np.asarray(table.reshape((-1, 1))))
        res = candidateEdgeDataSum + np.asarray(example)
        return res

    def getMaxGradient1(self,dataLength,currSearchSpace, candidateEdgeDataSum):
        """Compute the gradient for the current weight vector """
        marginals, mapVec = self.bpGraft.computeCandidateEdgeBelief( currSearchSpace)
        gradient = (np.exp(marginals) - np.asarray(candidateEdgeDataSum) / dataLength).squeeze()
        #gradient += self.l1Regularization * np.sign(weights)
        #gradient += self.l2Regularization * weights
        selectedfeature = np.abs(gradient).argmax(axis=0)
        activatedEdge =  mapVec[selectedfeature]
        maxGrad = np.abs(gradient).max(axis=0)

        print('gradient')
        print(gradient)
        '''
        print('mapVec')
        print(mapVec)
        print('selectedfeature')
        print(selectedfeature)
        '''
        return activatedEdge, maxGrad

    def graft(self, variables, numstates, data, l1coeff):
        MapWeightsToVariables = []
        activeSet = []
        np.random.seed(0)
        numWeightsOpt = 0
        for var in variables:
            self.mn.setUnaryFactor(var, np.zeros(numstates[var]))
            numWeightsOpt += numstates[var]
            for t in range(numstates[var]):
                MapWeightsToVariables.append([var])
        aml_optimize = GraftingForPairwiseMRF(self.mn)
        aml_optimize.setRegularization(1, 0)
        self.mn.initSearchSpace()
        candidateEdgeDataSum = []
        for t in range(self.mn.numWeightsSearchSpace):
            candidateEdgeDataSum.append([0])
        for i in range(len(data)):
            aml_optimize.addData(data[i])
            candidateEdgeDataSum = self.getCandidateEdgeDataSum(data[i], self.mn, candidateEdgeDataSum)
        weightsOpt = np.random.randn(numWeightsOpt)
        weightsOpt = minimize(aml_optimize.objective, weightsOpt, method='L-BFGS-B', jac=aml_optimize.gradient)

        k = 0
        while k < 4 and (len(self.mn.searchSpace) > 0):
            selectedVar, maxgrad = aml_optimize.getMaxGradient1(len(data), self.mn.searchSpace, candidateEdgeDataSum )
            if np.abs(maxgrad) > l1coeff:
                print('ACTIVATED EDGE')
                print(selectedVar)
                activeSet.append(selectedVar)
                numWeightsOpt += (len(self.mn.unaryPotentials[selectedVar[0]]) * len(self.mn.unaryPotentials[selectedVar[1]]))
                for t in range((len(self.mn.unaryPotentials[selectedVar[0]]) * len(self.mn.unaryPotentials[selectedVar[1]]))):
                    MapWeightsToVariables.append([selectedVar])
                self.mn.updateSearchSpace(selectedVar)
                self.mn.setEdgeFactor(selectedVar, np.zeros((len(aml_optimize.mn.unaryPotentials[selectedVar[0]]),
                                                         len(aml_optimize.mn.unaryPotentials[selectedVar[1]]))))
                aml_optimize = GraftingForPairwiseMRF(self.mn)
                aml_optimize.setRegularization(1, 0)
                candidateEdgeDataSum = []
                for t in range(self.mn.numWeightsSearchSpace):
                    candidateEdgeDataSum.append([0])
                for i in range(len(data)):
                    aml_optimize.addData(data[i])
                    candidateEdgeDataSum = self.getCandidateEdgeDataSum(data[i], self.mn, candidateEdgeDataSum)
                weightsOpt = np.random.randn(numWeightsOpt)
                weightsOpt = minimize(aml_optimize.objective, weightsOpt, method='L-BFGS-B', jac=aml_optimize.gradient)
            k += 1
            print('ACTIVE SPACE')
            print(activeSet)
        return aml_optimize.mn, weightsOpt.x, MapWeightsToVariables

def main():
    """Simple test function for maximum likelihood."""
    mn = MarkovNetForGrafting()
    grafting = GraftingForPairwiseMRF(mn)
    variables={0,1,2}
    numstates={0:2, 1:2, 2:2}
    data= ({0: 0, 1: 0, 2: 0},
           {0: 0, 1: 0, 2: 1},
           {0: 0, 1: 1, 2: 0},
           {0: 0, 1: 1, 2: 1},
           {0: 1, 1: 0, 2: 1},
           {0: 1, 1: 0, 2: 1},
           {0: 1, 1: 1, 2: 1},
           {0: 1, 1: 1, 2: 1},
           )

    MN, W, V = grafting.graft(variables, numstates, data, .1)
    print('LEARNED WEIGHTS')
    print(W)
    print('MAP')
    print(V)
if  __name__ =='__main__':
    main()
