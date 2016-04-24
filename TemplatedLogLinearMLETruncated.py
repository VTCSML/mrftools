import autograd.numpy as np
from TemplatedLogLinearMLE import TemplatedLogLinearMLE

class TemplatedLogLinearMLETruncated(TemplatedLogLinearMLE):
    
    def __init__(self, baseModel, bpIters = 3):
        """bpIters sets how many inference iterations are allowed per call to inference"""
        super(TemplatedLogLinearMLETruncated, self).__init__(baseModel)
        self.bpIters = bpIters


    def getFeatureExpectations(self):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        :rtype: numpy.ndarray
        """
        marginalSum = 0
        for i in range(len(self.labels)):
            bp = self.beliefPropagators[i]
            model = self.models[i]
            if self.needInference:
                bp.initMessages()
                bp.runInference(display = 'final', maxIter = self.bpIters)
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()

            # make vector form of marginals
            marginals = []
            for i in range(len(self.potentials)):
                if isinstance(self.potentials[i], tuple):
                    # get pairwise belief
                    table = np.exp(bp.pairBeliefs[self.potentials[i]])
                else:
                    # get unary belief and multiply by features
                    var = self.potentials[i]
                    table = np.outer(np.exp(bp.varBeliefs[var]), model.unaryFeatures[var])

                # flatten table and append
                marginals.extend(list(table.reshape((-1, 1))))
            marginalSum += np.array(marginals)

        return marginalSum / len(self.labels)