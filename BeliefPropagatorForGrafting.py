"""BeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet
from MarkovNetForGrafting import MarkovNetForGrafting
from scipy.misc import logsumexp
from BeliefPropagator import BeliefPropagator

class BeliefPropagatorForGrafting(BeliefPropagator):
    """Object that can run belief propagation on a MarkovNet."""

    def __init__(self, MNGrafting):
        """Initialize belief propagator for markovNet."""
        BeliefPropagator.__init__(self, MNGrafting)
        self.candidatePairBeliefs = dict()
        self.candidatePairBeliefsVec = []
        self.mapCandidatePairBeliefsVec = []


    def computeCandidateEdgeBelief(self, currSearchSpace):
        marginals = []
        mapVec =[]
        print('currSearchSpace')
        print(currSearchSpace)
        for edge in currSearchSpace:
            size = len(self.mn.unaryPotentials[edge[0]]) * len(self.mn.unaryPotentials[edge[1]])
            #belief = np.log(np.outer(np.exp(self.varBeliefs[edge[0]] - self.mn.unaryPotentials[edge[0]]), np.exp(self.varBeliefs[edge[1]] - self.mn.unaryPotentials[edge[1]]).T))
            belief = self.varBeliefs[edge[0]] - self.mn.unaryPotentials[edge[0]] + np.matrix(self.varBeliefs[edge[1]] - self.mn.unaryPotentials[edge[1]]).T
            #print(((self.varBeliefs[edge[1]] - self.mn.unaryPotentials[edge[1]]).T).shape)
            marginals.extend(belief.reshape((-1, 1)).tolist())
            for t in range(size):
                mapVec.append(edge)
        return marginals, mapVec

def main():
    """Test basic functionality of BeliefPropagator."""
    mn = MarkovNetForGrafting()
    np.random.seed(1)
    mn.setUnaryFactor(0, np.random.randn(4))
    mn.setUnaryFactor(1, np.random.randn(3))
    mn.setUnaryFactor(2, np.random.randn(6))
    mn.setUnaryFactor(3, np.random.randn(2))
    mn.initCandidateEdges()
    # mn.setEdgeFactor((3,0), np.random.randn(2,4)) # uncomment this to make loopy
    bp = BeliefPropagatorForGrafting(mn)
    # for t in range(15):
    #     change = bp.updateMessages()
    #     disagreement = bp.computeInconsistency()
    #     print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))
    bp.runInference(display='full')
    bp.computePairwiseBeliefs()
    #print(bp.CandidatePairBeliefs)

'''
    from BruteForce import BruteForce
    bf = BruteForce(mn)
    for i in range(2):
        print "Brute force unary marginal of %d: %s" % (i, repr(bf.unaryMarginal(i)))
        print "Belief prop unary marginal of %d: %s" % (i, repr(bf.unaryMarginal(i)))
    print "Brute force pairwise marginal: " + repr(bf.pairwiseMarginal(0,1))
    print "Belief prop pairwise marginal: " + repr(np.exp(bp.pairBeliefs[(0,1)]))
    print "Bethe energy functional: %f" % bp.computeEnergyFunctional()
'''

if  __name__ =='__main__':
    main()
