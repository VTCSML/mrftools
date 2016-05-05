"""CountingNumberBeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet
from BeliefPropagator import BeliefPropagator
from BeliefPropagator import logsumexp
from random import shuffle

class TreeReweightedBeliefPropagator(BeliefPropagator):

    def __init__(self, markovNet, tree_probabilities = None):

        if tree_probabilities:
            self._set_tree_probabilities(tree_probabilities)

        super(TreeReweightedBeliefPropagator, self).__init__(markovNet)

    def _set_tree_probabilities(self, tree_probabilities):
        self.tree_probabilities = tree_probabilities

        for (edge, prob) in tree_probabilities.items():
            if edge[::-1] not in tree_probabilities:
                self.tree_probabilities[edge[::-1]] = prob

    def computeMessage(self, var, neighbor):
        """Compute the message from var to factor."""
        # compute the product of all messages coming into var except the one from neighbor

        pair = (var, neighbor)

        adjustedMessageProduct = self.varBeliefs[var] - self.messages[(neighbor, var)]

        # partial log-sum-exp operation
        matrix = self.mn.getPotential((neighbor, var)) / self.tree_probabilities[pair] + adjustedMessageProduct
        # the dot product with ones is slightly faster than calling sum
        message = np.log(np.exp(matrix - matrix.max()).dot(np.ones(matrix.shape[1])))

        # pseudo-normalize message
        message -= np.max(message)

        return message

    def computeBetheEntropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        entropy = 0.0

        unaryEntropy = dict()

        for var in self.mn.variables:
            unaryEntropy[var] = -np.sum(np.exp(self.varBeliefs[var]) * np.nan_to_num(self.varBeliefs[var]))
            entropy += unaryEntropy[var]
        for var in self.mn.variables:
            for neighbor in self.mn.neighbors[var]:
                if var < neighbor:
                    pairEntropy = -np.sum(np.exp(self.pairBeliefs[(var, neighbor)]) * np.nan_to_num(self.pairBeliefs[(var, neighbor)]))
                    mutual_information = unaryEntropy[var] + unaryEntropy[neighbor] - pairEntropy
                    entropy -= self.tree_probabilities[(var, neighbor)] * mutual_information
        return entropy


    def computeBeliefs(self):
        """Compute unary beliefs based on current messages."""
        for var in self.mn.variables:
            belief = self.mn.unaryPotentials[var]
            for neighbor in self.mn.getNeighbors(var):
                belief = belief + self.messages[(neighbor, var)] * self.tree_probabilities[(neighbor, var)]
            logZ = logsumexp(belief)
            belief = belief - logZ
            self.varBeliefs[var] = belief


    def computePairwiseBeliefs(self):
        """Compute pairwise beliefs based on current messages."""
        for var in self.mn.variables:
            for neighbor in self.mn.getNeighbors(var):
                if var < neighbor:
                    belief = self.mn.getPotential((var, neighbor)) / self.tree_probabilities[(var, neighbor)]

                    # compute product of all messages to var except from neighbor
                    varMessageProduct = self.varBeliefs[var] - self.messages[(neighbor, var)]
                    belief = (belief.T + varMessageProduct).T

                    # compute product of all messages to neighbor except from var
                    neighborMessageProduct = self.varBeliefs[neighbor] - self.messages[(var, neighbor)]
                    belief = belief + neighborMessageProduct

                    logZ = logsumexp(belief)
                    belief = belief - logZ
                    self.pairBeliefs[(var, neighbor)] = belief
                    self.pairBeliefs[(neighbor, var)] = belief.T

    def _random_tree(self):

        # TODO: implement random spanning tree algorithms (this current code is wrong)

        visited_nodes = set()

        all_edges = [x for x in self.mn.edgePotentials.iterkeys()]
        shuffle(all_edges)

        tree = []

        for (x, y) in all_edges:
            if x not in visited_nodes and y not in visited_nodes:
                tree.append((x, y))
                visited_nodes.add(x)
                visited_nodes.add(y)

        print tree

        return tree


    def sample_tree_probabilities(self, coverage):

        counts = dict()
        for edge in self.mn.edgePotentials:
            counts[edge] = 0

        total = 0

        while min(counts.itervalues()) < coverage:
            tree = self._random_tree()
            for edge in tree:
                counts[edge] += 1
            total += 1

        for edge in counts:
            counts[edge] /= float(total)

        self._set_tree_probabilities(counts)





def main():
    """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet()

    # np.random.seed(1)

    k = [4, 3, 6, 2]
    # k = [4, 4, 4, 4]

    mn.setUnaryFactor(0, np.random.randn(k[0]))
    mn.setUnaryFactor(1, np.random.randn(k[1]))
    mn.setUnaryFactor(2, np.random.randn(k[2]))
    mn.setUnaryFactor(3, np.random.randn(k[3]))

    mn.setEdgeFactor((0,1), np.random.randn(k[0], k[1]))
    mn.setEdgeFactor((1,2), np.random.randn(k[1], k[2]))
    mn.setEdgeFactor((3,2), np.random.randn(k[3], k[2]))
    mn.setEdgeFactor((3,0), np.random.randn(k[3], k[0])) # uncomment this to make loopy

    print("Neighbors of 0: " + repr(mn.getNeighbors(0)))
    print("Neighbors of 1: " + repr(mn.getNeighbors(1)))

    temperature = 1

    edgeProbabilities = dict()

    for edge in mn.edgePotentials:
        edgeProbabilities[edge] = 0.75 # TRBP
        # edgeProbabilities[edge] = 1 # BP

    trbp = TreeReweightedBeliefPropagator(mn, edgeProbabilities)
    bp = BeliefPropagator(mn)

    # for t in range(15):
    #     change = bp.updateMessages()
    #     disagreement = bp.computeInconsistency()
    #     print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))

    trbp.runInference(display='full')

    bp.runInference()

    trbp.computePairwiseBeliefs()

    from BruteForce import BruteForce

    bf = BruteForce(mn)

    for i in range(2):
        print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unaryMarginal(i))))
        print ("TRBP unary marginal of %d:        %s" % (i, repr(np.exp(trbp.varBeliefs[i]))))

    print ("Brute force pairwise marginal: " + repr(bf.pairwiseMarginal(0,1)))
    print ("TRBP pairwise marginal:        " + repr(np.exp(trbp.pairBeliefs[(0,1)])))

    print ("Tree Bethe energy functional:       %f" % trbp.computeEnergyFunctional())
    print ("Bethe energy functional:            %f" % bp.computeEnergyFunctional())
    print ("Brute force log partition function: %f" % np.log(bf.computeZ()))


    print("Setting new tree probabilities using sampling method")
    trbp.sample_tree_probabilities(1)
    trbp.runInference(display = 'None')
    print ("Tree Bethe energy functional:       %f" % trbp.computeEnergyFunctional())
    print trbp.tree_probabilities

if  __name__ =='__main__':
    main()
