"""CountingNumberBeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet
from BeliefPropagator import BeliefPropagator
from BeliefPropagator import logsumexp
from random import shuffle
from Inference import Inference

class TreeReweightedBeliefPropagator(Inference):

    def __init__(self, markov_net, tree_probabilities = None):

        if tree_probabilities:
            self._set_tree_probabilities(tree_probabilities)

        super(TreeReweightedBeliefPropagator, self).__init__(markov_net)

    def _set_tree_probabilities(self, tree_probabilities):
        self.tree_probabilities = tree_probabilities

        for (edge, prob) in tree_probabilities.items():
            if edge[::-1] not in tree_probabilities:
                self.tree_probabilities[edge[::-1]] = prob

    def compute_message(self, var, neighbor):
        """Compute the message from var to factor."""
        # compute the product of all messages coming into var except the one from neighbor

        pair = (var, neighbor)

        adjusted_message_product = self.var_beliefs[var] - self.messages[(neighbor, var)]

        # partial log-sum-exp operation
        matrix = self.mn.get_potential((neighbor, var)) / self.tree_probabilities[pair] + adjusted_message_product
        # the dot product with ones is slightly faster than calling sum
        message = np.log(np.exp(matrix - matrix.max()).dot(np.ones(matrix.shape[1])))

        # pseudo-normalize message
        message -= np.max(message)

        return message

    def compute_bethe_entropy(self):
        """Compute Bethe entropy from current beliefs. Assume that the beliefs have been computed and are fresh."""
        entropy = 0.0

        unary_entropy = dict()

        for var in self.mn.variables:
            unary_entropy[var] = -np.sum(np.exp(self.var_beliefs[var]) * np.nan_to_num(self.var_beliefs[var]))
            entropy += unary_entropy[var]
        for var in self.mn.variables:
            for neighbor in self.mn.neighbors[var]:
                if var < neighbor:
                    pair_entropy = -np.sum(np.exp(self.pair_beliefs[(var, neighbor)]) * np.nan_to_num(self.pair_beliefs[(var, neighbor)]))
                    mutual_information = unary_entropy[var] + unary_entropy[neighbor] - pair_entropy
                    entropy -= self.tree_probabilities[(var, neighbor)] * mutual_information
        return entropy


    def compute_beliefs(self):
        """Compute unary beliefs based on current messages."""
        for var in self.mn.variables:
            belief = self.mn.unaryPotentials[var]
            for neighbor in self.mn.get_neighbors(var):
                belief = belief + self.messages[(neighbor, var)] * self.tree_probabilities[(neighbor, var)]
            log_z = logsumexp(belief)
            belief = belief - log_z
            self.var_beliefs[var] = belief


    def compute_pairwise_beliefs(self):
        """Compute pairwise beliefs based on current messages."""
        for var in self.mn.variables:
            for neighbor in self.mn.get_neighbors(var):
                if var < neighbor:
                    belief = self.mn.get_potential((var, neighbor)) / self.tree_probabilities[(var, neighbor)]

                    # compute product of all messages to var except from neighbor
                    var_message_product = self.var_beliefs[var] - self.messages[(neighbor, var)]
                    belief = (belief.T + var_message_product).T

                    # compute product of all messages to neighbor except from var
                    neighbor_message_product = self.var_beliefs[neighbor] - self.messages[(var, neighbor)]
                    belief = belief + neighbor_message_product

                    log_z = logsumexp(belief)
                    belief = belief - log_z
                    self.pair_beliefs[(var, neighbor)] = belief
                    self.pair_beliefs[(neighbor, var)] = belief.T

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

    mn.set_unary_factor(0, np.random.randn(k[0]))
    mn.set_unary_factor(1, np.random.randn(k[1]))
    mn.set_unary_factor(2, np.random.randn(k[2]))
    mn.set_unary_factor(3, np.random.randn(k[3]))

    mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
    mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
    mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
    mn.set_edge_factor((3, 0), np.random.randn(k[3], k[0])) # uncomment this to make loopy

    print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
    print("Neighbors of 1: " + repr(mn.get_neighbors(1)))


    edge_probabilities = dict()

    for edge in mn.edgePotentials:
        edge_probabilities[edge] = 0.75 # TRBP
        # edge_probabilities[edge] = 1 # BP

    trbp = TreeReweightedBeliefPropagator(mn, edge_probabilities)
    bp = BeliefPropagator(mn)

    # for t in range(15):
    #     change = bp.update_messages()
    #     disagreement = bp.compute_inconsistency()
    #     print("Iteration %d, change in messages %f. Calibration disagreement: %f" % (t, change, disagreement))

    trbp.runInference(display='full')

    bp.runInference()

    trbp.compute_pairwise_beliefs()

    from BruteForce import BruteForce

    bf = BruteForce(mn)

    for i in range(2):
        print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unary_marginal(i))))
        print ("TRBP unary marginal of %d:        %s" % (i, repr(np.exp(trbp.var_beliefs[i]))))

    print ("Brute force pairwise marginal: " + repr(bf.pairwise_marginal(0, 1)))
    print ("TRBP pairwise marginal:        " + repr(np.exp(trbp.pair_beliefs[(0,1)])))

    print ("Tree Bethe energy functional:       %f" % trbp.computeEnergyFunctional())
    print ("Bethe energy functional:            %f" % bp.computeEnergyFunctional())
    print ("Brute force log partition function: %f" % np.log(bf.compute_z()))

    # Start testing upper bound property

    trials = 200

    tr_diff = np.zeros(trials)
    bp_diff = np.zeros(trials)

    for trial in range(trials):

        mn = MarkovNet()

        width = 3
        height = 4

        k = 3

        for x in range(width):
            for y in range(height):
                mn.set_unary_factor((x, y), np.random.random(k))

        for x in range(width - 1):
            for y in range(height - 1):
                mn.set_edge_factor(((x, y), (x + 1, y)), np.random.random((k, k)))
                mn.set_edge_factor(((y, x), (y, x + 1)), np.random.random((k, k)))

        bf = BruteForce(mn)
        bp = BeliefPropagator(mn)

        edge_probabilities = dict()

        for edge in mn.edgePotentials:
            edge_probabilities[edge] = 0.5

        interior_prob = 0.5
        border_prob = 0.75

        # formula for square grids
        # interior_prob = (pow(width, height - 1) + pow(height, width)) / (2 * pow(width, height))
        # border_prob = interior_prob

        for x in range(width):
            edge_probabilities[(x, 0)] = interior_prob
            edge_probabilities[(x, height-1)] = interior_prob

        for y in range(height):
            edge_probabilities[(0, y)] = border_prob
            edge_probabilities[(width-1, y)] = border_prob

        trbp = TreeReweightedBeliefPropagator(mn, edge_probabilities)

        bp.runInference(display = 'off')
        trbp.runInference(display = 'off')

        trbp_z = trbp.computeEnergyFunctional()
        true_z = np.log(bf.compute_z())
        bethe_z = bp.computeEnergyFunctional()

        print ("Tree Bethe energy functional:       %f" % trbp_z)
        print ("Bethe energy functional:            %f" % bethe_z)
        print ("Brute force log partition function: %f" % true_z)

        print ("Is the TRBP energy functional an upper bound? %s" %
               trbp_z >= true_z)

        bp_diff[trial] = bethe_z - true_z
        tr_diff[trial] = trbp_z - true_z

        print("Difference range between variational Z and truth:")
        print("TRBP:  %f to %f" % (min(tr_diff[:trial+1]), max(tr_diff[:trial+1])))
        print("Bethe: %f to %f" % (min(bp_diff[:trial+1]), max(bp_diff[:trial+1])))
        print("Average error. TRBP: %f, Bethe: %f" %
              (np.mean(np.abs(tr_diff[:trial+1])), np.mean(np.abs(bp_diff[:trial+1]))))

if  __name__ =='__main__':
    main()
