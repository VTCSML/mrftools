"""CountingNumberBeliefPropagator class."""
import numpy as np
from MarkovNet import MarkovNet
from BeliefPropagator import BeliefPropagator
from BeliefPropagator import logsumexp
from random import shuffle
from Inference import Inference

class TreeReweightedBeliefPropagator(BeliefPropagator):

    def __init__(self, markov_net, tree_probabilities=None):

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
            belief = self.mn.unary_potentials[var]
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

        all_edges = [x for x in self.mn.edgePotentials.iterkeys()]
        shuffle(all_edges)

        group = dict()
        for var in self.mn.variables:
            group[var] = set([var])

        tree = []

        for (x, y) in all_edges:
            # if adding (x, y) doesn't create a cycle
            if group[x] is not group[y]:
                tree.append((x, y))

                for node in group[x]:
                    group[y].add(node) # add nodes connected to x to y's group
                    group[node] = group[y]

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
