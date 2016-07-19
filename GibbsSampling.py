from __future__ import division
import numpy as np
from scipy.misc import logsumexp
from MarkovNet import MarkovNet
import pandas as pd
import random
from collections import Counter
import collections

class Gibbs(object):
    "Object that can run gibbs sampling on a MarkovNet"

    def __init__(self, markov_net):
        """Initialize belief propagator for markov_net."""
        self.mn = markov_net
        self.states = dict()
        self.unary_weights = dict()
        self.samples = list()

    def generate_state(self, weight):
        """Generate state according to the given weight"""
        r = random.uniform(0, 1)
        # Sum = sum(weight.values())
        Sum = sum(weight)
        rnd = r * Sum
        for i in range(len(weight)):
            rnd = rnd - weight[i]
            if rnd < 0:
                return i

    def init_states(self):
        """Initialize the state of each node."""
        for var in self.mn.variables:
            weight = self.mn.unaryPotentials[var]
            weight = np.exp(weight - logsumexp(weight))
            self.unary_weights[var] = weight
            self.states[var] = self.generate_state(self.unary_weights[var])

    def update_states(self):
        """Update the state of each node based on neighbor states."""
        for var in self.mn.variables:
            weight = self.mn.unaryPotentials[var]
            for neighbor in self.mn.neighbors[var]:
                weight = weight + self.mn.get_potential((var, neighbor))[:, self.states[neighbor]]
            weight = np.exp(weight - logsumexp(weight))
            self.states[var] = self.generate_state(weight)


    def mix(self, ite):
        """Run the state Update procedure until mix, ite: number of iterations for mixing"""
        for i in range(0, ite):
                self.update_states()

    def sampling(self, num):
        """Run the sampling: num, number of samples; s, gap between two samples (So when s = 1, means take consecutive samples)"""
        for i in range(0, num):
            self.update_states()
            self.samples.append(self.states.copy())
        # for i in range(0, s-1):
        #     self.update_states()

    def gibbs_sampling(self, itr, num):
        self.mix(itr)
        self.sampling(num)


    def counter(self, var):
        counts = Counter(pd.DataFrame(self.samples)[var])
        count_array = np.asarray(list(counts.values()))
        return count_array


def main():
    # # """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet()

    np.random.seed(1)

    unary_potential = np.random.randn(2)
    edge_potential = np.random.randn(2, 2)

    print unary_potential
    print edge_potential


    mn.set_unary_factor(0, unary_potential)
    mn.set_unary_factor(1, unary_potential)
    mn.set_unary_factor(2, unary_potential)
    mn.set_unary_factor(3, unary_potential)

    mn.set_edge_factor((0, 1), edge_potential)
    mn.set_edge_factor((1, 2), edge_potential)
    mn.set_edge_factor((2, 3), edge_potential)
    # mn.set_edge_factor((3, 0), edge_potential) # uncomment this to make loopy

    gb = Gibbs(mn)
    gb.init_states()
    print "result:"
    print(gb.states)

    itr = 1000
    num = 100000
    gb.gibbs_sampling(itr, num)

    from BruteForce import BruteForce

    bf = BruteForce(mn)
    for var in mn.variables:
        gb_result = gb.counter(var) / num
        bf_result = bf.unary_marginal(var)
        print gb_result
        print bf_result
        np.testing.assert_allclose(gb_result, bf_result, rtol=2e-2, atol=0)




if  __name__ =='__main__':
    main()
