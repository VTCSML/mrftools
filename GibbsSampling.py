import numpy as np
from util import logsumexp
from MarkovNet import MarkovNet
import pandas as pd
import random
from collections import Counter

class Gibbs(object):
    "Object that can run gibbs sampling on a MarkovNet"

    def __init__(self, markov_net):
        """Initialize belief propagator for markov_net."""
        self.mn = markov_net
        self.States = dict()
        self.Unaryweights = dict()
        self.Samples = list()

    def weighted_choice(self, weight):
        """Generate state according to the given weight"""
        rnd = random.random() * sum(weight)
        for i, w in enumerate(weight):
            rnd = rnd - w
            if rnd < 0:
                return i

    def init_states(self):
        """Initialize the state of each node."""
        for var in self.mn.variables:
            weight = self.mn.getUnaryPotential(var)
            log_z = logsumexp(weight)
            weight = weight - log_z
            weight = np.exp(weight)
            self.Unaryweights[var] = weight;
            self.States[var] = self.weighted_choice(self.Unaryweights[var])

    def update_states(self):
        """Update the state of each node based on neighbor states."""
        for var in self.mn.variables:
            weight = self.mn.getUnaryPotential(var)
            for neighbor in self.mn.neighbors[var]:
                weight =  weight + self.mn.get_potential((var, neighbor))[:, self.get_states(neighbor)]
                log_z = logsumexp(weight)
                weight = weight - log_z
                weight = np.exp(weight)
            self.States[var] = self.weighted_choice(weight)

    def get_states(self, variable):
        return self.States[variable]

    def mix(self, ite):
        """Run the state Update procedure until mix, ite: number of iterations for mixing"""
        for i in range(0, ite):
                self.update_states()

    def sampling(self, num, s):
        """Run the sampling: num, number of samples; s, gap between two samples (So when s = 1, means take consecutive samples)"""
        for i in range(0, num):
            self.update_states()
            self.Samples.append(self.States.copy())
        for i in range(0, s-1):
            self.update_states()

    def counter(self, samples):
        for var in self.mn.variables:
            counts = Counter(pd.DataFrame(samples)[var])
            print(counts)

    def gibbs_sampling(self, itr, num, s):
        self.mix(itr)
        self.sampling(num, s)
        self.counter(self.Samples)


def main():
    # """Test basic functionality of BeliefPropagator."""
    mn = MarkovNet()

    np.random.seed(1)

    mn.set_unary_factor(0, np.random.randn(4))
    mn.set_unary_factor(1, np.random.randn(3))
    mn.set_unary_factor(2, np.random.randn(6))
    mn.set_unary_factor(3, np.random.randn(2))

    mn.set_edge_factor((0, 1), np.random.randn(4, 3))
    mn.set_edge_factor((1, 2), np.random.randn(3, 6))
    mn.set_edge_factor((3, 2), np.random.randn(2, 6))
    mn.set_edge_factor((3, 0), np.random.randn(2, 4)) # uncomment this to make loopy

    gb = Gibbs(mn)
    gb.init_states()
    print(gb.States)

    itr = 10000
    num = 1000
    s = 1
    gb.gibbs_sampling(itr, num, s)


if  __name__ =='__main__':
    main()
