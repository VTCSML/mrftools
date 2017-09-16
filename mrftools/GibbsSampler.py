"""Gibbs sampling class"""
from __future__ import division

import random
from collections import Counter

import numpy as np
import pandas as pd
from scipy.misc import logsumexp


class GibbsSampler(object):
    """Object that can run Gibbs sampling on a MarkovNet"""

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

    def init_states(self, seed=None):
        """Initialize the state of each node."""

        if seed is not None:
            np.random.seed(seed)

        for var in self.mn.variables:
            weight = self.mn.unary_potentials[var]
            weight = np.exp(weight - logsumexp(weight))
            self.unary_weights[var] = weight
            self.states[var] = self.generate_state(self.unary_weights[var])

    def update_states(self):
        """Update the state of each node based on neighbor states."""
        for var in self.mn.variables:
            weight = self.mn.unary_potentials[var]
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
