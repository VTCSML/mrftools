from __future__ import division
import numpy as np
from scipy.misc import logsumexp
from MarkovNet import MarkovNet
import pandas as pd
import random
from collections import Counter
import collections
import copy

class GibbsSamplerMod(object):
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
        Sum = sum(weight) # ALWAYS EQUAL TO 1 BUG????
        # print(Sum) 
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
            self.unary_weights[var] = weight # WHY????
            self.states[var] = self.generate_state(weight)

    def update_states(self):
        """Update the state of each node based on neighbor states."""
        for var in self.mn.variables:
            weight = copy.deepcopy(self.mn.unary_potentials[var])
            for neighbor in self.mn.neighbors[var]:
                weight = weight + copy.deepcopy(self.mn.get_potential((var, neighbor))[:, self.states[neighbor]])
            weight = np.exp(weight - logsumexp(weight))
            self.states[var] = self.generate_state(weight)


    def mix(self, ite):
        """Run the state Update procedure until mix, ite: number of iterations for mixing"""
        for i in range(0, ite):
                # print('update')
                self.update_states()

    def sampling(self, num, itr):
        """Run the sampling: num, number of samples; s, gap between two samples (So when s = 1, means take consecutive samples)"""
        self.mix(1000)
        num = num * 10
        i = 0
        for j in range(0, num):
            i += 1
            # self.init_states()
            self.update_states()
            # self.mix(itr)
            if i == 10:
                self.samples.append(self.states.copy())
                i = 0
        # for i in range(0, s-1):
        #     self.update_states()

    def gibbs_sampling(self, itr, num):
        # self.mix(itr)
        self.sampling(num, itr)


    def counter(self, var):
        counts = Counter(pd.DataFrame(self.samples)[var])
        count_array = np.asarray(list(counts.values()))
        return count_array
