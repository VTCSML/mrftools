"""BruteForce class."""
import numpy as np
from MarkovNet import MarkovNet
import itertools
import operator


class BruteForce(object):
    """Object that can do inference via ugly brute force. Recommended only for sanity checking and debugging using tiny examples."""

    def __init__(self, markov_net):
        """Initialize belief propagator for markov_net."""
        self.mn = markov_net
        self.varBeliefs = dict()
        self.pairBeliefs = dict()

    def compute_z(self):
        """Compute the partition function."""
        z = 0.0

        variables = list(self.mn.variables)

        num_states = [self.mn.num_states[var] for var in variables]

        arg_list = [range(s) for s in num_states]

        for state_list in itertools.product(*arg_list):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = state_list[i]

            z += np.exp(self.mn.evaluate_state(states))

        return z

    def entropy(self):
        z = self.compute_z()

        log_z = np.log(z)

        h = 0.0

        variables = list(self.mn.variables)

        num_states = [self.mn.num_states[var] for var in variables]

        arg_list = [range(s) for s in num_states]

        for state_list in itertools.product(*arg_list):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = state_list[i]

            log_p = self.mn.evaluate_state(states)

            h -= (log_p - log_z) * np.exp(log_p - log_z)

        return h

    def unary_marginal(self, var):
        """Compute the P(var) vector."""
        variables = list(self.mn.variables)

        num_states = [self.mn.num_states[v] for v in variables]

        p = np.zeros(self.mn.num_states[var])

        arg_list = [range(s) for s in num_states]
        for state_list in itertools.product(*arg_list):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = state_list[i]

            p[states[var]] += np.exp(self.mn.evaluate_state(states))

        return p / np.sum(p)

    def pairwise_marginal(self, var_i, var_j):
        """Compute the P(var) vector."""
        variables = list(self.mn.variables)

        num_states = [self.mn.num_states[v] for v in variables]

        p = np.zeros((self.mn.num_states[var_i], self.mn.num_states[var_j]))

        arg_list = [range(s) for s in num_states]

        for state_list in itertools.product(*arg_list):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = state_list[i]

            p[states[var_i], states[var_j]] += np.exp(self.mn.evaluate_state(states))

        return p / np.sum(p)


    def map_inference(self):
        """Compute the best states configurations for MAP inference."""
        variables = list(self.mn.variables)

        num_states = [self.mn.num_states[v] for v in variables]

        arg_list = [range(s) for s in num_states]

        map_dic = {}

        for state_list in itertools.product(*arg_list):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = state_list[i]

            map_dic[state_list] = np.exp(self.mn.evaluate_state(states))

        map_results = max ( map_dic.iteritems ( ), key=operator.itemgetter ( 1 ) )
        map_values = map_results[1]
        map_states = map_results[0]
        belief_mat = -np.inf * np.ones (( max(num_states) , len(variables) ))

        for i in range(0,len(map_states)):
            belief_mat[map_states[i],i] = 0

        # print belief_mat
        return belief_mat



