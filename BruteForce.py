"""BruteForce class."""
import numpy as np
from MarkovNet import MarkovNet
import itertools

class BruteForce(object):
    """Object that can do inference via ugly brute force. Recommended only for sanity checking and debugging using tiny examples."""

    def __init__(self, markovNet):
        """Initialize belief propagator for markovNet."""
        self.mn = markovNet
        self.varBeliefs = dict()
        self.pairBeliefs = dict()

    def computeZ(self):
        """Compute the partition function."""
        Z = 0.0

        variables = list(self.mn.variables)

        numStates = [self.mn.numStates[var] for var in variables]

        argList = [range(s) for s in numStates]

        for stateList in itertools.product(*argList):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = stateList[i]

            Z += np.exp(self.mn.evaluateState(states))

        return Z

    def unaryMarginal(self, var):
        """Compute the P(var) vector."""
        variables = list(self.mn.variables)

        numStates = [self.mn.numStates[v] for v in variables]

        p = np.zeros(self.mn.numStates[var])

        argList = [range(s) for s in numStates]

        for stateList in itertools.product(*argList):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = stateList[i]

            p[states[var]] += np.exp(self.mn.evaluateState(states))

        return p / np.sum(p)

    def pairwiseMarginal(self, varI, varJ):
        """Compute the P(var) vector."""
        variables = list(self.mn.variables)

        numStates = [self.mn.numStates[v] for v in variables]

        p = np.zeros((self.mn.numStates[varI], self.mn.numStates[varJ]))

        argList = [range(s) for s in numStates]

        for stateList in itertools.product(*argList):
            states = dict()
            for i in range(len(variables)):
                states[variables[i]] = stateList[i]

            p[states[varI], states[varJ]] += np.exp(self.mn.evaluateState(states))

        return p / np.sum(p)
