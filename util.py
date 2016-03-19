"""Utility functions."""
import numpy as np

def logsumexp(x, dimension = None):
    """Compute log(sum(exp(x), dimension)) in a numerically stable manner."""
    maxVals = np.max(x, dimension)

    return np.log(np.sum(np.exp(x - maxVals), dimension)) + maxVals


def main():
    """Test basic functionality of util functions."""
    x = np.random.randn(6,4)

    print np.log(np.sum(np.exp(x), 0))
    print logsumexp(x, 0)

if  __name__ =='__main__':
    main()
