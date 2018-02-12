import numpy as np


def save_weights(weights, path):
    np.savetxt(path, weights)




def load_weights(path):
    weights = np.loadtxt(file(path))
    return weights
