import numpy as np
import hashlib

class MatrixCache(object):
    """Utility class that identifies identical matrices and allows them to be referenced to the same objects in memory. Note that changing any matrix after it has been cached will make this class behave unexpectedly."""

    def __init__(self):
        self.matrices = dict()


    def get_cached(self, matrix):
        """Check if we have already seen matrix before. If so, return the original copy of matrix and discard the new one.
        :type matrix: ndarray
        """
        # key = HashableMatrix(matrix)
        key = tuple([x for x in matrix.flat]) + matrix.shape
        if key in self.matrices:
            # get the cached version identical to the input matrix
            return self.matrices[key]
        else:
            # if the matrix has not been seen yet, add it to the cache
            self.matrices[key] = matrix
            return matrix
