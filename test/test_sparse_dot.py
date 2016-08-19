import unittest
from mrftools import *
import numpy as np
from scipy.optimize import check_grad
import time
from scipy.sparse import csc_matrix
try:
    from autograd import grad, jacobian
    from mrftools.MatrixBeliefPropagator import sparse_dot

    class TestSparseDot(unittest.TestCase):

        def test_jacobian(self):
            sparse_fun = lambda x, y: sparse_dot(x, y)
            dense_fun = lambda x, y: np.dot(x, y)
            sparse_grad = jacobian(sparse_fun)
            dense_grad = jacobian(dense_fun)

            x = np.random.randn(4, 3)
            y = np.random.randn(3, 2)

            y_sparse = csc_matrix(y)

            g_s = sparse_grad(x, y_sparse)
            g_d = dense_grad(x, y)

            print(g_s - g_d)
            assert np.sum(np.abs(g_s - g_d)) < 1e-6, "Jacobians do not agree"


except ImportError:
    print "Autograd could not be imported. Skipping tests for AutogradLearner"
