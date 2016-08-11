from scipy.sparse import csc_matrix
from scipy.optimize import check_grad
import unittest

try:
    import autograd.numpy as np
    from autograd import grad
    from autograd.core import primitive
    from autograd.core import getval
    from autograd.numpy.numpy_grads import make_grad_dot


    x = np.array([[0.50578627, 0.37340201, 0.52078787],
                  [0.96267211, 0.5459987, 0.18807968]])

    #y = np.array([[0.81637542], [1.12259795]])

    row = np.array([0, 2, 2, 0, 1, 2])
    col = np.array([0, 0, 1, 2, 2, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    y = csc_matrix((data, (row, col)))

    print x.shape
    print y.shape

    @primitive
    def sparse_dot(full_matrix, sparse_matrix):
        return sparse_matrix.T.dot(full_matrix.T).T

    print y.todense()

    sparse_dot.defgrad(lambda ans, full_matrix, sparse_matrix : make_grad_dot(0, ans, full_matrix, sparse_matrix.todense()))

    objective = lambda x: np.sum(sparse_dot(x.reshape((2, 3)), y))

    gradient = grad(objective)

    print gradient(x.ravel())

    print check_grad(objective, gradient, x.ravel())

except ImportError:
    pass