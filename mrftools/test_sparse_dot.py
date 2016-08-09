import autograd.numpy as np
from autograd import grad
from autograd.core import primitive
from autograd.core import getval
from scipy.sparse import csc_matrix

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

sparse_dot.defgrad(lambda ans, full_matrix, sparse_matrix : unbroadcast(ans, full_matrix, lambda g :   g * sparse_matrix))
#sparse_dot.defgrad(lambda ans, full_matrix, sparse_matrix : unbroadcast(ans, sparse_matrix, lambda g : - g * full_matrix * sparse_matrix), argnum=1)

isarray = lambda x : isinstance(getval(x), np.ndarray)

def unbroadcast(ans, x, gradfun):
    # x is the argument that we're differentiating with respect to.
    if isarray(x):
        shape = x.shape
        def new_fun(g):
            result = gradfun(g)
            while np.ndim(result) > len(shape):
                result = np.sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size == 1:
                    result = np.sum(result, axis=axis, keepdims=True)
            assert np.shape(result) == shape
            return result
    elif isarray(ans):
        new_fun = lambda g : np.sum(gradfun(g))
    else:
        return gradfun
    new_fun.__name__ = "unbroadcast_{0}".format(gradfun.__name__)
    return new_fun

print grad(lambda x, y: np.sum(sparse_dot(x, y)))(x, y)
