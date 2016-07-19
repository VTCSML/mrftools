from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from BeliefPropagator import BeliefPropagator
from matinference import MatrixInference, Model
import numpy as np

import matplotlib.pyplot as plt


def main():
    mn = LogLinearModel()

    length = 2

    k = 3
    d = 3

    model = Model(k, d)

    for x in range(length):
        for y in range(length):
            mn.set_unary_factor((x, y), np.zeros(k))
            features = np.random.randn(d, 1)
            mn.set_unary_features((x, y), features)
            model.add_var((x, y), features)

    edge_weights = np.random.randn(k, k)

    for x in range(length - 1):
        for y in range(length):
            mn.set_edge_factor(((x, y), (x + 1, y)), edge_weights)
            mn.set_edge_factor(((y, x), (y, x + 1)), edge_weights)
            model.add_edge(((x, y), (x + 1, y)))
            model.add_edge(((y, x), (y, x + 1)))

    unary_weight_mat = np.random.randn(d, k)
    mn.create_matrices()
    mn.set_weight_matrix(unary_weight_mat)
    mn.set_unary_matrix()

    bp = MatrixBeliefPropagator(mn)

    model.set_weights(np.append(unary_weight_mat.ravel(), edge_weights.ravel()))
    model.create_matrices()

    mat_inference = MatrixInference(model)
    mat_inference.create_matrices()
    mat_inference.set_counting_nums()

    import time

    t0 = time.time()
    bp.runInference(display='full', maxIter=30000)
    t1 = time.time()

    bp_time = t1 - t0

    t0 = time.time()
    mat_inference.infer(damping=0.001)
    t1 = time.time()

    convex_time = t1 - t0

    print("Matrix BP took %f, convex BP took %f." % (bp_time, convex_time))

    plt.subplot(121)
    plt.imshow(bp.belief_mat[0, :].reshape((length, length)), interpolation='nearest')
    plt.xlabel('Matrix BP')


    plt.subplot(122)
    plt.imshow(mat_inference.get_unary_mat()[0,:].reshape((length, length)), interpolation='nearest')
    plt.xlabel('Convex BP')
    plt.show()

if  __name__ =='__main__':
    main()
