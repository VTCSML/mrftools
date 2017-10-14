import numpy as np
from mrftools import *
import unittest
import time
from PartialMatrixBP import PartialMatrixBP


def create_grid_model(length):
        """Create a grid-structured MRF"""
        mn = MarkovNet()


        k = 8

        for x in range(length):
            for y in range(length):
                mn.set_unary_factor((x, y), np.random.random(k))

        for x in range(length - 1):
            for y in range(length):
                mn.set_edge_factor(((x, y), (x + 1, y)), np.random.random((k, k)))
                mn.set_edge_factor(((y, x), (y, x + 1)), np.random.random((k, k)))

        return mn
    
def test_MatrixBP(length):
    start = time.time()
    model = create_grid_model(length)
    mat_bp = MatrixBeliefPropagator(model)
    #mat_bp.update_messages()
    mat_bp.infer()
    end = time.time()
    print "running time: %f"%(end-start)
    #mat_bp = MatrixBeliefPropagator(model)
    #mat_bp.load_beliefs()
    #mat_bp.update_messages()
    #mat_bp.load_beliefs()


def test_PartialBP(N, length):
    start = time.time()
    model = create_grid_model(length)
    part_bp = PartialMatrixBP(model, N)
    part_bp.partial_infer()
    end = time.time()
    print "running time: %f"%(end-start)

if __name__ == '__main__':
    N = 3
    length = 256
    test_PartialBP(N, length)
    #test_MatrixBP(length)
    print "end"
    
    
    
