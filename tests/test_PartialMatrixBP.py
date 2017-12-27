import numpy as np
from mrftools import *
import unittest
import time
from mrftools.PartialMatrixBP import PartialMatrixBP
from mrftools.FastPartialBP import FastPartialMatrixBP
import create_tree

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
    
def create_chain_model(length):
        """Create a chain-structured Markov net with random potentials and different variable cardinalities."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))

        factor4 = np.random.randn(k[4])
        factor4[2] = -float('inf')

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((2, 3), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((3, 4), np.random.randn(k[3], k[4]))
        mn.create_matrices()

        return mn
    
def create_tree_model(num_nodes):
    np.random.seed(1)
    nodes_ids = list()
    for i in range(0, num_nodes):
        nodes_ids.append(i)
    edges = create_tree.create_tree(nodes_ids)
    mn = MarkovNet()
    k = 8
    for node_id in nodes_ids:
        mn.set_unary_factor(node_id, np.random.random(k))
    for edge in edges:
        mn.set_edge_factor(edge, np.random.random((k, k)))
    return mn

def beliefs_tree_model_test(N, num_nodes):
    mn = create_tree_model(num_nodes)
    
    matrix_bp = MatrixBeliefPropagator(mn)
    matrix_bp.infer()
    matrix_bp.load_beliefs()
    #bf = BruteForce(mn)
    #print bf.unary_marginal(1)
    #print np.exp(matrix_bp.var_beliefs[1])
    
    partial_bp = PartialMatrixBP(mn)
    for i in range(0, 50):
        partial_bp.partial_infer(N, tolerance=1e-8)
    partial_bp.load_beliefs()
    
    
    for i in mn.variables:
        print "Brute force unary marginal of %d: %s" % (i, repr(np.exp(partial_bp.var_beliefs[i])))
        print "Belief prop unary marginal of %d: %s" % (i, repr(np.exp(matrix_bp.var_beliefs[i])))
        assert np.allclose(np.exp(partial_bp.var_beliefs[i]), np.exp(matrix_bp.var_beliefs[i])), "beliefs aren't exact on tree model"
    
def MatrixBP_runtime_test(length):
    start = time.time()
    model = create_grid_model(length)
    mat_bp = MatrixBeliefPropagator(model)
    #mat_bp.update_messages()
    mat_bp.infer()
    end = time.time()
    print "running time: %f"%(end-start)

def PartialBP_runtime_test(N, length):
    start = time.time()
    model = create_grid_model(length)
    part_bp = PartialMatrixBP(model)
    part_bp.partial_infer(N)
    end = time.time()
    print "running time: %f"%(end-start)

def FastPartialBP_runtime_test(N, length):
    start = time.time()
    model = create_grid_model(length)
    part_bp = FastPartialMatrixBP(model)
    part_bp.partial_infer(N)
    end = time.time()
    print "running time: %f"%(end-start)

if __name__ == '__main__':
    N = 100
    length = 300
    num_nodes = 500
    #test_beliefs_tree_model(N, num_nodes)
    FastPartialBP_runtime_test(N, length)
    PartialBP_runtime_test(N, length)
    #MatrixBP_runtime_test(length)
    print "end"
    
    
    
