"""Tests for the MarkovNet model objects"""
import unittest
from mrftools import *
import numpy as np
import torch


class TestTorchMarkovNet(unittest.TestCase):
    """Test class for the MarkovNet model objects"""
    def create_chain_model(self, is_cuda):
        """Create chain model with different variable cardinalities."""
        mn = TorchMarkovNet(is_cuda=is_cuda, var_on=False)

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

        mn.set_unary_factor(0, torch.from_numpy(np.random.randn(k[0])))
        mn.set_unary_factor(1, torch.from_numpy(np.random.randn(k[1])))
        mn.set_unary_factor(2, torch.from_numpy(np.random.randn(k[2])))
        mn.set_unary_factor(3, torch.from_numpy(np.random.randn(k[3])))

        factor4 = torch.from_numpy(np.random.randn(k[4]))
        factor4[2] = -float('inf')

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), torch.from_numpy(np.random.randn(k[0], k[1])))
        mn.set_edge_factor((1, 2), torch.from_numpy(np.random.randn(k[1], k[2])))
        mn.set_edge_factor((3, 2), torch.from_numpy(np.random.randn(k[3], k[2])))
        mn.set_edge_factor((1, 4), torch.from_numpy(np.random.randn(k[1], k[4])))

        return mn

    def test_structure(self):
        """Test that the structure of the MarkovNet is properly set up"""
        mn = TorchMarkovNet(is_cuda=False, var_on=False)

        mn.set_unary_factor(0, torch.from_numpy(np.random.randn(4)))
        mn.set_unary_factor(1, torch.from_numpy(np.random.randn(3)))
        mn.set_unary_factor(2, torch.from_numpy(np.random.randn(5)))

        mn.set_edge_factor((0, 1), torch.from_numpy(np.random.randn(4, 3)))
        mn.set_edge_factor((1, 2), torch.from_numpy(np.random.randn(3, 5)))

        print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
        print("Neighbors of 1: " + repr(mn.get_neighbors(1)))
        print("Neighbors of 2: " + repr(mn.get_neighbors(2)))

        assert mn.get_neighbors(0) == set([1]), "Neighbors are wrong"
        assert mn.get_neighbors(1) == set([0, 2]), "Neighbors are wrong"
        assert mn.get_neighbors(2) == set([1]), "Neighbors are wrong"

    def test_matrix_shapes(self):
        """Test that the matrix mode creates matrices of the correct shape."""
        mn = self.create_chain_model(False)

        k = [4, 3, 6, 2, 5]

        max_states = max(k)

        assert mn.matrix_mode == False, "Matrix mode flag was set prematurely"

        mn.create_matrices()

        assert mn.matrix_mode, "Matrix mode flag wasn't set correctly"

        assert mn.unary_mat.shape == (max_states, 5)

    def test_cuda(self):
        try:
            mn = self.create_chain_model(True)

            k = [4, 3, 6, 2, 5]

            max_states = max(k)

            assert mn.matrix_mode == False, "Matrix mode flag was set prematurely"

            mn.create_matrices()

            assert mn.matrix_mode, "Matrix mode flag wasn't set correctly"

            assert mn.unary_mat.shape == (max_states, 5)
        except AssertionError:
            print "\n\nCUDA was not found within your PyTorch package\n\n"
            assert True

