import unittest
from MarkovNet import MarkovNet
import numpy as np
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator


class TestLogLinearModel(unittest.TestCase):

    def create_chain_model(self):
        """Test basic functionality of BeliefPropagator."""
        mn = LogLinearModel()

        np.random.seed(1)

        k = [4, 4, 4, 4, 4]

        for i in range(len(k)):
            mn.set_unary_factor(i, np.random.randn(k[i]))

        factor4 = np.random.randn(k[4])
        factor4[2] = -float('inf')

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
        mn.set_edge_factor((1, 4), np.random.randn(k[1], k[4]))

        d = 4

        for i in range(len(k)):
            mn.set_unary_features(i, np.random.randn(d))

        return mn

    def test_structure(self):
        mn = LogLinearModel()

        mn.set_unary_factor(0, np.random.randn(4))
        mn.set_unary_factor(1, np.random.randn(3))
        mn.set_unary_factor(2, np.random.randn(5))

        mn.set_edge_factor((0, 1), np.random.randn(4, 3))
        mn.set_edge_factor((1, 2), np.random.randn(3, 5))

        print("Neighbors of 0: " + repr(mn.get_neighbors(0)))
        print("Neighbors of 1: " + repr(mn.get_neighbors(1)))
        print("Neighbors of 2: " + repr(mn.get_neighbors(2)))

        assert mn.get_neighbors(0) == set([1]), "Neighbors are wrong"
        assert mn.get_neighbors(1) == set([0, 2]), "Neighbors are wrong"
        assert mn.get_neighbors(2) == set([1]), "Neighbors are wrong"

    def test_matrix_shapes(self):
        mn = self.create_chain_model()

        k = [4, 4, 4, 4, 4]

        max_states = max(k)

        assert mn.matrix_mode == False, "Matrix mode flag was set prematurely"

        mn.create_matrices()

        assert mn.matrix_mode, "Matrix mode flag wasn't set correctly"

        assert mn.unary_mat.shape == (max_states, 5)

    def test_feature_computation(self):
        mn = self.create_chain_model()

        k = [4, 4, 4, 4, 4]

        d = 4

        for i in range(len(k)):
            mn.set_unary_weights(i, np.random.randn(k[i], d))

    def test_edge_features(self):
        mn = self.create_chain_model()

        d = 3

        for i in range(5):
            mn.set_edge_features((i, i+1), np.random.randn(d))

        mn.create_matrices()
        mn.set_weight_matrix(np.random.randn(4, 4))
        mn.set_edge_weight_matrix(np.random.randn(d, 16))

        bp = MatrixBeliefPropagator(mn)

        bp.infer()
        bp.load_beliefs()

        unconditional_marginals = bp.var_beliefs[4]

        bp.condition(0, 2)
        bp.infer()
        bp.load_beliefs()

        conditional_marginals = bp.var_beliefs[4]

        assert not np.allclose(unconditional_marginals, conditional_marginals), \
                "Conditioning on variable 0 did not change marginal of variable 4"

        mn.set_edge_features((2, 3), np.zeros(d))
        mn.create_matrices()
        mn.set_weight_matrix(np.random.randn(4, 4))
        mn.set_edge_weight_matrix(np.random.randn(d, 16))

        bp.infer()
        bp.load_beliefs()

        unconditional_marginals = bp.var_beliefs[4]

        bp.condition(0, 2)
        bp.infer()
        bp.load_beliefs()

        conditional_marginals = bp.var_beliefs[4]

        assert np.allclose(unconditional_marginals, conditional_marginals), \
            "Conditioning on var 0 changed marginal of var 4, when the features should have made them independent"