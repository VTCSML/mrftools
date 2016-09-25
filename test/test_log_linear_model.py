import unittest
import numpy as np
from mrftools import *

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

    def test_indicator_form(self):
        mn = MarkovNet()

        num_states = [3, 2, 4, 5]

        for i in range(len(num_states)):
            mn.set_unary_factor(i, np.random.randn(num_states[i]))

        edges = [(0, 1), (0, 2), (2, 3), (1, 3)]

        for edge in edges:
            mn.set_edge_factor(edge, np.random.randn(num_states[edge[0]], num_states[edge[1]]))

        model = LogLinearModel()
        model.create_indicator_model(mn)

        bp = BeliefPropagator(mn)
        bp.infer(display='final')
        bp.compute_beliefs()
        bp.compute_pairwise_beliefs()

        bp_ind = MatrixBeliefPropagator(model)
        bp_ind.infer(display='final')
        bp_ind.load_beliefs()

        for i in range(len(num_states)):
            assert np.allclose(bp_ind.var_beliefs[i], bp.var_beliefs[i]), "unary beliefs disagree"

        for edge in edges:
            assert np.allclose(np.sum(np.exp(bp_ind.pair_beliefs[edge])), 1.0), "Pair beliefs don't normalize to 1"
            assert np.allclose(bp_ind.pair_beliefs[edge], bp.pair_beliefs[edge]), "edge beliefs disagree: \n" +\
                "indicator:\n" + repr(bp_ind.pair_beliefs[edge]) + "\noriginal:\n" + repr(bp.pair_beliefs[edge])

    def test_matrix_structure(self):
        model = self.create_chain_model()

        model.create_matrices()

        for edge, i in model.edge_index.items():
            from_index = model.var_index[edge[0]]
            to_index = model.var_index[edge[1]]
            assert model.message_from[i] == from_index, "Message sender index is wrong"
            assert model.message_to[i] == to_index, "Message receiver index is wrong"
            assert model.message_to_map[i, to_index] == 1, "Message receiver matrix map is wrong"

        assert np.all(np.sum(model.message_to_map.todense(), axis=1) == 1), \
            "Message sender map has a row that doesn't sum to 1.0"


    def test_indicator_model(self):
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))

        factor4 = np.random.randn(k[4])

        mn.set_unary_factor(4, factor4)

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
        mn.set_edge_factor((1, 4), np.random.randn(k[1], k[4]))

        # create indicator model

        model = LogLinearModel()
        model.create_indicator_model(mn)

        # try indices

        unary_indices, pairwise_indices = model.get_weight_factor_index()

        print unary_indices

        print pairwise_indices

        print np.sort(np.concatenate((unary_indices.ravel(), pairwise_indices.ravel())))

        assert np.allclose(np.sort(np.concatenate((unary_indices.ravel(), pairwise_indices.ravel()))), np.array(range(model.weight_dim)))

        # try uniform weights

        var = 2
        neighbor = 3
        other = 4

        weights = np.zeros(model.weight_dim)

        bp = MatrixBeliefPropagator(model)

        model.set_weights(weights)
        bp.infer()
        bp.load_beliefs()

        print bp.belief_mat
        # check that beliefs for var, neighbor, and other are all uniform
        assert np.allclose(np.exp(bp.var_beliefs[var][0]), 1.0 / float(k[var]))
        assert np.allclose(np.exp(bp.var_beliefs[neighbor][0]), 1.0 / float(k[neighbor]))
        assert np.allclose(np.exp(bp.var_beliefs[other][0]), 1.0 / float(k[other]))

        # change weights for unary potential

        weights[unary_indices[:, model.var_index[var]]] = np.random.randn(model.max_states)
        model.set_weights(weights)
        bp.infer()
        bp.load_beliefs()

        print bp.belief_mat


        # check that beliefs for var have changed but other remains uniform
        assert np.abs(bp.var_beliefs[var][0]) - 1.0 / float(k[var]) > 1e-8

        assert np.allclose(np.exp(bp.var_beliefs[other][0]), 1.0 / float(k[other]))

        # change weights for pairwise potential

        weights = np.zeros(model.weight_dim)
        weights[pairwise_indices[:, :, model.edge_index[(var, neighbor)]]] = np.random.randn(model.max_states, model.max_states)
        model.set_weights(weights)
        bp.infer()
        bp.compute_pairwise_beliefs()
        bp.load_beliefs()

        print bp.pair_beliefs[(var, neighbor)]

        # check that beliefs for var and neighbor have changed, but not for other
        assert np.abs(bp.var_beliefs[var][0]) - 1.0 / float(k[var]) > 1e-8
        assert np.abs(bp.var_beliefs[neighbor][0]) - 1.0 / float(k[neighbor]) > 1e-8

        assert np.allclose(np.exp(bp.var_beliefs[other][0]), 1.0 / float(k[other]))

