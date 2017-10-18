"""Tests for Matrix belief propagation"""
import numpy as np
from mrftools import *
import unittest
import time
import torch

class TestTorchMatrixBeliefPropagator(unittest.TestCase):
    """Test class for TorchMatrixBeliefPropagator"""
    def create_chain_model(self):
        """Create chain MRF with variable of different cardinalities."""
        mn = TorchMarkovNet()

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
        mn.set_edge_factor((2, 3), torch.from_numpy(np.random.randn(k[2], k[3])))
        mn.set_edge_factor((3, 4), torch.from_numpy(np.random.randn(k[3], k[4])))
        mn.create_matrices()

        return mn

    def create_chain_model_old(self):
        """Create chain MRF with variable of different cardinalities."""
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

    def create_loop_model(self):
        """Create a loop-structured MRF"""
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        mn.set_edge_factor((3, 0), torch.from_numpy(np.random.randn(k[3], k[0])))
        mn.create_matrices()
        return mn

    def create_grid_model(self):
        """Create a grid-structured MRF"""
        mn = TorchMarkovNet()

        length = 64

        k = 8

        for x in range(length):
            for y in range(length):
                mn.set_unary_factor((x, y), torch.from_numpy(np.random.random(k)))

        for x in range(length - 1):
            for y in range(length):
                mn.set_edge_factor(((x, y), (x + 1, y)), torch.from_numpy(np.random.random((k, k))))
                mn.set_edge_factor(((y, x), (y, x + 1)), torch.from_numpy(np.random.random((k, k))))

        return mn

    def create_grid_model_old(self):
        """Create a grid-structured MRF"""
        mn = MarkovNet()

        length = 64

        k = 8

        for x in range(length):
            for y in range(length):
                mn.set_unary_factor((x, y), np.random.random(k))

        for x in range(length - 1):
            for y in range(length):
                mn.set_edge_factor(((x, y), (x + 1, y)), np.random.random((k, k)))
                mn.set_edge_factor(((y, x), (y, x + 1)), np.random.random((k, k)))

        return mn

    def create_grid_model_simple_edges(self):
        """Create a grid-structured MRFs with edge potentials that are attractive."""
        mn = TorchMarkovNet()

        length = 2

        k = 8

        for x in range(length):
            for y in range(length):
                mn.set_unary_factor((x, y), torch.from_numpy(np.random.random(k)))

        for x in range(length - 1):
            for y in range(length):
                mn.set_edge_factor(((x, y), (x + 1, y)), torch.eye(k))
                mn.set_edge_factor(((y, x), (y, x + 1)), torch.eye(k))

        return mn

    def create_grid_model_simple_edges_old(self):
        """Create a grid-structured MRFs with edge potentials that are attractive."""
        mn = MarkovNet()

        length = 2

        k = 8

        for x in range(length):
            for y in range(length):
                mn.set_unary_factor((x, y), np.random.random(k))

        for x in range(length - 1):
            for y in range(length):
                mn.set_edge_factor(((x, y), (x + 1, y)), np.eye(k))
                mn.set_edge_factor(((y, x), (y, x + 1)), np.eye(k))

        return mn

    def test_exactness(self):
        """Test that Matrix BP produces the true marginals in a chain model."""
        mn = self.create_chain_model()
        bp = TorchMatrixBeliefPropagator(mn)
        bp.infer(display='full')
        bp.load_beliefs()

        mn_old = self.create_chain_model_old()
        bf = BruteForce(mn_old)


        for i in mn.variables:
            print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unary_marginal(i))))
            print ("Belief prop unary marginal of %d: %s" % (i, repr(torch.exp(bp.var_beliefs[i]))))
            assert np.allclose(bf.unary_marginal(i), torch.exp(bp.var_beliefs[i]).numpy()), "beliefs aren't exact on chain model"

        print ("Brute force pairwise marginal: " + repr(bf.pairwise_marginal(0, 1)))
        print ("Belief prop pairwise marginal: " + repr(torch.exp(bp.pair_beliefs[(0, 1)])))

        print ("Bethe energy functional: %f" % bp.compute_energy_functional())

        print ("Brute force log partition function: %f" % np.log(bf.compute_z()))

        assert np.allclose(np.log(bf.compute_z()), bp.compute_energy_functional()),\
            "log partition function is not exact on chain model"

    def test_consistency(self):
        """Test that loopy matrix BP infers marginals that are locally consistent."""
        mn = self.create_loop_model()

        bp = TorchMatrixBeliefPropagator(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = torch.exp(bp.var_beliefs[var])
            for neighbor in mn.get_neighbors(var):
                pair_belief = torch.sum(torch.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                print pair_belief, unary_belief
                assert np.allclose(pair_belief.numpy(), unary_belief.numpy()), "unary and pairwise beliefs are inconsistent"

    def test_normalization(self):
        """Test that the unary and pairwise beliefs properly sum to 1.0"""
        mn = self.create_loop_model()

        bp = TorchMatrixBeliefPropagator(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = torch.exp(bp.var_beliefs[var])
            assert np.allclose(torch.sum(unary_belief), 1.0), "unary belief is not normalized"
            for neighbor in mn.get_neighbors(var):
                pair_belief = torch.exp(bp.pair_beliefs[(var, neighbor)])
                assert np.allclose(torch.sum(pair_belief), 1.0), "pairwise belief is not normalize"

    def test_conditioning(self):
        """Test that conditioning on variable properly sets variables to conditioned state"""
        mn = self.create_loop_model()

        bp = TorchMatrixBeliefPropagator(mn)

        bp.condition(2, 0)

        bp.infer()
        bp.load_beliefs()

        assert np.allclose(bp.var_beliefs[2][0], 0), "Conditioned variable was not set to correct state"

        beliefs0 = bp.var_beliefs[0]

        bp.condition(2, 1)
        bp.infer()
        bp.load_beliefs()
        beliefs1 = bp.var_beliefs[0]

        assert not np.allclose(beliefs0.numpy(), beliefs1.numpy()), "Conditioning var 2 did not change beliefs of var 0"

    def test_overflow(self):
        """Test that MatrixBP does not fail when given very large, poorly scaled factors"""
        mn = self.create_chain_model()

        # set a really large factor
        mn.set_unary_factor(0, torch.FloatTensor([1000, 2000, 3000, 4000]))

        mn.create_matrices()

        bp = TorchMatrixBeliefPropagator(mn)

        with np.errstate(all='raise'):
            bp.infer()
            bp.load_beliefs()

    # GRID MODEL FAILS BECAUSE OF FLOAT v DOUBLE
    # GRID MODEL (SIMPLE EDGES) FAILS BECAUSE .ravel METHOD DOES NOT EXIST YET
    '''
    def test_speedup(self):
        """Test that matrix BP is faster than loop-based BP"""
        mn = self.create_grid_model()
        old_mn = self.create_grid_model_old()

        bp = TorchMatrixBeliefPropagator(mn)
        old_bp = MatrixBeliefPropagator(old_mn)

        bp.set_max_iter(1000)
        old_bp.set_max_iter(1000)

        t0 = time.time()
        bp.infer(display='final')
        t1 = time.time()

        bp_time = t1 - t0

        t0 = time.time()
        old_bp.infer(display='final')
        t1 = time.time()

        old_bp_time = t1 - t0

        print("Torch Matrix BP took %f, Matrix BP took %f. Speedup was %f" %
              (bp_time, old_bp_time, old_bp_time / bp_time))
        assert bp_time < old_bp_time, "Torch Matrix form was slower than Matrix BP"

        # check marginals
        bp.load_beliefs()
        old_bp.compute_beliefs()
        old_bp.compute_pairwise_beliefs()

        for var in mn.variables:
            assert np.allclose(bp.var_beliefs[var], old_bp.var_beliefs[var]), "unary beliefs don't agree"
            for neighbor in mn.get_neighbors(var):
                edge = (var, neighbor)
                assert np.allclose(bp.pair_beliefs[edge], old_bp.pair_beliefs[edge]), "pairwise beliefs don't agree" \
                           + "\n" + repr(bp.pair_beliefs[edge]) \
                           + "\n" + repr(old_bp.pair_beliefs[edge])

    def test_grid_consistency(self):
        """Test that matrix BP infers consistent marginals on a grid MRF"""
        mn = self.create_grid_model()
        bp = TorchMatrixBeliefPropagator(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                # print pair_belief, unary_belief
                assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"

    def test_belief_propagator_messages(self):
        """Test that matrix BP and loop-based BP calculate the same messages and beliefs each iteration of inference"""
        model_old = self.create_grid_model_simple_edges_old()
        model = self.create_grid_model_simple_edges()
        bp = BeliefPropagator(model_old)
        bp.load_beliefs()

        mat_bp = TorchMatrixBeliefPropagator(model)
        mat_bp.load_beliefs()

        for i in range(4):
            for var in sorted(bp.mn.variables):
                for neighbor in sorted(bp.mn.get_neighbors(var)):
                    edge = (var, neighbor)
                    bp_message = bp.messages[edge]

                    if edge in mat_bp.mn.message_index:
                        edge_index = mat_bp.mn.message_index[edge]
                    else:
                        edge_index = mat_bp.mn.message_index[(edge[1], edge[0])] + mat_bp.mn.num_edges

                    mat_bp_message = mat_bp.message_mat[:, edge_index].ravel()

                    assert np.allclose(bp_message, mat_bp_message), \
                        "BP and matBP did not agree on message for edge %s in iter %d" % (repr(edge), i) \
                        + "\nBP: " + repr(bp_message) + "\nmatBP: " + repr(mat_bp_message)

                    # print "Message %s is OK" % repr(edge)

                    assert np.allclose(bp.pair_beliefs[edge], mat_bp.pair_beliefs[edge]), \
                        "BP and matBP did not agree on pair beliefs after %d message updates" % i

                assert np.allclose(bp.var_beliefs[var], mat_bp.var_beliefs[var]), \
                    "BP and matBP did not agree on unary beliefs after %d message updates" % i

            bp.update_messages()
            bp.load_beliefs()
            mat_bp.update_messages()
            mat_bp.load_beliefs()

    '''
