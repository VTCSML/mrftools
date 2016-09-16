import numpy as np
from mrftools import *
import unittest
import time


class TestMatrixBeliefPropagator(unittest.TestCase):

    def create_chain_model(self):
        """Test basic functionality of BeliefPropagator."""
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
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        mn.set_edge_factor((3, 0), np.random.randn(k[3], k[0]))
        mn.create_matrices()
        return mn

    def create_grid_model(self):
        mn = MarkovNet()

        length = 16

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
        mn = self.create_chain_model()
        bp = MatrixBeliefPropagator(mn)
        bp.infer(display='full')
        bp.load_beliefs()

        bf = BruteForce(mn)

        for i in mn.variables:
            print ("Brute force unary marginal of %d: %s" % (i, repr(bf.unary_marginal(i))))
            print ("Belief prop unary marginal of %d: %s" % (i, repr(np.exp(bp.var_beliefs[i]))))
            assert np.allclose(bf.unary_marginal(i), np.exp(bp.var_beliefs[i])), "beliefs aren't exact on chain model"

        print ("Brute force pairwise marginal: " + repr(bf.pairwise_marginal(0, 1)))
        print ("Belief prop pairwise marginal: " + repr(np.exp(bp.pair_beliefs[(0, 1)])))

        print ("Bethe energy functional: %f" % bp.compute_energy_functional())

        print ("Brute force log partition function: %f" % np.log(bf.compute_z()))

        assert np.allclose(np.log(bf.compute_z()), bp.compute_energy_functional()),\
            "log partition function is not exact on chain model"

    def test_consistency(self):
        mn = self.create_loop_model()

        bp = MatrixBeliefPropagator(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                print pair_belief, unary_belief
                assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"

    def test_normalization(self):
        mn = self.create_loop_model()

        bp = MatrixBeliefPropagator(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            assert np.allclose(np.sum(unary_belief), 1.0), "unary belief is not normalized"
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.exp(bp.pair_beliefs[(var, neighbor)])
                assert np.allclose(np.sum(pair_belief), 1.0), "pairwise belief is not normalize"

    def test_speedup(self):
        mn = self.create_grid_model()

        slow_bp = BeliefPropagator(mn)

        bp = MatrixBeliefPropagator(mn)

        bp.set_max_iter(30000)
        slow_bp.set_max_iter(30000)

        t0 = time.time()
        bp.infer(display='final')
        t1 = time.time()

        bp_time = t1 - t0

        t0 = time.time()
        slow_bp.infer(display='final')
        t1 = time.time()

        slow_bp_time = t1 - t0

        print("Matrix BP took %f, loop-based BP took %f. Speedup was %f" % \
              (bp_time, slow_bp_time, slow_bp_time / bp_time))
        assert bp_time < slow_bp_time, "matrix form was slower than loop-based BP"

        # check marginals
        bp.load_beliefs()
        slow_bp.compute_beliefs()
        slow_bp.compute_pairwise_beliefs()

        for var in mn.variables:
            assert np.allclose(bp.var_beliefs[var], slow_bp.var_beliefs[var]), "unary beliefs don't agree"
            for neighbor in mn.get_neighbors(var):
                edge = (var, neighbor)
                assert np.allclose(bp.pair_beliefs[edge], slow_bp.pair_beliefs[edge]), "pairwise beliefs don't agree" \
                           + "\n" + repr(bp.pair_beliefs[edge]) \
                           + "\n" + repr(slow_bp.pair_beliefs[edge])

    def test_conditioning(self):
        mn = self.create_loop_model()

        bp = MatrixBeliefPropagator(mn)

        bp.condition(2, 0)

        bp.infer()
        bp.load_beliefs()

        assert np.allclose(bp.var_beliefs[2][0], 0), "Conditioned variable was not set to correct state"

        beliefs0 = bp.var_beliefs[0]

        bp.condition(2, 1)
        bp.infer()
        bp.load_beliefs()
        beliefs1 = bp.var_beliefs[0]

        assert not np.allclose(beliefs0, beliefs1), "Conditioning var 2 did not change beliefs of var 0"

    def test_overflow(self):
        mn = self.create_chain_model()

        # set a really large factor
        mn.set_unary_factor(0, [1000, 2000, 3000, 4000])

        mn.create_matrices()

        bp = MatrixBeliefPropagator(mn)

        with np.errstate(all='raise'):
            bp.infer()
            bp.load_beliefs()

    def test_grid_consistency(self):
        mn = self.create_grid_model()
        bp = MatrixBeliefPropagator(mn)
        bp.infer(display='full')

        bp.load_beliefs()

        for var in mn.variables:
            unary_belief = np.exp(bp.var_beliefs[var])
            for neighbor in mn.get_neighbors(var):
                pair_belief = np.sum(np.exp(bp.pair_beliefs[(var, neighbor)]), 1)
                # print pair_belief, unary_belief
                assert np.allclose(pair_belief, unary_belief), "unary and pairwise beliefs are inconsistent"

    def test_belief_propagator_messages(self):
        model = self.create_grid_model_simple_edges()
        bp = BeliefPropagator(model)
        bp.load_beliefs()

        mat_bp = MatrixBeliefPropagator(model)
        mat_bp.load_beliefs()

        for i in range(4):
            for var in sorted(bp.mn.variables):
                for neighbor in sorted(bp.mn.get_neighbors(var)):
                    edge = (var, neighbor)
                    bp_message = bp.messages[edge]

                    if edge in mat_bp.mn.edge_index:
                        edge_index = mat_bp.mn.edge_index[edge]
                    else:
                        edge_index = mat_bp.mn.edge_index[(edge[1], edge[0])] + mat_bp.mn.num_edges

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
