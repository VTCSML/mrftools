import numpy as np
from MarkovNet import MarkovNet
from BeliefPropagator import BeliefPropagator
from MatrixBeliefPropagator import MatrixBeliefPropagator

import unittest


class TestBeliefPropagator(unittest.TestCase):

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
        mn.set_edge_factor((3, 2), np.random.randn(k[3], k[2]))
        mn.set_edge_factor((1, 4), np.random.randn(k[1], k[4]))
        mn.create_matrices()

        return mn

    def create_loop_model(self):
        mn = self.create_chain_model()

        k = [4, 3, 6, 2, 5]

        mn.set_edge_factor((3, 0), np.random.randn(k[3], k[0]))
        mn.create_matrices()
        return mn

    def test_exactness(self):
        mn = self.create_chain_model()
        bp = MatrixBeliefPropagator(mn)

        bp.infer(display='full')

        bp.load_beliefs()

        from BruteForce import BruteForce

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

        slow_bp = BeliefPropagator(mn)

        bp = MatrixBeliefPropagator(mn)

        bp.set_max_iter(30000)
        slow_bp.set_max_iter(30000)

        import time

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