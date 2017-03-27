from __future__ import division
from mrftools import *
import numpy as np

import unittest
from mrftools.grafting_util import compute_likelihood as pseudolikelihood

class TestPseudoLikelihood(unittest.TestCase):

    def test_pseudolikelihood(self):
        mn = MarkovNet()

        np.random.seed(0)

        unary_potential = np.random.randn(2)
        edge_potential = np.random.randn(2, 2)

        mn.set_unary_factor(0, unary_potential)
        mn.set_unary_factor(1, unary_potential)
        mn.set_unary_factor(2, unary_potential)
        mn.set_unary_factor(3, unary_potential)

        mn.set_edge_factor((0, 1), edge_potential)
        mn.set_edge_factor((1, 2), edge_potential)
        mn.set_edge_factor((2, 3), edge_potential)
        # mn.set_edge_factor((3, 0), edge_potential) # uncomment this to make loopy

        gb = GibbsSampler(mn)
        gb.init_states()
        itr = 1000
        num = 10000
        gb.gibbs_sampling(itr, num)

        bf = BruteForce(mn)
        for var in mn.variables:
            gb_result = gb.counter(var) / num
            bf_result = bf.unary_marginal(var)
            print gb_result
            print bf_result
            np.testing.assert_allclose(gb_result, bf_result, rtol=1e-1, atol=0)

        # check pseudolikelihood of data

        npll = pseudolikelihood(mn, 4, gb.samples, mn.variables)

        mn2 = copy.deepcopy(mn)

        # set a random unary potential. Neg pseudoliklihood should go up
        mn2.set_unary_factor(0, np.random.randn(2))
        npll2 = pseudolikelihood(mn2, 4, gb.samples, mn.variables)

        assert(npll <= npll2, "Giving the wrong unary weights made NPLL go up")

        mn2 = copy.deepcopy(mn)

        # set a random edge potential. Neg pseudoliklihood should go up
        mn2.set_edge_factor((1, 2), np.random.randn(2, 2))
        npll2 = pseudolikelihood(mn2, 4, gb.samples, mn.variables)

        assert(npll <= npll2, "Giving the wrong edge weights made NPLL go up")

        mn2 = copy.deepcopy(mn)

        # zero-out an edge potential. Neg pseudoliklihood should go up
        mn2.set_edge_factor((1, 2), np.zeros((2, 2)))
        npll2 = pseudolikelihood(mn2, 4, gb.samples, mn.variables)

        assert(npll <= npll2, "Zeroing out edge weights made NPLL go up")

        # create an exact copy of true Markov net except missing one edge
        # Neg pll should go up

        mn2 = MarkovNet()

        mn2.set_unary_factor(0, unary_potential)
        mn2.set_unary_factor(1, unary_potential)
        mn2.set_unary_factor(2, unary_potential)
        mn2.set_unary_factor(3, unary_potential)

        mn2.set_edge_factor((0, 1), edge_potential)
        mn2.set_edge_factor((1, 2), edge_potential)

        # set a random unary potential. Neg pseudoliklihood should go up
        mn2.set_edge_factor((1, 2), np.zeros((2, 2)))
        npll2 = pseudolikelihood(mn2, 4, gb.samples, mn.variables)

        assert(npll <= npll2, "Omitting an edge made NPLL go up")