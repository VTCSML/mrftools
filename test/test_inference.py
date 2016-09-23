import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, minimize

class TestConvexBP(unittest.TestCase):
    def create_q_model(self):
        """Test basic functionality of BeliefPropagator."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]
        k = [3, 3, 3, 3, 3]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))
        mn.set_unary_factor(4, np.random.randn(k[4]))

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((2, 3), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((0, 3), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((0, 4), np.random.randn(k[0], k[4]))
        mn.create_matrices()
        mn.tree_probabilities = {(0, 1): 0.75, (1, 2): 0.75, (2, 3): 0.75, (0, 3): 0.75, (0, 4): 1.0}

        return mn

    def create_tree_model(self):
        """Test basic functionality of BeliefPropagator."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]
        k = [3, 3, 3, 3, 3]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))
        mn.set_unary_factor(4, np.random.randn(k[4]))

        # 0-1-2-3, 2-4
        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((2, 3), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((2, 4), np.random.randn(k[0], k[4]))
        mn.create_matrices()
        mn.tree_probabilities = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (2, 4): 1.0}

        return mn

    def test_message_stationarity(self):
        mn = self.create_q_model()

        for inference_type in [ConvexBeliefPropagator, MatrixBeliefPropagator, MatrixTRBeliefPropagator]:
            bp = inference_type(mn)
            bp.set_max_iter(10000)
            bp.infer(display = "full", tolerance=1e-12)

            messages = bp.message_mat.copy()

            def dual(message_delta):
                bp.message_mat = messages + message_delta.reshape(messages.shape)

                return bp.compute_dual_objective()

            grad = approx_fprime(np.zeros(messages.size), dual, 1e-12)

            opt_delta = minimize(dual, np.zeros(messages.size))

            bp.message_mat = messages
            print "Dual objective using BP solution:  %e" % bp.compute_dual_objective()
            print "Inconsistency using BP solution:  %e" % bp.compute_inconsistency()
            bp.message_mat = messages + opt_delta.x.reshape(messages.shape)
            print "Dual objective via numerical grad: %e" % bp.compute_dual_objective()
            print "Inconsistency via numerical grad:  %e" % bp.compute_inconsistency()

            # print inference_type, grad

            assert np.linalg.norm(grad) < 1e-6, "gradient norm was not close enough to zero"

    def test_belief_stationarity(self):
        mn = self.create_tree_model()

        bf = BruteForce(mn)

        for inference_type in [MatrixBeliefPropagator, ConvexBeliefPropagator, MatrixTRBeliefPropagator]:
            bp = inference_type(mn)

            # bp.update_messages()
            # bp.update_messages()
            bp.set_max_iter(3000)
            bp.infer(display='off', tolerance=1e-16)

            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()

            bp.load_beliefs()

            beliefs = bp.belief_mat

            def dual(belief_change):
                bp.belief_mat = beliefs + belief_change.reshape(beliefs.shape)
                bp.belief_mat -= logsumexp(bp.belief_mat, 0)
                return bp.compute_energy() + bp.compute_bethe_entropy() + \
                    np.sum(bp.message_mat * bp._compute_inconsistency_vector())

            print ("Bethe energy functional: %f" % dual(np.zeros(beliefs.size)))

            true_log_z = np.log(bf.compute_z())

            print ("Brute force log partition function: %f" % true_log_z)

            assert np.allclose(dual(np.zeros(beliefs.size)), true_log_z), "Partition function estimate is off"

            grad = approx_fprime(np.zeros(beliefs.size), dual, 1e-12)

            print grad

            assert np.linalg.norm(grad) < 1e-6, "gradient norm was not close enough to zero"

    def test_belief_independence(self):
        mn = self.create_q_model()

        for inference_type in [ConvexBeliefPropagator, MatrixBeliefPropagator, MatrixTRBeliefPropagator]:
            bp = inference_type(mn)

            bp.update_messages()

            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()
            old_beliefs = bp.belief_mat
            old_pairwise_beliefs = bp.pair_belief_tensor
            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()
            assert np.allclose(old_beliefs, bp.belief_mat), \
                "Unary beliefs changed despite no change in messages"
            assert np.allclose(old_pairwise_beliefs, bp.pair_belief_tensor), \
                "Pair beliefs changed despite no change in messages"

if __name__ == '__main__':
    unittest.main()
