import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt

class TestConvexBP(unittest.TestCase):
    def create_q_model(self):
        """Test basic functionality of BeliefPropagator."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [4, 3, 6, 2, 5]

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

        return mn

    def create_q_model1(self):
        """Test basic functionality of BeliefPropagator."""
        mn = MarkovNet()

        np.random.seed(1)

        k = [2, 2, 2, 2]

        mn.set_unary_factor(0, np.random.randn(k[0]))
        mn.set_unary_factor(1, np.random.randn(k[1]))
        mn.set_unary_factor(2, np.random.randn(k[2]))
        mn.set_unary_factor(3, np.random.randn(k[3]))

        # mn.set_unary_factor(4, np.random.randn(k[4]))
        mn.set_unary_factor(4, np.random.randn(k[0]))
        mn.set_unary_factor(5, np.random.randn(k[1]))
        mn.set_unary_factor(6, np.random.randn(k[2]))
        mn.set_unary_factor(7, np.random.randn(k[3]))

        mn.set_unary_factor(8, np.random.randn(k[0]))
        mn.set_unary_factor(9, np.random.randn(k[1]))
        mn.set_unary_factor(10, np.random.randn(k[2]))
        mn.set_unary_factor(11, np.random.randn(k[3]))

        mn.set_edge_factor((0, 1), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((1, 2), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((2, 3), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((0, 3), np.random.randn(k[0], k[3]))
        # mn.set_edge_factor((0, 4), np.random.randn(k[0], k[4]))

        mn.set_edge_factor((4, 5), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((5, 6), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((6, 7), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((4, 7), np.random.randn(k[0], k[3]))

        mn.set_edge_factor((8, 9), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((9, 10), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((10, 11), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((8, 11), np.random.randn(k[0], k[3]))

        # connect block 1 with block 2

        mn.set_edge_factor((1, 4), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((2, 7), np.random.randn(k[0], k[3]))

        # connect block 2 with block 3

        mn.set_edge_factor((5, 8), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((6, 11), np.random.randn(k[0], k[3]))

        # -----------------------------------------

        mn.set_unary_factor(12, np.random.randn(k[0]))
        mn.set_unary_factor(13, np.random.randn(k[1]))
        mn.set_unary_factor(14, np.random.randn(k[2]))
        mn.set_unary_factor(15, np.random.randn(k[3]))

        # mn.set_unary_factor(4, np.random.randn(k[4]))
        mn.set_unary_factor(16, np.random.randn(k[0]))
        mn.set_unary_factor(17, np.random.randn(k[1]))
        mn.set_unary_factor(18, np.random.randn(k[2]))
        mn.set_unary_factor(19, np.random.randn(k[3]))

        mn.set_unary_factor(20, np.random.randn(k[0]))
        mn.set_unary_factor(21, np.random.randn(k[1]))
        mn.set_unary_factor(22, np.random.randn(k[2]))
        mn.set_unary_factor(23, np.random.randn(k[3]))

        mn.set_edge_factor((12, 13), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((13, 14), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((14, 15), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((12, 15), np.random.randn(k[0], k[3]))
        # mn.set_edge_factor((0, 4), np.random.randn(k[0], k[4]))


        mn.set_edge_factor((16, 17), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((17, 18), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((18, 19), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((16, 19), np.random.randn(k[0], k[3]))

        mn.set_edge_factor((20, 21), np.random.randn(k[0], k[1]))
        mn.set_edge_factor((21, 22), np.random.randn(k[1], k[2]))
        mn.set_edge_factor((22, 23), np.random.randn(k[2], k[3]))
        mn.set_edge_factor((20, 23), np.random.randn(k[0], k[3]))

        # connect block 4 with block 5
        mn.set_edge_factor((13, 16), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((14, 19), np.random.randn(k[0], k[3]))

        # connect block 5 with block 6
        mn.set_edge_factor((17, 20), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((23, 22), np.random.randn(k[0], k[3]))

        ## Vertical:

        # connect block 1 with block 4
        mn.set_edge_factor((3, 12), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((2, 13), np.random.randn(k[0], k[3]))

        # connect block 2 with block 5
        mn.set_edge_factor((7, 16), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((6, 17), np.random.randn(k[0], k[3]))

        # connect block 3 with block 6
        mn.set_edge_factor((11, 20), np.random.randn(k[0], k[3]))
        mn.set_edge_factor((10, 21), np.random.randn(k[0], k[3]))


        mn.create_matrices()

        return mn

    def test_comparison_to_trbp(self):
        mn = self.create_q_model()

        probs = {(0, 1): 0.75, (1, 2): 0.75, (2, 3): 0.75, (0, 3): 0.75, (0, 4): 1.0}

        trbp_mat = MatrixTRBeliefPropagator(mn, probs)
        trbp_mat.infer(mn.unary_mat, mn.edge_pot_tensor, display='full')
        trbp_mat.load_beliefs()

        counting_numbers = probs.copy()
        counting_numbers[0] = 1.0 - 2.5
        counting_numbers[1] = 1.0 - 1.5
        counting_numbers[2] = 1.0 - 1.5
        counting_numbers[3] = 1.0 - 1.5
        counting_numbers[4] = 1.0 - 1.0

        cbp = ConvexBeliefPropagator(mn, counting_numbers)
        message_mat = cbp.infer(mn.unary_mat, mn.edge_pot_tensor, display='full')
        cbp.load_beliefs(mn.unary_mat, message_mat, mn.edge_pot_tensor)

        for i in mn.variables:
            print ("Convex unary marginal of %d: %s" % (i, repr(np.exp(cbp.var_beliefs[i]))))
            print ("Matrix TRBP unary marginal of %d: %s" % (i, repr(np.exp(trbp_mat.var_beliefs[i]))))
            assert np.allclose(np.exp(cbp.var_beliefs[i]), np.exp(trbp_mat.var_beliefs[i])), "unary beliefs don't match"

        print ("Convex pairwise marginal: " + repr(np.exp(cbp.pair_beliefs[(0, 1)])))
        print ("Matrix TRBP pairwise marginal: " + repr(np.exp(trbp_mat.pair_beliefs[(0, 1)])))

        print ("Pairwise marginal error %f" %
               np.sum(np.abs(np.exp(cbp.pair_beliefs[(0, 1)]) - np.exp(trbp_mat.pair_beliefs[(0, 1)]))))

        # plt.subplot(211)
        # plt.imshow(cbp.pair_beliefs[(0, 1)], interpolation='nearest')
        # plt.xlabel('CBP')
        # plt.subplot(212)
        # plt.imshow(trbp_mat.pair_beliefs[(0, 1)], interpolation='nearest')
        # plt.xlabel('TRBP')
        # plt.show()

        assert np.allclose(cbp.pair_beliefs[(0, 1)], trbp_mat.pair_beliefs[(0, 1)]), "Pair beliefs don't match: " + \
                                                                                     "\nCBP:" + repr(
            np.exp(cbp.pair_beliefs[(0, 1)])) + "\nMatTRBP:" + repr(np.exp(trbp_mat.pair_beliefs[(0, 1)]))

        print ("TRBP matrix energy functional: %f" % trbp_mat.compute_energy_functional())
        print ("Convex energy functional: %f" % cbp.compute_energy_functional())

        assert np.allclose(trbp_mat.compute_energy_functional(), cbp.compute_energy_functional()), \
            "Energy functional is not exact. Convex: %f, Matrix TRBP: %f" % (cbp.compute_energy_functional(),
                                                                             trbp_mat.compute_energy_functional())

    def test_comparison_to_bethe(self):
        mn = self.create_q_model()

        bp = MatrixBeliefPropagator(mn)
        message_mat = bp.infer(mn.unary_mat, mn.edge_pot_tensor, display='full')
        bp.load_beliefs(mn.unary_mat, message_mat, mn.edge_pot_tensor)

        counting_numbers = {(0, 1): 1.0,
                            (1, 2): 1.0,
                            (2, 3): 1.0,
                            (0, 3): 1.0,
                            (0, 4): 1.0,
                            0: 1.0 - 3.0,
                            1: 1.0 - 2.0,
                            2: 1.0 - 2.0,
                            3: 1.0 - 2.0,
                            4: 1.0 - 1.0}

        cbp = ConvexBeliefPropagator(mn, counting_numbers)
        message_mat = cbp.infer(mn.unary_mat, mn.edge_pot_tensor, display='full')
        cbp.load_beliefs(mn.unary_mat, message_mat, mn.edge_pot_tensor)

        for i in mn.variables:
            print ("Convex unary marginal of %d: %s" % (i, repr(np.exp(cbp.var_beliefs[i]))))
            print ("Matrix BP unary marginal of %d: %s" % (i, repr(np.exp(bp.var_beliefs[i]))))
            assert np.allclose(np.exp(cbp.var_beliefs[i]), np.exp(bp.var_beliefs[i])), "unary beliefs don't match"

        print ("Convex pairwise marginal: " + repr(np.exp(cbp.pair_beliefs[(0, 1)])))
        print ("Matrix BP pairwise marginal: " + repr(np.exp(bp.pair_beliefs[(0, 1)])))

        assert np.allclose(cbp.pair_beliefs[(0, 1)], bp.pair_beliefs[(0, 1)]), "Pair beliefs don't match: " + \
                                                                                     "\nCBP:" + repr(
            np.exp(cbp.pair_beliefs[(0, 1)])) + "\nMatBP:" + repr(np.exp(bp.pair_beliefs[(0, 1)]))

        print ("Bethe matrix energy functional: %f" % bp.compute_energy_functional())
        print ("Convex energy functional: %f" % cbp.compute_energy_functional())

        assert np.allclose(bp.compute_energy_functional(), cbp.compute_energy_functional()), \
            "Energy functional is not exact. Convex: %f, BP: %f" % (cbp.compute_energy_functional(),
                                                                             bp.compute_energy_functional())

    def test_bounds(self):
        mn = self.create_q_model()

        edge_count = 1
        node_count = 1

        counting_numbers = {(0, 1): edge_count,
                            (1, 2): edge_count,
                            (2, 3): edge_count,
                            (0, 3): edge_count,
                            (0, 4): edge_count,
                            0: node_count,
                            1: node_count,
                            2: node_count,
                            3: node_count,
                            4: node_count,
                            }
        bp = ConvexBeliefPropagator(mn, counting_numbers)

        max_iter = 30

        primal = np.zeros(max_iter)
        dual = np.zeros(max_iter)
        inconsistency = np.zeros(max_iter)

        print "t\tprimal\t\t\tdual obj\t\tinconsistency\tdiff"

        for t in range(max_iter):
            primal[t] = bp.compute_energy_functional()
            dual[t] = bp.compute_dual_objective()
            inconsistency[t] = bp.compute_inconsistency()

            print "%d\t%e\t%e\t%e" % (t, primal[t], dual[t], inconsistency[t])

            message_mat = bp.initialize_messages()
            bp.update_messages(mn.unary_mat, mn.edge_pot_tensor, message_mat)

        assert np.allclose(primal[-1], dual[-1]), "Primal and dual are not close after %d iters" % max_iter

        opt = primal[-1]

        print "t\tdual obj\t\tdiff"
        for t in range(max_iter):
            print "%d\t%e\t%e" % (t, dual[t], dual[t] - opt)
            assert dual[t] >= opt, "dual objective was lower than optimum"


    def test_convexity(self):
        mn = self.create_q_model1()

        edge_count = 1
        node_count = 1.0

        counting_numbers = {(0, 1): 1.0,
                            (1, 2): 1.0,
                            (2, 3): 1.0,
                            (0, 3): 1.0,

                            (4, 5): 1.0,
                            (5, 6): 1.0,
                            (6, 7): 1.0,
                            (4, 7): 1.0,

                            (8, 9): 1.0,
                            (9, 10): 1.0,
                            (10, 11): 1.0,
                            (8, 11): 1.0,

                            (12, 13): 1.0,
                            (13, 14): 1.0,
                            (14, 15): 1.0,
                            (12, 15): 1.0,

                            (16, 17): 1.0,
                            (17, 18): 1.0,
                            (18, 19): 1.0,
                            (16, 19): 1.0,

                            (20, 21): 1.0,
                            (21, 22): 1.0,
                            (22, 23): 1.0,
                            (20, 23): 1.0,


                            (1, 4): 1.0,
                            (2, 7): 1.0,

                            (5, 8): 1.0,
                            (6, 11): 1.0,

                            (13, 16): 1.0,
                            (14, 19): 1.0,

                            (17, 20): 1.0,
                            (18, 23): 1.0,

                            # vertical
                            (3, 12): 1.0,
                            (2, 13): 1.0,

                            (7, 16): 1.0,
                            (6, 17): 1.0,

                            (11, 20): 1.0,
                            (10, 21): 1.0,
                            0: node_count,
                            1: node_count,
                            2: node_count,
                            3: node_count,
                            4: node_count,
                            5: node_count,
                            6: node_count,
                            7: node_count,
                            8: node_count,
                            9: node_count,
                            10: node_count,
                            11: node_count,
                            12: node_count,
                            13: node_count,
                            14: node_count,
                            15: node_count,
                            16: node_count,
                            17: node_count,
                            18: node_count,
                            19: node_count,
                            20: node_count,
                            21: node_count,
                            22: node_count,
                            23: node_count}

        bp = ConvexBeliefPropagator(mn, counting_numbers)
        # bp = MatrixBeliefPropagator(mn)
        bp.set_max_iter(10)
        message_mat = bp.infer(mn.unary_mat, mn.edge_pot_tensor, display = "full", tolerance=1e-12)

        # why does the dual objective go below the primal solution? numerical, or bug?

        messages = bp.message_mat.copy()

        noise = 0.1 * np.random.randn(messages.shape[0], messages.shape[1])
        res = 121

        x = np.linspace(-100, 100, res)
        y = np.zeros(res)
        z = np.zeros(res)
        primal = np.zeros(res)
        dual_penalty = np.zeros(res)

        for i in range(len(x)):
            mod_messages = messages + x[i] * noise
            bp.set_messages(mod_messages)
            y[i] = bp.compute_dual_objective()
            z[i] = bp.compute_inconsistency()
            primal[i] = bp.compute_energy_functional()
            dual_penalty[i] = y[i] - primal[i]

        # mod_messages = messages
        # bp.set_messages(mod_messages)
        # y[0] = bp.compute_dual_objective()
        # z[0] = bp.compute_inconsistency()
        # primal[0] = bp.compute_energy_functional()
        # dual_penalty[0] = y[0] - primal[0]
        #
        # for i in range(1, res):
        #     noise =  np.random.rand(1) * np.random.randn(messages.shape[0], messages.shape[1])
        #     mod_messages = messages + noise
        #     bp.set_messages(mod_messages)
        #     y[i] = bp.compute_dual_objective()
        #     z[i] = bp.compute_inconsistency()
        #     primal[i] = bp.compute_energy_functional()
        #     dual_penalty[i] = y[i] - primal[i]


        bp.load_beliefs(mn.unary_mat, message_mat, mn.edge_pot_tensor)
        # print np.exp(bp.var_beliefs[0])
        # print np.exp(bp.pair_beliefs[(0, 1)])

        print ("Minimum dual objective: %f" % np.min(y))
        print ("Inconsistency at argmin: %f" % z[np.argmin(y)])

        plt.subplot(411)
        plt.plot(y)
        plt.ylabel('dual objective')
        plt.subplot(412)
        plt.plot(z)
        plt.ylabel('inconsistency')
        plt.subplot(413)
        plt.plot(dual_penalty)
        plt.ylabel('dual penalty')
        plt.subplot(414)
        plt.plot(primal)
        plt.ylabel('(infeasible) primal objective')
        plt.show()

        assert np.allclose(y.min(), y[60]), "Minimum was not at converged messages"

        deriv = y[1:] - y[:-1]
        second_deriv = deriv[1:] - deriv[:-1]
        print second_deriv
        assert np.all(second_deriv >= 0), "Estimated second derivative was not non-negative"


    def test_unary_belief_update(self):
        mn = self.create_q_model()
        mn2 = self.create_q_model()

        edge_count = 1.0
        node_count = 1.0

        counting_numbers = {(0, 1): edge_count,
                            (1, 2): edge_count,
                            (2, 3): edge_count,
                            (0, 3): edge_count,
                            (0, 4): edge_count,
                            0: node_count,
                            1: node_count,
                            2: node_count,
                            3: node_count,
                            4: node_count}

        bp = ConvexBeliefPropagator(mn, counting_numbers)

        message_mat = bp.initialize_messages()
        change, message_mat = bp.update_messages(mn.unary_mat, mn.edge_pot_tensor, message_mat)
        # bp.update_messages()
        # bp.infer(display='off')

        belief_mat = bp.compute_beliefs(mn.unary_mat, message_mat)
        bp.compute_pairwise_beliefs(belief_mat, message_mat, mn.edge_pot_tensor)
        beliefs = bp.belief_mat
        pair_beliefs = bp.pair_belief_tensor

        res = 21
        x = np.linspace(-10, 10, res)
        y = np.zeros(res)
        z = np.zeros(res)
        entropy = np.zeros(res)
        energy = np.zeros(res)

        direction = np.random.randn(beliefs.shape[0], beliefs.shape[1])
        # direction *= 0
        # direction[:, 0] = np.random.randn(6)

        for i in range(res):
            new_beliefs = beliefs + x[i] * direction
            new_beliefs -= logsumexp(new_beliefs, 0)

            bp.belief_mat = new_beliefs

            z[i] = bp.compute_energy() + bp.compute_bethe_entropy()
            y[i] = z[i] + bp._compute_dual_penalty()
            entropy[i] = bp.compute_bethe_entropy()
            energy[i] = bp.compute_inconsistency()
            # print y[i]

        plt.subplot(411)
        plt.plot(x, y)
        plt.ylabel('dual objective')
        plt.xlabel('deviation from solution')
        plt.subplot(412)
        plt.plot(x, z)
        plt.ylabel('primal objective')
        plt.xlabel('deviation from solution')
        plt.subplot(413)
        plt.plot(x, entropy)
        plt.ylabel('entropy')
        plt.xlabel('deviation from solution')
        plt.subplot(414)
        plt.plot(x, energy)
        plt.ylabel('energy')
        plt.xlabel('deviation from solution')
        plt.show()

        assert np.allclose(y.max(), y[res / 2]), "Maximum was not at closed-form belief update"


if __name__ == '__main__':
    unittest.main()
