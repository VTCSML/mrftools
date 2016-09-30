import unittest
import numpy as np
from scipy.optimize import check_grad, approx_fprime
import matplotlib.pyplot as plt
from mrftools import *


class TestLearner(unittest.TestCase):

    def set_up_learner(self, learner, latent=True):
        d = 2
        num_states = 4

        np.random.seed(0)

        if latent:
            labels = [{0: 2,       2: 1},
                      {      1: 2, 2: 0},
                      {0: 2, 1: 3,     },
                      {0: 0, 1: 2, 2: 3}]
        else:
            labels = [{0: 2, 1: 3, 2: 1},
                      {0: 3, 1: 2, 2: 0},
                      {0: 2, 1: 3, 2: 1},
                      {0: 0, 1: 2, 2: 3}]

        models = []
        for i in range(len(labels)):
            m = self.create_random_model(num_states, d)
            models.append(m)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

    def test_gradient(self):
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        learner.set_regularization(0.0, 1.0)
        gradient_error = check_grad(learner.subgrad_obj, learner.subgrad_grad, weights)

        # numerical_grad = approx_fprime(weights, learner.subgrad_obj, 1e-4)
        # analytical_grad = learner.subgrad_grad(weights)
        # plt.plot(numerical_grad, 'r')
        # plt.plot(analytical_grad, 'b')
        # plt.show()

        print("Gradient error: %f" % gradient_error)
        assert gradient_error < 1e-1, "Gradient is wrong"

    def test_fully_observed_gradient(self):
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=False)
        learner.set_regularization(0.0, 1.0)
        gradient_error = check_grad(learner.subgrad_obj, learner.subgrad_grad, weights)

        # numerical_grad = approx_fprime(weights, learner.subgrad_obj, 1e-4)
        # analytical_grad = learner.subgrad_grad(weights)
        # plt.plot(numerical_grad, 'r')
        # plt.plot(analytical_grad, 'b')
        # plt.show()

        print("Gradient error: %f" % gradient_error)
        assert gradient_error < 1e-1, "Gradient is wrong"

    def test_m_step_gradient(self):
        weights = np.zeros(8 + 32)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        learner.set_regularization(0.0, 1.0)
        learner.e_step(weights)
        gradient_error = check_grad(learner.objective, learner.gradient, weights)

        # numerical_grad = approx_fprime(weights, learner.objective, 1e-4)
        # analytical_grad = learner.gradient(weights)
        # plt.plot(numerical_grad, 'r')
        # plt.plot(analytical_grad, 'b')
        # plt.show()

        print("Gradient error: %f" % gradient_error)
        assert gradient_error < 1e-1, "Gradient is wrong"

    def test_learner(self):
        weights = np.zeros(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights,wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]
        old_obj = np.Inf
        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i,:])
            assert (new_obj <= old_obj + 1e-8), "subgradient objective is not decreasing"
            old_obj = new_obj

            assert new_obj >= 0, "Learner objective was not non-negative"

    def test_EM(self):
        weights = np.zeros(8 + 32)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]

        old_obj = learner.subgrad_obj(weight_record[0,:])
        new_obj = learner.subgrad_obj(weight_record[-1,:])
        assert (new_obj <= old_obj), "EM objective did not decrease"

        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert new_obj >= 0, "EM objective was not non-negative"

    def test_paired_dual(self):
        weights = np.zeros(8 + 32)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]

        old_obj = learner.subgrad_obj(weight_record[0, :])
        new_obj = learner.subgrad_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "paired dual objective did not decrease"

        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert new_obj >= 0, "Paired dual objective was not non-negative"

    def test_dual(self):
        weights = np.zeros(8 + 32)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner, latent=False)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]

        old_obj = learner.subgrad_obj(weight_record[0, :])
        new_obj = learner.subgrad_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "Dual objective did not decrease"

        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert new_obj >= 0, "Dual objective was not non-negative"

    def test_overflow(self):
        """Initialize weights to a huge number and see if learner can escape it"""
        weights = 1000 * np.random.randn(8 + 32)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        assert not np.isnan(learner.subgrad_obj(weights)), \
            "Objective for learner was not a number"

    def create_random_model(self, num_states, d):
        model = LogLinearModel()

        model.declare_variable(0, num_states)
        model.declare_variable(1, num_states)
        model.declare_variable(2, num_states)

        model.set_unary_weights(0, np.random.randn(num_states, d))
        model.set_unary_weights(1, np.random.randn(num_states, d))
        model.set_unary_weights(2, np.random.randn(num_states, d))

        model.set_unary_features(0, np.random.randn(d))
        model.set_unary_features(1, np.random.randn(d))
        model.set_unary_features(2, np.random.randn(d))

        model.set_all_unary_factors()

        model.set_edge_factor((0, 1), np.zeros((num_states, num_states)))
        model.set_edge_factor((1, 2), np.zeros((num_states, num_states)))

        model.set_edge_features((0, 1), np.random.randn(d))
        model.set_edge_features((1, 2), np.random.randn(d))

        return model

