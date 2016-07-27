import unittest
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
import numpy as np
from LogLinearModel import LogLinearModel
from EM import EM
from PairedDual import PairedDual
from scipy.optimize import check_grad, approx_fprime
import matplotlib.pyplot as plt
from opt import WeightRecord
from GradientInferenceLearner import GradientInferenceLearner
from opt import ObjectivePlotter

class TestGradientInferenceLearner(unittest.TestCase):

    def set_up_learner(self, learner):
        d = 2
        num_states = 4

        np.random.seed(0)

        labels = [{0: 2,       2: 1},
                  {      1: 2, 2: 0}]

        models = []
        for i in range(len(labels)):
            m = self.create_random_model(num_states, d)
            models.append(m)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

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

    def test_message_convergence(self):
        weights = 0.05 * np.random.randn(8 + 32)
        learner = GradientInferenceLearner(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        learner.set_regularization(0.0, 1.0)
        learner.learn(weights)

        for bp in learner.belief_propagators:
            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()

            assert bp.compute_inconsistency() < 1e-2, "Belief propagator was not consistent"

    def test_agreement_with_paired_dual(self):
        weights = np.zeros(8 + 32)
        learner = GradientInferenceLearner(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        learner.set_regularization(0.0, 1.0)
        # plotter = ObjectivePlotter(learner.dual_obj)
        # plotter.interval = 0.1
        grad_weights = learner.learn(weights) #, plotter.callback)

        pd_learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(pd_learner)
        learner.set_regularization(0.0, 1.0)
        pd_weights = pd_learner.learn(weights)

        # plt.plot(pd_weights, 'r')
        # plt.plot(grad_weights, 'b')
        # plt.show()

        assert np.sum((pd_weights - grad_weights)**2) / (np.sum(pd_weights**2 + grad_weights**2)) <= 1e-2

    def test_gradient_learner(self):
        weights = np.zeros(8 + 32)
        learner = GradientInferenceLearner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]

        old_obj = learner.subgrad_obj(weight_record[0, :40])
        new_obj = learner.subgrad_obj(weight_record[-1, :40])
        assert (new_obj <= old_obj), "paired dual objective did not decrease"

        obj_record = []
        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :40])
            obj_record.append(new_obj)
            assert new_obj >= 0, "Paired dual objective was not non-negative"

        # plt.plot(obj_record)
        # plt.show()