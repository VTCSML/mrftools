import unittest
from mrftools import *
import numpy as np
from scipy.optimize import check_grad
try:
    from autograd import grad

    class TestAutogradLearner(unittest.TestCase):

        def set_up_learner(self, learner):
            d = 2
            num_states = 4

            np.random.seed(0)

            labels = [{0: 2,       2: 1},
                      {      1: 2, 2: 0},
                      {0: 2, 1: 3,     },
                      {0: 0, 1: 2, 2: 3}]

            models = []
            for i in range(len(labels)):
                m = self.create_random_model(num_states, d)
                models.append(m)

            for model, states in zip(models, labels):
                learner.add_data(states, model)

        def test_gradient(self):
            weights = np.zeros(8 + 32)
            learner = AutogradLearner(MatrixBeliefPropagator)
            self.set_up_learner(learner)
            learner.set_regularization(0.0, 1.0)
            gradient_error = check_grad(learner.subgrad_obj, grad(learner.subgrad_obj), weights)
            print("Gradient error: %f" % gradient_error)
            assert gradient_error < 1e-1, "Gradient is wrong"

        def test_learner(self):
            weights = np.zeros(8 + 32)
            learner = AutogradLearner(MatrixBeliefPropagator)
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

        def test_learner_dual(self):
            weights = np.zeros(8 + 32)
            learner = AutogradLearner(MatrixBeliefPropagator)
            self.set_up_learner(learner)

            wr_obj = WeightRecord()
            learner.learn_dual(weights,wr_obj.callback)
            weight_record = wr_obj.weight_record
            time_record = wr_obj.time_record
            l = (weight_record.shape)[0]
            old_obj = np.Inf
            for i in range(l):
                new_obj = learner.subgrad_obj_dual(weight_record[i,:])
                assert (new_obj <= old_obj + 1e-8), "subgradient objective is not decreasing"
                old_obj = new_obj

                assert new_obj >= 0, "Learner objective was not non-negative"

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

except ImportError:
    pass
