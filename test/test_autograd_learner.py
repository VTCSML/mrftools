import unittest
from mrftools import *
import numpy as np
from scipy.optimize import check_grad
import time
try:
    from autograd import grad

    class TestAutogradLearner(unittest.TestCase):
        def test_gradient(self):
            n = 8
            k = 2
            d = 4
            learner = self.set_up_grid_learner(n, k, d)
            learner.set_inference_truncation(5)
            weights = np.random.randn(d * k + k * k * d)

            gradient_error = check_grad(learner.subgrad_obj, grad(learner.subgrad_obj), weights)
            print("Gradient error: %x" % gradient_error)
            assert gradient_error < 1e-1, "Gradient is wrong"

        def test_dual_gradient(self):
            n = 8
            k = 2
            d = 4
            learner = self.set_up_grid_learner(n, k, d)
            learner.set_inference_truncation(5)
            weights = np.random.randn(d * k + k * k * d)

            gradient_error = check_grad(learner.dual_obj, grad(learner.dual_obj), weights)
            print("Gradient error: %x" % gradient_error)
            assert gradient_error < 1e-1, "Gradient is wrong"

        def test_learner(self):
            n = 8
            k = 2
            d = 4
            learner = self.set_up_grid_learner(n, k, d)
            learner.set_inference_truncation(5)
            weights = np.random.randn(d * k + k * k * d)

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
            n = 8
            k = 2
            d = 4
            learner = self.set_up_grid_learner(n, k, d)
            learner.set_inference_truncation(5)
            weights = np.random.randn(d * k + k * k * d)

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

        def test_learner_speed(self):
            n = 20
            d = 64
            k = 8

            learner = self.set_up_grid_learner(n, k, d)
            weights = np.random.randn(d * k + k * k * d)

            learner.set_inference_truncation(10)

            start = time.time()
            obj = learner.subgrad_grad(weights)
            obj_time = time.time() - start
            print("Computing the objective took %f seconds" % obj_time)

            start = time.time()
            gradient = learner.subgrad_grad(weights)
            grad_time = time.time() - start
            print("Computing the standard gradient took %f seconds" % grad_time)

            grad_fun = grad(learner.subgrad_obj)

            start = time.time()
            gradient = grad_fun(weights)
            backprop_time = time.time() - start
            print("Computing the back-propagated gradient took %f seconds" % backprop_time)

        def set_up_grid_learner(self, n, k, d):
            model, labels = self.create_grid_model(n, k, d)

            learner = AutogradLearner(ConvexBeliefPropagator)
            learner.add_data(labels, model)
            return learner

        def create_grid_model(self, length, num_states, feature_dim):
            model = LogLinearModel()

            for x in range(length):
                for y in range(length):
                    model.declare_variable((x, y), np.random.randn(num_states))
                    model.set_unary_features((x, y), np.random.randn(feature_dim))
                    model.set_unary_factor((x, y), np.zeros(num_states))

            for x in range(length - 1):
                for y in range(length):
                    model.set_edge_factor(((x, y), (x + 1, y)), np.random.randn(num_states, num_states))
                    model.set_edge_factor(((y, x), (y, x + 1)), np.random.randn(num_states, num_states))

                    model.set_edge_features(((x, y), (x + 1, y)), np.random.randn(feature_dim))
                    model.set_edge_features(((y, x), (y, x + 1)), np.random.randn(feature_dim))

            model.create_matrices()

            unary_weights = np.random.randn(feature_dim, num_states)
            pairwise_weights = np.random.randn(feature_dim, num_states, num_states)
            pairwise_weights = 0.5 * (pairwise_weights + np.transpose(pairwise_weights, [0, 2, 1]))
            weights = np.concatenate((unary_weights.ravel(), pairwise_weights.ravel()))

            model.set_weights(weights)

            bp = ConvexBeliefPropagator(model)
            bp.set_max_iter(50)
            bp.infer(display='off')
            bp.load_beliefs()

            labels = dict()
            for (var, beliefs) in bp.var_beliefs.items():
                if np.random.rand(1) < 0.9:
                    labels[var] = np.argmax(beliefs)
                # else:
                #     print("Leaving %s latent" % repr(var))

            return model, labels

except ImportError:
    print "Autograd could not be imported. Skipping tests for AutogradLearner"
