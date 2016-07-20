import unittest
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
import numpy as np
from LogLinearModel import LogLinearModel
from EM import EM
from PairedDual import PairedDual

class TestLearner(unittest.TestCase):

    def set_up_learner(self, learner):
        d = 2
        num_states = 4

        np.random.seed(0)

        data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
                ({0: -100, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

        models = []
        labels = []
        for i in range(len(data)):
            m = self.create_model(num_states,d)
            models.append(m)
            dic = data[i][0]
            label_vec = dic
            labels.append(label_vec)
            # for keys,values in dic.items():
            #     print values

        for model, states in zip(models, labels):
            learner.add_data(states, model)


    def test_learner(self):
        weights = np.zeros(24)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        learner.learn(weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            new_obj = learner.subgrad_obj(learner.weight_record[i,:])
            assert (new_obj <= old_obj), "subgradient objective is not decreasing"
            old_obj = new_obj

    def test_EM(self):
        weights = np.zeros(24)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        learner.learn(weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        t = learner.time_record[0]
        old_obj = learner.subgrad_obj(learner.weight_record[0,:])
        new_obj = learner.subgrad_obj(learner.weight_record[-1,:])
        assert (new_obj <= old_obj), "EM objective did not decrease"
    def test_paired_dual(self):
        weights = np.zeros(24)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        learner.learn(weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        t = learner.time_record[0]

        old_obj = learner.dual_obj(learner.weight_record[0,:])
        new_obj = learner.dual_obj(learner.weight_record[-1,:])
        assert (new_obj <= old_obj), "paired dual objective did not decrease"



    def create_model(self,num_states,d):
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

        return  model

