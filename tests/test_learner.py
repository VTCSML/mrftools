import unittest
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
import numpy as np
from LogLinearModel import LogLinearModel
from EM import EM
from PairedDual import PairedDual

class TestLearner(unittest.TestCase):

    def test_learner(self):
        d = 2
        num_states = 4

        data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
                ({0: -100, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

        learner = Learner(MatrixBeliefPropagator)

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

        weights = np.zeros(24)

        learner.learn(weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        subgrad_obj_list = []
        subgrad_time_list = []
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            subgrad_time_list.append(time_record[i] - t)
            new_obj = learner.subgrad_obj(learner.weight_record[i,:], 'subgradient')
            subgrad_obj_list.append(new_obj)
            assert (new_obj <= old_obj), "subgradient objective is not decreasing"
            old_obj = new_obj

    def test_EM(self):
        d = 2
        num_states = 4

        data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
                ({0: -100, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

        learner = EM(MatrixBeliefPropagator)

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

        weights = np.zeros(24)

        learner.learn(weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        EM_obj_list = []
        EM_time_list = []
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            EM_time_list.append(time_record[i] - t)
            new_obj = learner.subgrad_obj(learner.weight_record[i,:], 'EM')
            EM_obj_list.append(new_obj)
            assert (new_obj <= old_obj), "EM objective is not decreasing"
            old_obj = new_obj



    def test_paired_dual(self):
        d = 2
        num_states = 4

        data = [({0: 2, 1: -100, 2: 1}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)}),
                ({0: -100, 1: 2, 2: 0}, {0: np.random.randn(d), 1: np.random.randn(d), 2: np.random.randn(d)})]

        learner = PairedDual(MatrixBeliefPropagator)

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

        weights = np.zeros(24)
        learner.learn(weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        paired_obj_list = []
        paired_time_list = []
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            paired_time_list.append(time_record[i] - t)
            new_obj = learner.subgrad_obj(learner.weight_record[i,:], 'subgradient')
            paired_obj_list.append(new_obj)
            assert (new_obj <= old_obj), "paired dual objective is not decreasing"
            old_obj = new_obj



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

