import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
import itertools


class TestImageSegmentation(unittest.TestCase):
    data = [({1: 0, 2: 1, 3: 0, 4: -100},
             {1: np.array([1, 0, 0]), 2: np.array([0, 1, 0]), 3: np.array([1, 0, 0]), 4: np.array([0, 0, 1])},
             {1: 0, 2: 1, 3: 0, 4: 2}),
            ({1: 0, 2: 0, 3: -100, 4: 2},
             {1: np.array([1, 0, 0]), 2: np.array([1, 0, 0]), 3: np.array([0, 1, 0]), 4: np.array([0, 0, 1])},
             {1: 0, 2: 0, 3: 1, 4: 2})]

    def get_ave_accuracy(self, iters, weight_records):
        d = 3
        num_states = 3
        num_pixels = 4
        accuracy_ave_train = np.zeros(iters)
        counter = 0

        for i in range(len(self.data)):
            data = self.data[i]
            counter += 1
            accuracy = []
            for j in range(weight_records.shape[0]):
                w = weight_records[j, :]
                w_unary = np.reshape(w[0:d * num_states], (d, num_states)).T
                w_pair = np.reshape(w[num_states * d:], (num_states, num_states))
                w_unary = np.array(w_unary, dtype=float)
                w_pair = np.array(w_pair, dtype=float)
                mn = self.create_markov_net(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display="off")
                bp.compute_beliefs()
                bp.compute_pairwise_beliefs()
                bp.load_beliefs()
                prediction = []
                for pixel_id in range(1, num_pixels + 1):
                    prediction.append(np.argmax(bp.var_beliefs[pixel_id]))
                correct = 0
                for pixel_id in range(len(prediction)):
                    if data[2].values()[pixel_id] == prediction[pixel_id]:
                        correct += 1

                accuracy.append(np.true_divide(correct, num_pixels))

            accuracy_ave_train = np.add(accuracy_ave_train, accuracy)

        accuracy_ave_train = np.true_divide(accuracy_ave_train, counter)

        return accuracy_ave_train

    def get_all_edges(self, width, height):
        edges = []

        # add horizontal edge_index
        for x in range(width - 1):
            for y in range(height):
                edge = ((x, y), (x + 1, y))
                edges.append(edge)

        # add vertical edge_index
        for x in range(width):
            for y in range(height - 1):
                edge = ((x, y), (x, y + 1))
                edges.append(edge)

        return edges

    def create_markov_net(self, height, width, w_unary, w_pair, pixels):
        mn = MarkovNet()
        num_pixels = height * width
        self.get_all_edges(width, height)

        # Set Unary Factor
        for i in range(1, num_pixels + 1):
            mn.set_unary_factor(i, np.dot(w_unary, pixels[i]))

        # Set Pairwise
        south_west_ind = num_pixels - width + 1

        left_pixels = [i for i in range(1, num_pixels + 1) if (i % width) == 1 and i != south_west_ind]
        down_pixels = range(south_west_ind + 1, num_pixels + 1)
        usual_pixels = [i for i in range(1, num_pixels + 1) if i not in left_pixels and
                        i not in down_pixels and i not in [south_west_ind]]

        # set south neighbour for left pixels
        for i in left_pixels:
            mn.set_edge_factor((i, i + width), w_pair)

        # set left neighbour for sought pixels
        for i in down_pixels:
            mn.set_edge_factor((i - 1, i), w_pair)

        # set south and left neighbour for all the remaining pixels
        for i in usual_pixels:
            mn.set_edge_factor((i, i + width), w_pair)
            mn.set_edge_factor((i - 1, i), w_pair)

        return mn

    def create_model(self, num_states, d, data):
        model = LogLinearModel()

        for key in data.keys():
            model.declare_variable(key, num_states)
            model.set_unary_weights(key, np.random.randn(num_states, d))

        for key, value in data.items():
            model.set_unary_features(key, value)

        model.set_all_unary_factors()

        model.set_edge_factor((1, 2), np.zeros((num_states, num_states)))
        model.set_edge_factor((2, 4), np.zeros((num_states, num_states)))
        model.set_edge_factor((3, 4), np.zeros((num_states, num_states)))
        model.set_edge_factor((1, 3), np.zeros((num_states, num_states)))

        return model

    def set_up_learner(self, learner):
        d = 3
        num_states = 3

        models = []
        labels = []
        for i in range(len(self.data)):
            m = self.create_model(num_states, d, self.data[i][1])
            models.append(m)
            dic = self.data[i][0]
            for key in dic.keys():
                if dic[key] == -100:
                    del dic[key]
            label_vec = dic
            labels.append(label_vec)

        for model, states in zip(models, labels):
            learner.add_data(states, model)

    def test_subgradient_obj(self):
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]

        old_obj = np.Inf
        for i in range(l):
            new_obj = learner.subgrad_obj(weight_record[i, :])
            assert (new_obj <= old_obj + 1e-8), "subgradient objective did not decrease" + repr((new_obj, old_obj))
            old_obj = new_obj

    def test_EM_obj(self):
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = learner.time_record
        l = (weight_record.shape)[0]
        old_obj = learner.subgrad_obj(weight_record[0, :])
        new_obj = learner.subgrad_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "EM objective did not decrease"

    def test_paired_dual_obj(self):
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        l = (weight_record.shape)[0]

        old_obj = learner.dual_obj(weight_record[0, :])
        new_obj = learner.dual_obj(weight_record[-1, :])
        assert (new_obj <= old_obj), "paired dual objective did not decrease"

    def test_subgradient_training_accuracy(self):
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        subgrad_weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        t = wr_obj.time_record[0]
        subgrad_time = np.true_divide(np.array(time_record) - t, 1000)

        subgrad_accuracy_ave_train = self.get_ave_accuracy(len(subgrad_time), subgrad_weight_record)

        for i in range(len(subgrad_accuracy_ave_train)):
            if i != len(subgrad_accuracy_ave_train) - 1:
                assert (subgrad_accuracy_ave_train[i] <=
                        subgrad_accuracy_ave_train[i + 1]), "subgradient accuracy is not increasing"

        return subgrad_accuracy_ave_train

    def test_EM_training_accuracy(self):
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        em_weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        t = time_record[0]
        em_time = np.true_divide(np.array(time_record) - t, 1000)

        em_accuracy_ave_train = self.get_ave_accuracy(len(em_time), em_weight_record)

        for i in range(len(em_accuracy_ave_train)):
            if i != len(em_accuracy_ave_train) - 1:
                assert (em_accuracy_ave_train[i] <= em_accuracy_ave_train[i + 1]), \
                    "subgradient accuracy is not increasing"

        return em_accuracy_ave_train

    def test_paired_dual_accuracy(self):
        np.random.seed(0)
        weights = np.zeros(9 + 9)
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)

        wr_obj = WeightRecord()
        learner.learn(weights, wr_obj.callback)
        em_weight_record = wr_obj.weight_record
        time_record = wr_obj.time_record
        t = time_record[0]
        em_time = np.true_divide(np.array(time_record) - t, 1000)

        paired_dual_accuracy_ave_train = self.get_ave_accuracy(len(em_time), em_weight_record)

        for i in range(len(paired_dual_accuracy_ave_train)):
            if i != len(paired_dual_accuracy_ave_train) - 1:
                assert (paired_dual_accuracy_ave_train[i] <= paired_dual_accuracy_ave_train[
                    i + 1]), "subgradient accuracy is not increasing"

        return paired_dual_accuracy_ave_train

    def test_fixed_points(self):
        np.random.seed(0)
        # # =====================================
        # # first train by subgradient
        # # =====================================
        initial_weights = np.zeros(9 + 9)
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        subgrad_weights = learner.learn(initial_weights, None)

        learner.reset()
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        em_weights = learner.learn(subgrad_weights, None)
        # assert (np.allclose(em_weights, subgrad_weights)), "Model learned by subgrad is different from EM"

        learner.reset()
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        paired_weights = learner.learn(subgrad_weights, None)
        assert (np.allclose(paired_weights, subgrad_weights)), "Model learned by subgrad is different from paired dual"

        # =====================================
        # first train by EM
        # =====================================
        learner.reset()
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        em_weights = learner.learn(initial_weights, None)

        learner.reset()
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        subgrad_weights = learner.learn(em_weights, None)
        assert (np.allclose(em_weights, subgrad_weights, atol=0.1)), "Model learned by EM is different from subgrad"

        learner.reset()
        learner = PairedDual( MatrixBeliefPropagator)
        self.set_up_learner(learner)
        paired_weights = learner.learn(em_weights, None)
        assert (np.allclose(em_weights, paired_weights, atol=0.1)), "Model learned by EM is different from paired dual"
        # =====================================
        # first train by paired dual
        # =====================================
        learner.reset()
        learner = PairedDual(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        paired_weights = learner.learn(initial_weights, None)

        learner.reset()
        learner = EM(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        em_weights = learner.learn(paired_weights, None)
        assert (np.allclose(em_weights, paired_weights, atol=0.1)), "Model learned by paired dual is different from EM"

        learner.reset()
        learner = Learner(MatrixBeliefPropagator)
        self.set_up_learner(learner)
        subgrad_weights = learner.learn(paired_weights)
        assert (np.allclose(subgrad_weights, paired_weights, atol=0.1)), "Model learned by paired dual is different from subgrad"


if __name__ == '__main__':
    unittest.main()
