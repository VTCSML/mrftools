import unittest

import numpy as np
from MatrixInference import MatrixInference
from Model import Model


class InferenceTest(unittest.TestCase):
    def create_independent(self):
        model = Model(2, 3, 1)
        vec = np.zeros(3)
        model.add_var("A", vec)
        model.add_var("B", vec)
        model.add_var("C", vec)

        model.create_matrices()
        return model

    def create_chain_model(self):
        # create a simple model

        model = Model(4, 3, 2)

        model.add_var("A", np.random.randn(3))
        model.add_var("B", np.random.randn(3))
        model.add_var("C", np.random.randn(3))

        model.add_edge(("A", "B"), np.random.randn(2))
        model.add_edge(("C", "B"), np.random.randn(2))

        model.create_matrices()

        model.set_weights(np.random.randn(3 * 4 + 2 * 16))

        return model

    def create_loop_model(self):
        # create a simple model

        model = Model(4, 3, 2)

        model.add_var("A", np.random.randn(3))
        model.add_var("B", np.random.randn(3))
        model.add_var("C", np.random.randn(3))
        model.add_var("D", np.random.randn(3))

        model.add_edge(("A", "B"), np.random.randn(2))
        model.add_edge(("C", "B"), np.random.randn(2))
        model.add_edge(("C", "D"), np.random.randn(2))
        model.add_edge(("A", "D"), np.random.randn(2))

        model.create_matrices()

        model.set_weights(np.random.randn(3 * 4 + 2 * 16))

        return model

    def infer_chain(self):
        model = self.create_chain_model()
        inference = MatrixInference(model)
        inference.create_matrices()
        inference.set_counting_nums()

        inference.infer()

        return inference

    def test_normalization(self):
        inference = self.infer_chain()

        belief_ab = np.exp(inference.get_beliefs(("A", "B")))

        assert belief_ab is not None, "Belief was None"
        assert np.allclose(belief_ab.sum(), 1.0), "Beliefs are not normalized. " + repr(belief_ab)

        belief_a = np.exp(inference.get_beliefs(("A")))

        assert belief_a is not None, "Belief was None"
        assert np.allclose(belief_a.sum(), 1.0), "Beliefs are not normalized. " + repr(belief_a)

    def test_feasibility(self):
        inference = self.infer_chain()

        assert np.allclose(inference.norm_mat.dot(np.exp(inference.log_beliefs)), 1.0), \
            "Normalization constraint not met: " + repr(inference.norm_mat.dot(np.exp(inference.log_beliefs)))

        assert np.allclose(np.sum(np.exp(inference.get_unary_mat()), 0), 1.0), "Unary beliefs are not normalized: " + \
            repr(np.sum(np.exp(inference.get_unary_mat()), 0))
        assert np.allclose(np.sum(np.exp(inference.get_pair_tensor()), (0,1)), 1.0), "Pair beliefs are not normalized: " + \
                                                        repr(np.sum(np.exp(inference.get_pair_tensor()), (0,1)))

        assert np.allclose(inference.cons_left.dot(np.exp(inference.log_beliefs)), \
                           inference.cons_right.dot(np.exp(inference.log_beliefs))), "Calibration constraint not met"

    def test_calibration(self):
        inference = self.infer_chain()

        belief_a = np.exp(inference.get_beliefs("A", 2))

        belief_ab = np.exp(inference.get_beliefs(("A", "B"))).sum(1)[2]

        assert belief_a is not None, "Belief was None"
        assert np.allclose(belief_a, belief_ab), "Beliefs are not calibrated: %f, %f" % (belief_a, belief_ab)

    def test_unary_indexing(self):
        inference = self.infer_chain()

        belief_a1 = inference.get_beliefs("A", 1)
        belief_a = inference.get_beliefs("A")
        unary_beliefs = inference.get_unary_mat()

        a_index = inference.model.var_index["A"]

        assert belief_a1 is not None, "Belief was None"

        assert belief_a1 == belief_a[1], "Belief retrieved by passing in state was not the same as indexing in vector"

        assert belief_a1 == unary_beliefs[1, a_index], "Belief from passing in state was not the same as indexing matrix"

    def test_pair_indexing(self):
        inference = self.infer_chain()

        belief_ab01 = inference.get_beliefs(("A", "B"), (0, 1))
        belief_ab = inference.get_beliefs(("A", "B"))[0, 1]

        belief_ba10 = inference.get_beliefs(("B", "A"), (1, 0))

        pair_index = inference.model.edge_index[("A", "B")]

        belief_tensor = inference.get_pair_tensor()[0, 1, pair_index]

        assert belief_ab01 == belief_ab, "Belief retrieved by pair indexing isn't equal to by pair table"
        assert belief_ab == belief_ba10, "Belief retrieved by pair table isn't equal to reversed indexing"
        assert belief_ab == belief_tensor, "Belief retrieved by pair table isn't equal to tensor lookup"

    def test_independent_inference(self):
        model = self.create_independent()
        inference = MatrixInference(model)

        inference.set_counting_nums()

        inference.infer()

        belief_a = np.exp(inference.get_beliefs("A", 1))

        assert belief_a is not None, "Belief was None"

        belief_b = np.exp(inference.get_beliefs("B", 1))

        assert belief_a == belief_b, "Belief of identitical variables were not the same"
        assert np.allclose(belief_a, 0.5), "Belief of variable was not 0.5. Instead it was %f" % belief_a

    def test_loop_consistency(self):
        model = self.create_loop_model()
        inference = MatrixInference(model)
        inference.set_counting_nums()
        inference.infer()

        belief_ab01 = inference.get_beliefs(("A", "B"), (0, 1))
        belief_ab = inference.get_beliefs(("A", "B"))[0, 1]

        belief_ba10 = inference.get_beliefs(("B", "A"), (1, 0))

        pair_index = inference.model.edge_index[("A", "B")]

        belief_tensor = inference.get_pair_tensor()[0, 1, pair_index]

        assert belief_ab01 == belief_ab, "Belief retrieved by pair indexing isn't equal to by pair table"
        assert belief_ab == belief_ba10, "Belief retrieved by pair table isn't equal to reversed indexing"
        assert belief_ab == belief_tensor, "Belief retrieved by pair table isn't equal to tensor lookup"


if __name__ == '__main__':
    unittest.main()
