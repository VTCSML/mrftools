import unittest


from matinference import Model
import numpy as np


class ModelTest(unittest.TestCase):
    @staticmethod
    def create_chain_model():
        # create a simple model

        model = Model(4, 3, 2)

        model.add_var("A", np.random.randn(3))
        model.add_var("B", np.random.randn(3))
        model.add_var("C", np.random.randn(3))

        model.add_edge(("A", "B"), np.random.randn(2))
        model.add_edge(("C", "B"), np.random.randn(2))

        model.create_matrices()

        return model

    def test_indexing(self):
        model = self.create_chain_model()
        assert model.var_index["B"] == model.vars.index("B")

    def test_var_validation(self):
        model = self.create_chain_model()

        with self.assertRaises(AssertionError):
            model.add_var("A")

        with self.assertRaises(AssertionError):
            model.add_var("D", np.zeros(100))

    def test_edge_validation(self):
        model = self.create_chain_model()

        with self.assertRaises(AssertionError):
            model.add_edge(("B", "C"))

        with self.assertRaises(AssertionError):
            model.add_edge(("A", "C"), np.zeros(100))

    def test_weights(self):
        model = self.create_chain_model()

        model.set_weights(np.random.randn(3 * 4 + 2 * 16))

    def test_matrices(self):
        model = self.create_chain_model()

        model.set_weights(np.random.randn(3 * 4 + 2 * 16))

        phi = model.template_mat.T.dot(model.weights).T

        assert phi.size == 4 * 3 + 16 * 2

if __name__ == '__main__':
    unittest.main()
