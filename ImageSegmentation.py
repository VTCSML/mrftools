import numpy as np
from ImageLoader import ImageLoader
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Evaluation import Evaluation

def main():

     loader = ImageLoader()

     images, models, labels, names = loader.load_all_images_and_labels('./tests/train', 2)

     learner = Learner(MatrixBeliefPropagator)

     learner.set_regularization(0.0, 0.00001)

     for model, states in zip(models, labels):
         learner.add_data(states, model)

     d_unary = 65
     num_states = 2
     d_edge = 10

     weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

     new_weights = learner.learn(weights)

     unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
     pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states ** 2))
     print("Unary weights:\n" + repr(unary_mat))
     print("Pairwise weights:\n" + repr(pair_mat))

     print("Training:")
     Eval1 = Evaluation()
     train_errors = Eval1.evaluation_images('./tests/train', new_weights, 2)
     print ("Average Train Error rate: %f" % train_errors)

     print("Test:")
     Eval2 = Evaluation()
     test_errors = Eval2.evaluation_images('./tests/test', new_weights, 2)
     print ("Average Test Error rate: %f" % test_errors)


if __name__ == "__main__":
    main()