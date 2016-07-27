import numpy as np
from ImageLoader import ImageLoader
from Learner import Learner
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Evaluator import Evaluator

def main():

    d_unary = 65
    num_states = 2
    d_edge = 10
    max_height = 10
    max_width = 10
    num_training_images = 1
    num_testing_images = 0
    max_iter = 1
    inc = 'true'
    inference_type = MatrixBeliefPropagator

    loader = ImageLoader(max_height, max_width)

    images, models, labels, names = loader.load_all_images_and_labels('./tests/train', 2, num_training_images)

    learner = Learner(inference_type)

    learner.set_regularization(0.0, 0.00001)

    for model, states in zip(models, labels):
     learner.add_data(states, model)

    for bp in learner.belief_propagators_q:
     bp.set_max_iter(max_iter)
    for bp in learner.belief_propagators:
     bp.set_max_iter(max_iter)

    weights = np.zeros(d_unary * num_states + d_edge * num_states ** 2)

    new_weights = learner.learn(weights)

    unary_mat = new_weights[:d_unary * num_states].reshape((d_unary, num_states))
    pair_mat = new_weights[d_unary * num_states:].reshape((d_edge, num_states ** 2))
    print("Unary weights:\n" + repr(unary_mat))
    print("Pairwise weights:\n" + repr(pair_mat))

    if num_training_images > 0:
        print("Training:")
        Eval1 = Evaluator(max_height, max_width)
        train_errors = Eval1.evaluate_images('./tests/train', new_weights, 2, num_training_images, inference_type, max_iter, inc)
        print ("Average Train Error rate: %f" % train_errors)

    if num_testing_images > 0:
        print("Test:")
        Eval2 = Evaluator(max_height, max_width)
        test_errors = Eval2.evaluate_images('./tests/test', new_weights, 2, num_testing_images, inferency_type, max_iter)
        print ("Average Test Error rate: %f" % test_errors)


if __name__ == "__main__":
    main()