import numpy as np
from LogLinearModel import LogLinearModel
from ConvexBeliefPropagator import ConvexBeliefPropagator
import os
import pickle
from AutogradImageSegmentation import AutogradImageSegmentation


def build_diagonal_grid(height, width, num_states, d_unary, d_edge, n):
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
    saved_path = path + '/data/synthetic/diagonal_grid/'
    variable_ids = build_variable_index_diagnol_grad(height, width)
    edge_ids = build_edge_index_diagnol_grad(height, width)

    variable_feature_mat = np.random.randn(len(variable_ids), d_unary)
    edge_feature_mat = np.random.randn(len(edge_ids), d_edge)

    variable_feature_vectors = [np.array(x) for x in variable_feature_mat.tolist()]
    variable_feature_dict = dict(zip(variable_ids, variable_feature_vectors))

    f = open(saved_path + str(n) + '_unary_features.txt','w')
    pickle.dump(variable_feature_dict, f)
    f.close()

    edge_feature_vectors = [np.array(x) for x in edge_feature_mat.tolist()]
    edge_feature_dict = dict(zip(edge_ids, edge_feature_vectors))

    f = open(saved_path + str(n) + '_edge_features.txt','w')
    pickle.dump(edge_feature_dict, f)
    f.close()

    model = LogLinearModel()

    for variable, feature_vec in variable_feature_dict.items():
        model.declare_variable(variable, num_states)
        model.set_unary_features(variable, feature_vec)
        model.set_unary_factor(variable, np.zeros(num_states))

    for edge, edge_feature_vec in edge_feature_dict.items():
        model.set_edge_features(edge, edge_feature_vec)
        model.set_edge_factor(edge, np.eye(num_states))

    model.create_matrices()

    return model

def build_variable_index_diagnol_grad(height, width):
    variable_ids = []
    for i in range(0, height):
        for j in range(0, width):
            variable_ids.append((i, j))

    return variable_ids

def build_edge_index_diagnol_grad(height, width):
    edge_ids = []

    # vertical
    for i in range(0, height-1):
        for j in range(0, width):
            edge_ids.append(((i, j), (i + 1, j)))

    # horizontal
    for i in range(0, height):
        for j in range(0, width-1):
            edge_ids.append(((i, j), (i, j + 1)))

    # lower left
    for i in range(0, height-1):
        for j in range(1, width):
            edge_ids.append(((i, j), (i + 1, j - 1)))

    # lower right
    for i in range(0, height-1):
        for j in range(0, width-1):
            edge_ids.append(((i, j), (i + 1, j + 1)))

    return edge_ids


def load_diagonal_grid(num_states, n):
    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
    read_path = path + '/data/synthetic/diagonal_grid/'

    f = open(read_path + str(n) + '_label.txt','rb')
    label = pickle.load(f)

    f = open(read_path + str(n) + '_unary_features.txt','rb')
    variable_feature_dict = pickle.load(f)

    f = open(read_path + str(n) + '_edge_features.txt','rb')
    edge_feature_dict = pickle.load(f)

    model = LogLinearModel()

    for variable, feature_vec in variable_feature_dict.items():
        model.declare_variable(variable, num_states)
        model.set_unary_features(variable, feature_vec)
        model.set_unary_factor(variable, np.zeros(num_states))

    for edge, edge_feature_vec in edge_feature_dict.items():
        model.set_edge_features(edge, edge_feature_vec)
        model.set_edge_factor(edge, np.eye(num_states))

    model.create_matrices()

    return model, label


def main():
    height = 2
    width = 2
    num_states = 2
    num_models = 1
    d_unary = 65
    d_edge = 11
    max_iter = 5
    num_training_models = 1
    num_testing_models = 0

    path = os.path.abspath(os.path.join(os.path.dirname('settings.py'), os.path.pardir))
    path = path + '/data/synthetic/diagonal_grid/'

    def generate_data():
        weights = np.random.randn(d_unary * num_states + d_edge * num_states ** 2)
        np.savetxt(path + 'weights.txt', weights)

        for n in range(0, num_models):
            model = build_diagonal_grid(height, width, num_states, d_unary, d_edge, n)
            model.set_weights(weights)

            bp = ConvexBeliefPropagator(model, dict())
            bp.infer(display = 'final')

            bp.load_beliefs()

            beliefs = np.zeros((height, width))
            labels_mat = np.zeros((height, width))

            labels = dict()

            for i in range(0, height):
                for j in range(0, width):
                    beliefs[i, j] = np.exp(bp.var_beliefs[(i, j)][1])
                    labels_mat[i, j] = np.round(beliefs[i, j])
                    labels[(i, j)] = int(labels_mat[i, j])

            print labels_mat

            f = open(path + str(n) + '_label.txt', 'w')
            pickle.dump(labels, f)

    def loading_check():

        weights_file = path + 'weights.txt'
        weights = np.loadtxt(weights_file)

        for n in range(0, num_models):
            model, label = load_diagonal_grid(num_states, n)
            model.set_weights(weights)

            bp = ConvexBeliefPropagator(model, dict())
            bp.infer(display='final')

            bp.load_beliefs()

            beliefs = np.zeros((height, width))
            labels = np.zeros((height, width))

            for i in range(0, height):
                for j in range(0, width):
                    beliefs[i, j] = np.exp(bp.var_beliefs[(i, j)][1])
                    labels[i, j] = np.round(beliefs[i, j])

            print labels

    def image_segmentation():
        ais = AutogradImageSegmentation()
        ais.max_iter = max_iter
        ais.max_height = height
        ais.max_width = width
        ais.num_training_images = num_training_models
        ais.num_testing_images = num_testing_models

        for n in range(0, num_training_models):
            model, label = load_diagonal_grid(num_states, n)
            ais.models.append(model)
            ais.labels.append(label)
            ais.dimensions.append((height, width))
            ais.names.append(str(n))

        ais.set_up2()
        primal_weights = ais.learn_primal()
        ais.evaluating3(primal_weights)
        ais.learner.tau_q = None
        dual_weights = ais.learn_dual()
        ais.evaluating3(dual_weights)

    # generate_data()
    # loading_check()
    image_segmentation()

if __name__ == "__main__":
    main()