import unittest
from mrftools import *
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import itertools
from mrftools import save_load_weights
import skimage.color
import skimage.util
from skimage.io import imsave

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def batch_load_images(dir, size, num_class):
    #dir = "/Users/youlu/Documents/workspace/mrftools/tests/train_data/"
    IL = ImageLoader(max_width=size, max_height=size)
    images, models, labels, names = IL.load_all_images_and_labels(dir, num_class, num_images=np.inf)
    return images, models, labels, names

def batch_load_images_features(dir, size, num_class):
    #dir = "/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse"
    IFL = ImageFeatureLoader(max_width=size, max_height=size)
    models, labels, names = IFL.load_all_features_labels(dir, "dataset", num_class)
    return models, labels, names

def train_model(models, labels, num_class, inference_type):
    plt.clf()
    learner = Learner(inference_type)
    num_states = num_class
    d_edge = models[0].num_edge_features.values()[0]
    d_unary = len(models[0].unary_features[(0,0)])
    d_weights = d_unary * num_states + d_edge * np.power(num_states, 2)
    initial_weights = np.zeros(d_weights)
    for model, label in zip(models, labels):
        learner.add_data(label, model)

    #weights = learner.learn(initial_weights, callback=None)

    plotter = ObjectivePlotter(func=learner.objective)
    weights = learner.learn(initial_weights, callback=plotter.callback)
    filename = "ConvexMBP_S%d.jpg"%size
    plt.savefig("/Users/youlu/Documents/workspace/mrftools/tests/test_results/%s"%filename)
    return weights

def predict_labels(weights, models, inference_type, size):
    predictions = list()
    for i in range(0, len(models)):
        models[i].set_weights(weights)
        bp = inference_type(models[i])
        bp.infer(display='off')
        bp.load_beliefs()
        prediction = np.zeros((size, size), dtype = np.int)
        for x in range(0, size):
            for y in range(0, size):
                prediction[x,y] = np.argmax(np.exp(bp.var_beliefs[(x,y)]))
        predictions.append(prediction)
    return predictions

def process_labels(labels, size):
    processed_labels = list()
    for label in labels:
        processed_label = np.zeros((size, size), dtype = np.int)
        for i in range(0,size):
            for j in range(0, size):
                processed_label[i,j] = label[(i,j)]
        processed_labels.append(processed_label)
    return processed_labels

def old_features_train(dir, output_dir, size, num_class, output_name, inference_type):
    images, models, labels, names = batch_load_images(dir, size, num_class)
    weights = train_model(models, labels, num_class, inference_type)
    output_path = osp.join(output_dir, "%s.txt"%output_name)
    save_load_weights.save_weights(weights, output_path)
    return weights

def old_features_evaluate(dir, output_dir, weights, size, num_class, output_name, inference_type):
    images, models, labels, names = batch_load_images(dir, size, num_class)
    predictions = predict_labels(weights, models, inference_type, size)
    processed_labels = process_labels(labels, size)
    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(processed_labels, predictions, num_class)
    with open(osp.join(output_dir, '%s.txt'%output_name), 'w') as f:
        f.write("acc: " + "%s"%acc + "\n")
        f.write("acc_cls: " + "%s"%acc_cls + "\n")
        f.write("mean_iu: " + "%s"%mean_iu + "\n")
        f.write("fwavacc: " + "%s"%fwavacc)

    viz_output_dir = osp.join(output_dir, "visualization")
    visualize_labels_predictions(processed_labels, predictions, viz_output_dir, names)

def visualize_labels_predictions(labels, predictions, output_dir, names):
    cmap = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cmap = cmap.astype(np.float32)
    for i in range(0, len(labels)):
        label = labels[i].T
        prediction = predictions[i].T
        name = names[i]
        output_path = osp.join(output_dir, "%s"%name)
        viz_label = skimage.color.label2rgb(label, colors=cmap[1:], bg_label=0)
        viz_pred = skimage.color.label2rgb(prediction, colors=cmap[1:], bg_label=0)
        out = np.concatenate((viz_label,viz_pred),axis=1)
        imsave(output_path, out)


def old_features_main(train_dir, test_dir, output_dir, size, num_class, inference_type, output_weights_name, output_evaluation_name):
    old_features_train(train_dir, output_dir, size, num_class, output_weights_name, inference_type)
    weights = save_load_weights.load_weights(osp.join(output_dir, "%s.txt"%output_weights_name))
    old_features_evaluate(test_dir, output_dir, weights, size, num_class, output_evaluation_name, inference_type)








if __name__ == '__main__':
    train_dir = "/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse_test/train_data/"
    test_dir = "/Users/youlu/Documents/PycharmProjects/fcn_8s_pytorch/data/horse_test/test_data/"
    output_dir = "/Users/youlu/Documents/workspace/mrftools/tests/test_results/horse_test/"
    size = 100
    num_class = 2
    inference_type = ConvexBeliefPropagator
    output_weights_name = "Convex_weights"
    output_evaluation_name = "Convex_test_results"
    old_features_main(train_dir, test_dir, output_dir, size, num_class, inference_type, output_weights_name, output_evaluation_name)













