#!/usr/bin/env python2.7
#import numpy as np
import autograd.numpy as np
from autograd import value_and_grad
from autograd import grad
from autograd.util import *
import matplotlib.image
import matplotlib.pyplot
import math
import os
from shutil import copyfile
from io import StringIO
import  io
from sklearn import linear_model
import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from MarkovNet import MarkovNet
from LogLinearModel import LogLinearModel
from scipy.optimize import minimize, check_grad
from MatrixLogLinearMLETruncated import MatrixLogLinearMLETruncated
import PIL
from PIL import Image
import time
from functionsHorse import *
from sklearn.metrics import roc_curve, auc
from skimage.feature import hog
from BruteForce import BruteForce
from BeliefPropagator import BeliefPropagator
from MatrixBeliefPropagatorTruncated import MatrixBeliefPropagatorTruncated



def main():

    # =====================================
    # Create Model
    # =====================================

    height = 8
    width = 8
    num_pixels = height * width
    d = 64
    num_states = 2
    bp_iter = 3
    model = Create_LogLinearModel(height, width, d, num_states)
    learner = MatrixLogLinearMLETruncated(model, bp_iter)

    # =====================================
    # Load images and resize them
    # =====================================

    train_path = "./train/"
    num_pictures = 0.0
    for file in os.listdir(train_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            num_pictures = num_pictures + 1
            small_pix = Load_Resize_Image(train_path+file, height, width).load()
            lbl_small = Load_Resize_Label(train_path+file[:-4]+"_label.txt", height, width)
            lbl = []
            lbl = np.array(lbl)
            k = 1
            dic1 = {}
            dic2 = {}
            for i in range(0, height):
                for j in range(0, width):
                    dic1[k] = lbl_small[i, j]
                    dic2[k] = get_augmented_pixels(small_pix[j, i], i, j, height, width)
                    k += 1
            learner.addData(dic1, dic2)
            print ('%d th data is added' %(num_pictures))

    # =====================================
    # Learning
    # =====================================


    print ("-------------Learning with primal objective function-----------")


    # add node weights
    weights = np.ones(num_states * d)
    # add edge weights
    weights = np.append(weights, np.ones(num_states * num_states))
    learner.setRegularization(0, 1)

    print "\n\nGradient check:"
    print check_grad(learner.objective, learner.gradient, weights)

    print "\n\nOptimization:"
    res = minimize(learner.objective, weights, method='L-BFGS-b', jac=learner.gradient, options={'gtol': 1e-4, 'maxiter': 10})

    print "\n\nGradient check:"
    print check_grad(learner.objective, learner.gradient, res.x * (1 + 1e-9))

    weights = res.x
    print("objective")
    print(learner.dualObjective(weights))

    training_loss_and_grad = value_and_grad(learner.objective)
    res = minimize(training_loss_and_grad, weights, method='L-BFGS-b', jac=True, options={'gtol': 1e-4})

    print "\n\nGradient check at optimized solution:"
    print check_grad(learner.objective, learner.gradient, res.x * (1 + 1e-9))


    # =====================================
    # Inference
    # =====================================

    w = res.x
    w_unary = np.reshape(w[0:d * num_states],(num_states,d))
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)

    # set total training accuracy and total training inconsistency
    Ttinc = 0.0
    num_pictures = 0.0
    score_all = []

    for file in os.listdir(train_path):
        if file.endswith(".jpg") or file.endswith(".png"):

            num_pictures = num_pictures + 1
            pixels = Load_Resize_Image(train_path+file,height,width).load()
            mn = Create_MarkovNet(height,width,w_unary,w_pair,pixels)
            bp = MatrixBeliefPropagatorTruncated(mn)
            bp.runInference(display = 'iter', maxIter = 3)
            bp.runInference(display='iter')
            bp.computePairwiseBeliefs()
            bp.load_beliefs()
            bp.initialize_messages()

            print ('done inference---------------------')

            score = []

            for i in range(1,num_pixels+1):
                score_all.append(np.exp(bp.varBeliefs[i][1]))
                score.append(np.exp(bp.varBeliefs[i][1]))

            Score = np.reshape(score,(height, width))

            #calculate inconsistency
            inc = bp.computeInconsistency()
            print("inconsistency")
            print(inc)
            Ttinc = Ttinc + inc

            # Plot segmented image
            Plot_Segmented(Score, train_path + file, height, width)

    print("inconsistency :")
    print(Ttinc / num_pictures)


if __name__ == "__main__":
    main()