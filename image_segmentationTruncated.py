#!/usr/bin/env python3
#import numpy as np
import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad
from autograd.util import quick_grad_check
import matplotlib.image
import matplotlib.pyplot
import math
import os
from shutil import copyfile
from io import StringIO
import  io
# import pandas as pd
from sklearn import linear_model
import time
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import MarkovNet
from MarkovNet import MarkovNet
from LogLinearModel import LogLinearModel
from scipy.optimize import minimize, check_grad
from TemplatedLogLinearMLETruncated import TemplatedLogLinearMLETruncated
import PIL
from PIL import Image
import time
from functionsHorse import *
from autograd.util import quick_grad_check
from BeliefPropagator import BeliefPropagator



def main():

    # =====================================
    # Create Model
    # =====================================


    height = 10
    width = 10
    num_pixels = height * width
    d = 3
    num_states = 8
    model = Create_LogLinearModel(height,width,d, num_states)
    learner = TemplatedLogLinearMLETruncated(model, 3)

    # =====================================
    # Load images and resize them
    # =====================================

    train_path = "./train/"
    for file in os.listdir(train_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            
            small_pix = Load_Resize_Image(train_path+file,height,width).load()
            lbl_small = Load_Resize_Label(train_path+file[:-4]+"_label.txt",height,width)
            
            lbl = []
            lbl = np.array(lbl)
            k = 1
            dic1 = {}
            dic2 = {}
            for i in range(0,height):
                for j in range(0,width):
                    dic1[k] = lbl_small[i,j]
                    dic2[k] = np.array(small_pix[j,i])
                    k += 1
            learner.addData(dic1,dic2)
            print ('data is added')


    # =====================================
    # Optimization
    # =====================================

  
    # add node weights
    weights = np.zeros( num_states * d)
    #weights = np.zeros(num_states * num_pixels * d)
    # add edge weights
    weights = np.append(weights, np.zeros( num_states * num_states))

    learner.setRegularization(0, 1) # gradient checking doesn't work well with the l1 regularizer
    training_loss_and_grad = value_and_grad(learner.objective)
    objective = learner.objective(weights)
    print ("Objective:")
    print (objective)

    # print "Gradient check:"
    # print check_grad(learner.objective, learner.gradient, weights)

    def printObj(weights):
        print("Objective: %f" % (learner.objective(weights)))

    def printObjAndGradCheck(weights):
        print("Objective: %f" % (learner.objective(weights)))
        quick_grad_check(learner.objective, weights)

    # Check the gradients numerically, just to be safe
    quick_grad_check(learner.objective, weights)


    print ("Optimization:")
    # res = minimize(training_loss_and_grad, weights, method='L-BFGS-b', jac = True, printObj = printObj)
    # res = minimize(learner.objective, weights, method='L-BFGS-b', jac = gradient, printObj = printObj)
    res = minimize(learner.objective, weights, method='L-BFGS-b', jac = learner.gradient, callback = printObj, options = {'maxiter': 10})

    weights = res.x
    quick_grad_check(learner.objective, weights)
    res = minimize(training_loss_and_grad, weights, method='L-BFGS-b', jac = True, callback = printObjAndGradCheck)
    print res

    f = open('weights.txt','w')
    for item in res.x:
        f.write(str(item)+',')

    # =====================================
    # Inference
    # =====================================


    w = []
    w = res.x
    w_unary = np.reshape(w[0:d * num_states],(num_states,d))
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)

    test_path = "./test/"
    for file in os.listdir(test_path):
        if file.endswith(".jpg") or file.endswith(".png"):

            pixels = Load_Resize_Image(test_path+file,height,width).load()
            mn = Create_MarkovNet(height,width,w_unary,w_pair,pixels)

            bp = BeliefPropagator(mn)
            bp.runInference(display = 'iter', maxIter = 3)
            bp.computePairwiseBeliefs()

            print ('done inference---------------------')


            Z = []
            for i in range(1,num_pixels+1):
                Z.append(np.argmax(bp.varBeliefs[i]))

            Z1 = np.reshape(Z,(height,width))


            #calculate accuracy
            acc = Accuracy(Z1,test_path+file,height,width)
            print("accuracy:")
            print(acc)
            
            # Plot segmented image
            # Plot_Segmented(Z1,test_path+file,height,width)




if __name__ == "__main__":
    main()
