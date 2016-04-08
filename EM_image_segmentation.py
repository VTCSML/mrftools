 #!/usr/bin/env python3
import numpy as np
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
from BeliefPropagator import BeliefPropagator
from LogLinearModel import LogLinearModel
from LogLinearMLE import LogLinearMLE
from scipy.optimize import minimize, check_grad
from TemplatedLogLinearMLE import TemplatedLogLinearMLE
import PIL
from PIL import Image
import time
from functions import *
from TemplatedLogLinearMLE_EM import *



def main():

    # =====================================
    # Create Model
    # =====================================


    height = 8
    width = 8
    num_pixels = height * width
    d = 3
    
    num_states = 8
    # add node weights
    weights = np.random.randn( num_states * d)
    # add edge weights
    weights = np.append(weights, np.random.randn( num_states * num_states))



    model = Create_LogLinearModel(height,width,d)
    learner = TemplatedLogLinearMLE_EM(model)

    train_path = "./train/"
    for file in os.listdir(train_path):
        if file.endswith(".jpg"):
            

            small_pix = Load_Resize_Image(train_path+file,height,width)
            lbl_small = Load_Resize_Label(train_path+file[:-4]+"_label.txt",height,width)
            
            for k in range(height):
                mmn =  np.random.randint(width ,size = 1)
                lbl_small[k,np.int(mmn)] = -100
            
            #
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
            
            
           
           
    maxIter = 10
    for i in range(maxIter):
        print 'step: ' + str(i)
        # =====================================
        # E-step: inference
        # =====================================
        learner.E_step(weights)


        # =====================================
        # M-step: learning parameters
        # =====================================
        weights = learner.M_step(weights)
        print weights





# =====================================
# Testing
# =====================================
    w = weights
    w_unary = np.reshape(w[0:d * num_states],(num_states,d))
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)


    test_path = "./test/"
    for file in os.listdir(test_path):
        if file.endswith('.jpg'):
            pixels = Load_Resize_Image(test_path+file,height,width)
            mn = Create_MarkovNet(height,width,w_unary,w_pair,pixels)

            bp = BeliefPropagator(mn)
            bp.runInference()
            bp.computePairwiseBeliefs()


            Z = []
            for i in range(1,num_pixels+1):
                Z.append(np.argmax(bp.varBeliefs[i]))

            Z1 = np.reshape(Z,(height,width))

            # Plot segmented image
            Plot_Segmented(Z1,test_path+file)

            


if __name__ == "__main__":
    main()
