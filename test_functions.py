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
# from functions import *
from TemplatedLogLinearMLE_EM import *
import time
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
import datetime
import pickle
import itertools
from pandas.compat import lrange
import colorsys
from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image
import matplotlib.pyplot as plt
from MatrixTemplatedLogLinearMLE_EM import MatrixTemplatedLogLinearMLE_EM
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from EM import EM



train_data_dic = {}
test_data_dic = {}

def add_test_data():
    dic1 = {}
    dic1[1] = 1
    dic1[2] = 1
    dic1[3] = 0
    dic1[4] = 0
    dic2 = {}
    dic2[1] = np.array([0,1,0])
    dic2[2] = np.array([0,1,0])
    dic2[3] = np.array([1,0,0])
    dic2[4] = np.array([1,0,0])
    test_data_dic[0] = [dic1,dic2]
    
    
    dic1 = {}
    dic1[1] = 0
    dic1[2] = 1
    dic1[3] = 1
    dic1[4] = 0
    dic2 = {}
    dic2[1] = np.array([1,0,0])
    dic2[2] = np.array([0,1,0])
    dic2[3] = np.array([0,1,0])
    dic2[4] = np.array([1,0,0])
    test_data_dic[1] = [dic1,dic2]
    
    
    dic1 = {}
    dic1[1] = 1
    dic1[2] = 1
    dic1[3] = 2
    dic1[4] = 2
    dic2 = {}
    dic2[1] = np.array([0,1,0])
    dic2[2] = np.array([0,1,0])
    dic2[3] = np.array([0,0,1])
    dic2[4] = np.array([0,0,1])
    test_data_dic[2] = [dic1,dic2]


def add_synthetic_data(learner):
    dic1 = {}
    dic1[1] = 0
    dic1[2] = 1
    dic1[3] = 0
    dic1[4] = -100
    dic2 = {}
    dic2[1] = np.array([1,0,0])
    dic2[2] = np.array([0,1,0])
    dic2[3] = np.array([1,0,0])
    dic2[4] = np.array([0,0,1])
    dic3 = {}
    dic3[1] = int(0)
    dic3[2] = int(1)
    dic3[3] = int(0)
    dic3[4] = int(2)
    learner.add_data(dic1,dic2)
    train_data_dic[0] = [dic1,dic2,dic3]
#     print ('data is added')
    dic1 = {}
    dic1[1] = 0
    dic1[2] = 0
    dic1[3] = -100
    dic1[4] = 2
    dic2 = {}
    dic2[1] = np.array([1,0,0])
    dic2[2] = np.array([1,0,0])
    dic2[3] = np.array([0,1,0])
    dic2[4] = np.array([0,0,1])
    learner.add_data(dic1,dic2)
    dic3 = {}
    dic3[1] = 0
    dic3[2] = 0
    dic3[3] = 1
    dic3[4] = 2
    train_data_dic[1] = [dic1,dic2,dic3]
#     print ('data is added')
         
    dic1 = {}
    dic1[1] = -100
    dic1[2] = 2
    dic1[3] = 1
    dic1[4] = 2
    dic2 = {}
    dic2[1] = np.array([1,0,0])
    dic2[2] = np.array([0,0,1])
    dic2[3] = np.array([0,1,0])
    dic2[4] = np.array([0,0,1])
    learner.add_data(dic1,dic2)
#     print ('data is added')
    dic3 = {}
    dic3[1] = 0
    dic3[2] = 2
    dic3[3] = 1
    dic3[4] = 2
    train_data_dic[2] = [dic1,dic2,dic3]
         
    dic1 = {}
    dic1[1] = 1
    dic1[2] = -100
    dic1[3] = 0
    dic1[4] = 2
    dic2 = {}
    dic2[1] = np.array([0,1,0])
    dic2[2] = np.array([0,1,0])
    dic2[3] = np.array([1,0,0])
    dic2[4] = np.array([0,0,1])
    learner.add_data(dic1,dic2)
    dic3 = {}
    dic3[1] = 1
    dic3[2] = 1
    dic3[3] = 0
    dic3[4] = 2
    train_data_dic[3] = [dic1,dic2,dic3]
#     print ('data is added')
    learner.setRegularization(0, 0.25)



# =====================================
# Create Log linear model
# =====================================


def Create_LogLinearModel(height,width,d,numStates):
    num_pixels = height * width
    model = LogLinearModel()

    for i in range(1,num_pixels+1):
        model.declareVariable(i, numStates)
        model.setUnaryWeights(i,np.random.randn(numStates, d))
        model.setUnaryFeatures(i, np.random.randn(d))


    model.setAllUnaryFactors()
#    print ('unary factors are done')

#     ########### Set Edge Factor
    #
    left_ind = num_pixels - width + 1
      
    left_pixels = [i for i in range(1,num_pixels+1) if (i % width) == 1 and i != 1 and i != left_ind]
    right_pixels = [i for i in range(1,num_pixels+1) if (i % width) == 0 and i != width and i != num_pixels]
    up_pixels = range(2,width)
    down_pixels = range(left_ind+1 , num_pixels)
    usual_pixels = [i for i in range(1,num_pixels+1) if i not in left_pixels and i not in right_pixels and i not in up_pixels and i not in down_pixels and i not in (1,width,left_ind,num_pixels)]
      
      
    up_pixels = set(up_pixels)
    down_pixels = set(down_pixels)
    usual_pixels = set(usual_pixels)
    left_pixels = set(left_pixels)
    right_pixels = set(right_pixels)
     
     
    all_edges = set()
        
    model.setEdgeFactor((1,2),np.eye(numStates))
    all_edges.add((1,2))
    model.setEdgeFactor((1,1+width),np.eye(numStates))
    all_edges.add((1,1+width))
        
        
    model.setEdgeFactor((width,width-1),np.eye(numStates))
    all_edges.add((width,width-1))
    model.setEdgeFactor((width,width+width),np.eye(numStates))
    all_edges.add((width,width+width))
        
        
    model.setEdgeFactor((left_ind,left_ind +1),np.eye(numStates))
    all_edges.add((left_ind,left_ind +1))
    model.setEdgeFactor((left_ind,left_ind - width),np.eye(numStates))
    all_edges.add((left_ind,left_ind - width))
        
        
    model.setEdgeFactor((num_pixels,num_pixels - 1),np.eye(numStates))
    all_edges.add((num_pixels,num_pixels - 1))
    model.setEdgeFactor((num_pixels,num_pixels - width),np.eye(numStates))
    all_edges.add((num_pixels,num_pixels - width))
        
        
    for i in (left_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges :
            model.setEdgeFactor((i,i+1),np.eye(numStates))
            all_edges.add((i,i+1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.setEdgeFactor((i,i-width),np.eye(numStates))
            all_edges.add((i,i-width))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.setEdgeFactor((i,i+width),np.eye(numStates))
            all_edges.add((i,i+width))
        
        
    for i in (right_pixels):
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.setEdgeFactor((i,i-1),np.eye(numStates))
            all_edges.add((i,i-1))
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.setEdgeFactor((i,i-width),np.eye(numStates))
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.setEdgeFactor((i,i+width),np.eye(numStates))
            all_edges.add((i,i+width))
        
        
    for i in  up_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            model.setEdgeFactor((i,i+1),np.eye(numStates))
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.setEdgeFactor((i,i-1),np.eye(numStates))
            all_edges.add((i,i-1))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.setEdgeFactor((i,i+width),np.eye(numStates))
            all_edges.add((i,i+width))
        
        
        
    for i in  down_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            model.setEdgeFactor((i,i+1),np.eye(numStates))
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.setEdgeFactor((i,i-1),np.eye(numStates))
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.setEdgeFactor((i,i-width),np.eye(numStates))
            all_edges.add((i,i-width))
        
        
    for i in (usual_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            model.setEdgeFactor((i,i+1),np.eye(numStates))
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.setEdgeFactor((i,i-1),np.eye(numStates))
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.setEdgeFactor((i,i-width),np.eye(numStates))
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.setEdgeFactor((i,i+width),np.eye(numStates))
            all_edges.add((i,i+width))


#    print ('edge factors are done')
    return model

# =====================================
# Create Markov Model
# =====================================

    
def Create_MarkovNet(height,width,w_unary,w_pair,pixels):
    mn = MarkovNet()
    np.random.seed(1)
    num_pixels = height * width

    ########Set Unary Factor
    for i in range(num_pixels):
        k = i +1
        mn.setUnaryFactor(k,np.dot(w_unary,pixels[k]))
    
#     k = 1
#     for i in range(0,height):
#         for j in range(0,width):
#             pxl = (pixels[j,i])
# 
# #             pxl = np.array(pixels[j,i])
#             mn.setUnaryFactor(k,np.dot(w_unary,pxl))
# 
#             k += 1

#    print ('setUnaryFactor done------')
#     ##########Set Pairwise
# 
    
    left_ind = num_pixels - width + 1
        
    left_pixels = [i for i in range(1,num_pixels+1) if (i % width) == 1 and i != 1 and i != left_ind]
    right_pixels = [i for i in range(1,num_pixels+1) if (i % width) == 0 and i != width and i != num_pixels]
    up_pixels = range(2,width)
    down_pixels = range(left_ind+1 , num_pixels)
    usual_pixels = [i for i in range(1,num_pixels+1) if i not in left_pixels and i not in right_pixels and i not in up_pixels and i not in down_pixels and i not in (1,width,left_ind,num_pixels)]
        
        
#     up_pixels = set(up_pixels)
    down_pixels = set(down_pixels)
    usual_pixels = set(usual_pixels)
    left_pixels = set(left_pixels)
    right_pixels = set(right_pixels)
        
    all_edges = set()
        
        
        
    mn.setEdgeFactor((1,2), w_pair)
    all_edges.add((1,2))
    mn.setEdgeFactor((1,1+width),w_pair)
    all_edges.add((1,1+width))
        
    mn.setEdgeFactor((width,width-1),w_pair)
    all_edges.add((width,width-1))
    mn.setEdgeFactor((width,width+width),w_pair)
    all_edges.add((width,width+width))
        
        
    mn.setEdgeFactor((left_ind,left_ind +1),w_pair)
    all_edges.add((left_ind,left_ind +1))
    mn.setEdgeFactor((left_ind,left_ind - width),w_pair)
    all_edges.add((left_ind,left_ind - width))
        
    mn.setEdgeFactor((num_pixels,num_pixels - 1),w_pair)
    all_edges.add((num_pixels,num_pixels - 1))
    mn.setEdgeFactor((num_pixels,num_pixels - width),w_pair)
    all_edges.add((num_pixels,num_pixels - width))
        
        
    for i in (left_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges :
            mn.setEdgeFactor((i,i+1),w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.setEdgeFactor((i,i-width),w_pair)
            all_edges.add((i,i-width))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.setEdgeFactor((i,i+width),w_pair)
            all_edges.add((i,i+width))
        
    for i in (right_pixels):
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.setEdgeFactor((i,i-1),w_pair)
            all_edges.add((i,i-1))
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.setEdgeFactor((i,i-width),w_pair)
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.setEdgeFactor((i,i+width),w_pair)
            all_edges.add((i,i+width))
        
    for i in  up_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            mn.setEdgeFactor((i,i+1),w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.setEdgeFactor((i,i-1),w_pair)
            all_edges.add((i,i-1))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.setEdgeFactor((i,i+width),w_pair)
            all_edges.add((i,i+width))
        
    for i in  down_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            mn.setEdgeFactor((i,i+1),w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.setEdgeFactor((i,i-1),w_pair)
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.setEdgeFactor((i,i-width),w_pair)
            all_edges.add((i,i-width))
        
    for i in (usual_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            mn.setEdgeFactor((i,i+1),w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.setEdgeFactor((i,i-1),w_pair)
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.setEdgeFactor((i,i-width),w_pair)
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.setEdgeFactor((i,i+width),w_pair)
            all_edges.add((i,i+width))


#    print ('setEdgeFactor done------')
    return mn
    

