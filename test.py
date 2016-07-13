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



def main():
    d = 3
    num_states = 3
#     num_latent = 3

    weights = np.random.randn( num_states * d)
# #     # add edge weights
    weights = np.append(weights, np.random.randn( num_states * num_states))
    model = Create_LogLinearModel(2,2,d,num_states)
    learner = Learner(model,MatrixBeliefPropagator)
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
    learner.addData(dic1,dic2)
    print ('data is added')
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
    learner.addData(dic1,dic2)
    print ('data is added')
         
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
    learner.addData(dic1,dic2)
    print ('data is added')
         
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
    learner.addData(dic1,dic2)
    print ('data is added')
    
    
    learner.setRegularization(0, 0.25)
    num_pixels = 4
    
    w = learner.Learn(weights)
    
    w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)
    
    
##### Testing on Training data######
    dic1 = {}
    dic1[1] = 1
    dic1[2] = 1
    dic1[3] = 0
    dic1[4] = 2
    dic2 = {}
    dic2[1] = np.array([0,1,0])
    dic2[2] = np.array([0,1,0])
    dic2[3] = np.array([1,0,0])
    dic2[4] = np.array([0,0,1])
    mn = Create_MarkovNet(2, 2, w_unary, w_pair, dic2)

    
    bp = MatrixBeliefPropagator(mn)


    bp.infer(display = "off")
    bp.computeBeliefs()
    bp.computePairwiseBeliefs()
    bp.load_beliefs()
    Z = []
    for i in range(1,num_pixels+1):
        Z.append(np.argmax(bp.varBeliefs[i]))

    accuracy = 0
    for i in range(4):
        if dic1[i+1] == Z[i]:
            accuracy += 1
                
    print np.true_divide(accuracy,4)
    

##### Testing on Test data######
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
    mn = Create_MarkovNet(2, 2, w_unary, w_pair, dic2)

    bp = MatrixBeliefPropagator(mn)

    bp.infer(display = "off")
    bp.computeBeliefs()
    bp.computePairwiseBeliefs()
    bp.load_beliefs()
    Z = []
    for i in range(1,num_pixels+1):
        Z.append(np.argmax(bp.varBeliefs[i]))

    accuracy = 0
    for i in range(4):
        if dic1[i+1] == Z[i]:
            accuracy += 1
                
    print np.true_divide(accuracy,4)



    
    
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
    

if __name__ == "__main__":
    main()

