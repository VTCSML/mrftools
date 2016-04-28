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


def add_train_data(weights,height,width,d):
    print '........... adding data..........'
    model = Create_LogLinearModel(height,width,d,8)
    learner = TemplatedLogLinearMLE_EM(model)
    

    train_path = "./train/"
    for file in os.listdir(train_path):
        if file.endswith(".jpg"):
 
            small_pix = Load_Resize_Image(train_path+file,height,width).load()
            lbl_small = Load_Resize_Label(train_path+file[:-4]+"_label.txt",height,width)
             
            for k in range(height):
                mmn =  np.random.randint(width ,size = 2)
                for ss in mmn:
                    lbl_small[k,np.int(ss)] = -100
                             
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
            
    return learner

def train_EM(learner,weights):
# # =====================================
# # E-M
# # =====================================
    print '........... training by EM..........'

    maxIter = 20
    t1 = time.clock()
    for i in range(maxIter):
        # =====================================
        # E-step: inference
        # =====================================
        learner.E_step(weights,'EM')
  
        # =====================================
        # M-step: learning parameters
        # =====================================
        weights = learner.M_step(weights)
  
#     w = learner.dic
    t2 = time.clock()
    weight_record = learner.weight_record
    time_record = learner.time_record
#     print weight_record.shape
#     print time_record.shape
  
    learner.calculate_tau(weights, 'EM', 'q')
    learner.calculate_tau(weights, 'EM', 'p')
      
    l = weight_record.shape[0]
    obj_list = []
    time_list = []
    t = learner.time_record[0]
    for i in range(l):
        time_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'subgradient'))
  
          
    f = open('EM_time.txt','w')
    pickle.dump(time_list, f)
    f.close()
  
    f = open('EM_obj.txt','w')
    pickle.dump(obj_list, f)
    f.close()
    
    f = open('EM_step_weights.txt','w')
    pickle.dump(learner.weight_record, f)
    f.close()
    
    
#     f1 = open('EM_step_weights.txt','r')
#     test_w = pickle.load(f1)
#     print test_w
#     
#     
#     f = open('EM_obj.txt','r')
#     EM_obj = pickle.load(f)
#      
#     f = open('EM_time.txt','r')
#     EM_time = pickle.load(f)
    print 'Done in ' + str(t2-t1)+' seconds-----------------'

#     
    return weights

def train_subgrad(learner,weights):
# =====================================
# subgradient
# =====================================
    print '........... training by subgradient..........'

    t1 = time.clock()
    weights = learner.subGradient(weights)
    t2 = time.clock()
    learner.calculate_tau(weights, 'subgradient', 'q')
    learner.calculate_tau(weights, 'subgradient', 'p')
    weight_record = learner.weight_record
    time_record = learner.time_record
    l = weight_record.shape[0]
    obj_list = []
    time_list = []
    t = learner.time_record[0]
    for i in range(l):
        time_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'subgradient'))
   
   
    f = open('subgrad_time.txt','w')
    pickle.dump(time_list, f)
    f.close()
   
    f = open('subgrad_obj.txt','w')
    pickle.dump(obj_list, f)
    f.close()

    f = open('subgrad_step_weights.txt','w')
    pickle.dump(learner.weight_record, f)
    f.close()
  
     
#     f = open('subgrad_obj.txt','r')
#     sub_obj = pickle.load(f)
#       
#     f = open('subgrad_time.txt','r')
#     sub_time = pickle.load(f)
    print 'Done in ' + str(t2 - t1) +' seconds-----------------'

#      
    return weights

def train_pairedDual(learner,weights):
# =====================================
# PairedDual
# =====================================
    print '........... training by paired dual..........'

    t1 = time.clock()
    weights = learner.pairdDual_Learning(weights)
    t2 = time.clock()
    
    learner.calculate_tau(weights, 'paired', 'q')
    learner.calculate_tau(weights, 'paired', 'p')
    weight_record = learner.weight_record
    time_record = learner.time_record
    l = weight_record.shape[0]
    obj_list = []
    time_list = []
    t = learner.time_record[0]
    for i in range(l):
        time_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'subgradient'))
   
   
    f = open('paired_time.txt','w')
    pickle.dump(time_list, f)
    f.close()
   
    f = open('paired_obj.txt','w')
    pickle.dump(obj_list, f)
    f.close()
    
    f = open('paired_step_weights.txt','w')
    pickle.dump(learner.weight_record, f)
    f.close()
    
    
    print 'Done in ' + str(t2 - t1)+ 'seconds-----------------'

#     
     
#     f = open('paired_obj.txt','r')
#     paired_obj = pickle.load(f)
#       
#     f = open('paired_time.txt','r')
#     paired_time = pickle.load(f)

    return weights

def plot_objectives():
# # =====================================
# # plot
# # =====================================
    print '........... plotting..........'

    f = open('EM_obj.txt','r')
    EM_obj = pickle.load(f)
       
    f = open('EM_time.txt','r')
    EM_time = pickle.load(f)
      
      
    f = open('subgrad_obj.txt','r')
    sub_obj = pickle.load(f)
        
    f = open('subgrad_time.txt','r')
    sub_time = pickle.load(f)
      
    f = open('paired_obj.txt','r')
    paired_obj = pickle.load(f)
        
    f = open('paired_time.txt','r')
    paired_time = pickle.load(f)
      
    colors = itertools.cycle(["r", "b", "g"])
      
    plt.plot(EM_time,EM_obj,label='EM',color=next(colors))
    plt.plot(paired_time,paired_obj,label='paired-dual',color=next(colors))
    plt.plot(sub_time,sub_obj,label='sub-gradient',color=next(colors))
#     
# 
    plt.xlabel('time')
    plt.ylabel('objective')
    plt.legend(loc='upper right')
    plt.show()

def test_data(weights,height,width,d,num_states,image):
# =====================================
# Testing
# =====================================
    print '........... testing ..........'

    w = weights
    w_unary = np.reshape(w[0:d * num_states],(num_states,d))
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)
    num_pixels = height * width
    
    pixels = Load_Resize_Image(image,height,width).load()
    mn = Create_MarkovNet(height,width,w_unary,w_pair,pixels)
 
    bp = BeliefPropagator(mn)
    bp.runInference()
    bp.computePairwiseBeliefs()
    Z = []
    for i in range(1,num_pixels+1):
        Z.append(np.argmax(bp.varBeliefs[i]))
    return Z    
    
def plot_accuracy(height,width,d,num_states,image):
# EM ######################
    f1 = open('EM_step_weights.txt','r')
    EM_step_weights = pickle.load(f1)
    EM_step_weights = np.array(EM_step_weights)
    h = EM_step_weights.shape[0]
    true_label =  Load_Resize_Label(image[:-4]+"_label.txt", height, width)
    EM_accuracy = []
    for i in range(h):
        w = EM_step_weights[i,:]
        Z = test_data(w,height,width,d,num_states,image)
        Z1 = np.reshape(Z,(height,width))
        EM_accuracy.append(sum(sum(true_label == Z1)))
    
    f = open('EM_time.txt','r')
    EM_time = pickle.load(f)
#     print '----------------------------'
#     print EM_time
#     print EM_accuracy


# # sub gradient ######################
    f1 = open('subgrad_step_weights.txt','r')
    subgrad_step_weights = pickle.load(f1)
    subgrad_step_weights = np.array(subgrad_step_weights)
    h = subgrad_step_weights.shape[0]
    subgrad_accuracy = []
    for i in range(h):
        w = subgrad_step_weights[i,:]
        Z = test_data(w,height,width,d,num_states,image)
        Z1 = np.reshape(Z,(height,width))
        subgrad_accuracy.append(sum(sum(true_label == Z1)))
         
    f = open('subgrad_time.txt','r')
    subgrad_time = pickle.load(f)
# 
#     
#     
#     # paired dual ######################
    f1 = open('paired_step_weights.txt','r')
    paired_step_weights = pickle.load(f1)
    paired_step_weights = np.array(paired_step_weights)
    h = paired_step_weights.shape[0]
    paired_accuracy = []
 
 
    for i in range(h):
        w = paired_step_weights[i,:]
        Z = test_data(w,height,width,d,num_states,image)
        Z1 = np.reshape(Z,(height,width))
        paired_accuracy.append(sum(sum(true_label == Z1)))
         
    
    f = open('paired_time.txt','r')
    paired_time = pickle.load(f)


    
    colors = itertools.cycle(["r", "b", "g"])
      
    plt.plot(EM_time,EM_accuracy,label='EM',color=next(colors))
#     plt.scatter(paired_time,paired_accuracy,label='paired-dual',color=next(colors))
#     plt.scatter(subgrad_time,subgrad_accuracy,label='sub-gradient',color=next(colors))
#     
# 
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.show()
    

def main():

    # =====================================
    # Create Model
    # =====================================

    height = 3
    width = 3
    d = 3
    num_states = 8
# # #     # add node weights
    weights = np.random.randn( num_states * d)
#     # add edge weights
    weights = np.append(weights, np.random.randn( num_states * num_states))
    np.savetxt("initial_weights.csv", weights, delimiter=",")
#     
    weights = genfromtxt('initial_weights.csv', delimiter=',')
# #     
    learner = add_train_data(weights,height,width,d)
    joblib.dump(learner, 'learner/learner.pkl')
    
#     learner = joblib.load('learner/learner.pkl')
# #     

# #########################EM
    newWeight = train_EM(learner,weights)
    np.savetxt("EM_final_weights.csv", newWeight, delimiter=",")

    learner.clearRecord()
# #########################sub gradient
    newWeight = train_subgrad(learner,weights)
    np.savetxt("subgrad_final_weights.csv", newWeight, delimiter=",")


    learner.clearRecord()
 # #########################paired Dual
    newWeight = train_pairedDual(learner,weights)
    np.savetxt("pairedDual_final_weights.csv", newWeight, delimiter=",")
#
# #########################plot objective
    plot_objectives()

# #########################plot training and testing accuracy
#     test_path = "./test/"
#     for file in os.listdir(test_path):
#         if file.endswith('.jpg'):
#             plot_accuracy(height,width,d,num_states,test_path+file)
            

#########################plot image segmentation
    newWeight = genfromtxt('EM_final_weights.csv', delimiter=',')
    test_path = "./train/"
    for file in os.listdir(test_path):
        if file.endswith('.jpg'):
            Z = test_data(newWeight,height,width,d,num_states,test_path+file)
            true_label =  Load_Resize_Label(test_path+file[:-4]+"_label.txt", height, width)
            Z1 = np.reshape(Z,(height,width))
            Plot_Segmented(Z1,test_path+file,height,width)
            print sum(true_label == Z1)

            




            
if __name__ == "__main__":
    main()
