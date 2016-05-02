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



def add_train_data(weights,height,width,d,ltnt,num_states,data_type):
    print '........... adding data..........'
    model = Create_LogLinearModel(height,width,d,num_states)
    learner = TemplatedLogLinearMLE_EM(model)
    
    if data_type == 'synthetic':
        dic1 = {}
        dic1[1] = 0
        dic1[2] = 1
        dic1[3] = 0
        dic1[4] = -100
        dic2 = {}
        dic2[1] = get_augmented_pixels(np.array([255,0,0]),1)
        dic2[2] = get_augmented_pixels(np.array([0,255,0]),1)
        dic2[3] = get_augmented_pixels(np.array([255,0,0]),1)
        dic2[4] = get_augmented_pixels(np.array([0,0,255]),1)
        learner.addData(dic1,dic2)
        print ('data is added')
        dic1 = {}
        dic1[1] = 0
        dic1[2] = 0
        dic1[3] = -100
        dic1[4] = 2
        dic2 = {}
        dic2[1] = get_augmented_pixels(np.array([255,0,0]),1)
        dic2[2] = get_augmented_pixels(np.array([255,0,0]),1)
        dic2[3] = get_augmented_pixels(np.array([0,255,0]),1)
        dic2[4] = get_augmented_pixels(np.array([0,0,255]),1)
        learner.addData(dic1,dic2)
        print ('data is added')
             
        dic1 = {}
        dic1[1] = -100
        dic1[2] = 2
        dic1[3] = 1
        dic1[4] = 2
        dic2 = {}
        
            
        dic2[1] = get_augmented_pixels(np.array([255,0,0]),1)
        dic2[2] = get_augmented_pixels(np.array([0,0,255]),1)
        dic2[3] = get_augmented_pixels(np.array([0,255,0]),1)
        dic2[4] = get_augmented_pixels(np.array([0,0,255]),1)
        learner.addData(dic1,dic2)
        print ('data is added')
             
        dic1 = {}
        dic1[1] = 1
        dic1[2] = -100
        dic1[3] = 0
        dic1[4] = 2
        dic2 = {}
        dic2[1] = get_augmented_pixels(np.array([0,255,0]),1)
        dic2[2] = get_augmented_pixels(np.array([0,255,0]),1)
        dic2[3] = get_augmented_pixels(np.array([255,0,0]),1)
        dic2[4] = get_augmented_pixels(np.array([0,0,255]),1)
        learner.addData(dic1,dic2)
        print ('data is added')
    elif data_type == 'real':
        train_path = "./train/"
        for file in os.listdir(train_path):
            if file.endswith(".jpg"):
                small_pix = Load_Resize_Image(train_path+file,height,width).load()
                lbl_small = Load_Resize_Label(train_path+file[:-4]+"_label.txt",height,width)
                       
                for k in range(height):
                    mmn =  np.random.randint(width ,size = ltnt)
                    for ss in mmn:
                        lbl_small[k,np.int(ss)] = -100
                                       
                    
                lbl = []
                lbl = np.array(lbl)
                k = 1
                dic1 = {}
                dic2 = {}
                for i in range(0,height):
                    for j in range(0,width):
                        dic1[k] = lbl_small[i,j]
                        dic2[k] = get_augmented_pixels(small_pix[j,i],1)
                        k += 1
                learner.addData(dic1,dic2)
                print ('data is added')
            
    
            
            
    return learner

def train_EM(learner,weights,folder_name,height,width,d,data_type):
# # =====================================
# # E-M
# # =====================================
    print '........... training by EM..........'

    num_pixels = height * width
    maxIter = 10

    t1 = time.clock()
    EM_accuracy_0 = []
    EM_accuracy_2 = []
    EM_accuracy_3 = []
    EM_accuracy_4 = []

    for i in range(maxIter):
        # =====================================
        # E-step: inference
        # =====================================
        learner.E_step(weights,'EM')
  
        # =====================================
        # M-step: learning parameters
        # =====================================
        weights = learner.M_step(weights)
        if data_type == 'synthetic':
################################
# begin: training accuracy of synthetic data 
################################
            true_label_1 = np.array([0,1,0,2])
            true_label_2 = np.array( [0,0,1,2])
            true_label_3 = np.array([0,2,1,2])
            true_label_4 = np.array([1,1,0,2])
         
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(0,j)]))
            EM_accuracy_0.append(sum(true_label_1 == np.array(Z)))
                 
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(1,j)]))
            EM_accuracy_2.append(sum(true_label_2 == np.array(Z)))
         
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(2,j)]))
                
            EM_accuracy_3.append(sum(true_label_3 == np.array( Z)))
                 
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(3,j)]))
            EM_accuracy_4.append(sum(true_label_4 == np.array( Z))) 
#         
            
        elif data_type == 'real':
            lbl_small = Load_Resize_Label("./train/0000087_label.txt",height,width)
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(0,j)]))
            EM_accuracy_0.append(sum(lbl_small.flatten() == np.array(Z)))
     
            lbl_small = Load_Resize_Label("./train/0001677_label.txt",height,width)
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(1,j)]))
            EM_accuracy_2.append(sum(lbl_small.flatten() == np.array(Z)))
            lbl_small = Load_Resize_Label("./train/0002755_label.txt",height,width)
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(2,j)]))
            EM_accuracy_3.append(sum(lbl_small.flatten() == np.array(Z)))
            
            lbl_small = Load_Resize_Label("./train/0003793_label.txt",height,width)
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(learner.varBelief_p[(3,j)]))
            EM_accuracy_4.append(sum(lbl_small.flatten() == np.array(Z)))
            



            
    plt.plot(EM_accuracy_0,label='data 1',marker = '*')
    plt.plot(EM_accuracy_2,label='data 2',marker = 'o')
    plt.plot(EM_accuracy_3,label='data 3',marker = '^')
    plt.plot(EM_accuracy_4,label='data 4',marker = '<')
# # 
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.ylim(0, 16)
    plt.legend(loc='upper right')
    plt.show()






    t2 = time.clock()
    weight_record = learner.weight_record
    time_record = learner.time_record
    print 'Done in ' + str(t2-t1)+' seconds-----------------'
    f = open(folder_name+'/reports_'+folder_name+'.txt','a')
    f.write('train_EM: Done in ' + str(t2-t1)+' seconds-----------------')
    f.write("\n")

    learner.calculate_tau(weights, 'EM', 'q')
    learner.calculate_tau(weights, 'EM', 'p')
#       
    l = weight_record.shape[0]
    obj_list = []
    time_list = []
    t = learner.time_record[0]
    for i in range(l):
        time_list.append(time_record[i] - t)
        obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'EM'))
  
          
    f = open(folder_name+'/EM_time.txt','w')
    pickle.dump(time_list, f)
    f.close()
   
    f = open(folder_name+'/EM_obj.txt','w')
    pickle.dump(obj_list, f)
    f.close()
     
    f = open(folder_name+'/EM_step_weights.txt','w')
    pickle.dump(learner.weight_record, f)
    f.close()
     
    if data_type == 'synthetic':
        print 'plot test accuracy of synthetic data........'
        f1 = open(folder_name+'/EM_step_weights.txt','r')
        EM_step_weights = pickle.load(f1)
        EM_step_weights = np.array(EM_step_weights)
        h = EM_step_weights.shape[0]
        true_label =  np.array([3,3,1,2])
        EM_accuracy = []
        for i in range(h):
            w = EM_step_weights[i,:]
            w_unary = np.reshape(w[0:3 * d],(3,d))
            w_pair = np.reshape(w[3 * d:],(3,3))
            w_unary = np.array(w_unary,dtype = float)
            w_pair = np.array(w_pair,dtype = float)
            num_pixels = height * width
        
            pixels = np.array([[[0,0,255],[0,0,255]],[[255,0,0],[0,255,0]]])
            mn = Create_MarkovNet(height,width,w_unary,w_pair,pixels)
     
            bp = BeliefPropagator(mn)
            bp.runInference(display = "off")
            bp.computeBeliefs()
            bp.computePairwiseBeliefs()
            Z = []
            for j in range(1,num_pixels+1):
                Z.append(np.argmax(bp.varBeliefs[j]))
            
            EM_accuracy.append(sum(true_label == np.array(Z)+1))
            
        plt.plot(EM_accuracy,label='test',marker = '*')
        plt.xlabel('time')
        plt.ylabel('accuracy')
        plt.ylim(0, 5)
        plt.legend(loc='upper right')
        plt.show()
        
    
    

#     
    return weights

def train_subgrad(learner,weights,folder_name):
# =====================================
# subgradient
# =====================================
    print '........... training by subgradient..........'

    t1 = time.clock()
    weights = learner.subGradient(weights)
    t2 = time.clock()
    
    print 'Done in ' + str(t2 - t1) +' seconds-----------------'
    f = open(folder_name+'/reports_'+folder_name+'.txt','a')
    f.write('train_subgrad: Done in ' + str(t2-t1)+' seconds-----------------')
    f.write("\n")
    
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
   
   
    f = open(folder_name+'/subgrad_time.txt','w')
    pickle.dump(time_list, f)
    f.close()
   
    f = open(folder_name+'/subgrad_obj.txt','w')
    pickle.dump(obj_list, f)
    f.close()

    f = open(folder_name+'/subgrad_step_weights.txt','w')
    pickle.dump(learner.weight_record, f)
    f.close()
  
     
#     f = open('subgrad_obj.txt','r')
#     sub_obj = pickle.load(f)
#       
#     f = open('subgrad_time.txt','r')
#     sub_time = pickle.load(f)


#      
    return weights

def train_pairedDual(learner,weights,folder_name):
# =====================================
# PairedDual
# =====================================
    print '........... training by paired dual..........'

    t1 = time.clock()
    weights = learner.pairdDual_Learning(weights)
    t2 = time.clock()
    
    print 'Done in ' + str(t2 - t1)+ 'seconds-----------------'
    f = open(folder_name+'/reports_'+folder_name+'.txt','a')
    f.write('train_pairedDual: Done in ' + str(t2-t1)+' seconds-----------------')
    f.write("\n")
    
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
        obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'paired'))
   
   
    f = open(folder_name+'/paired_time.txt','w')
    pickle.dump(time_list, f)
    f.close()
   
    f = open(folder_name+'/paired_obj.txt','w')
    pickle.dump(obj_list, f)
    f.close()
    
    f = open(folder_name+'/paired_step_weights.txt','w')
    pickle.dump(learner.weight_record, f)
    f.close()
    


#     
     
#     f = open('paired_obj.txt','r')
#     paired_obj = pickle.load(f)
#       
#     f = open('paired_time.txt','r')
#     paired_time = pickle.load(f)

    return weights

def plot_objectives(folder_name):
# # =====================================
# # plot
# # =====================================
    print 'plotting objecitve..........'

    f = open(folder_name+'/EM_obj.txt','r')
    EM_obj = pickle.load(f)
       
    f = open(folder_name+'/EM_time.txt','r')
    EM_time = pickle.load(f)
    EM_time = np.array(EM_time)/1000
      
    f = open(folder_name+'/subgrad_obj.txt','r')
    sub_obj = pickle.load(f)
        
    f = open(folder_name+'/subgrad_time.txt','r')
    sub_time = pickle.load(f)
    sub_time = np.array(sub_time)/1000
      
    f = open(folder_name+'/paired_obj.txt','r')
    paired_obj = pickle.load(f)
        
    f = open(folder_name+'/paired_time.txt','r')
    paired_time = pickle.load(f)
    paired_time = np.array(paired_time)/1000
      
    colors = itertools.cycle(["r", "b", "g"])
      
    plt.plot(EM_time,EM_obj,label='EM',color=next(colors))
    plt.plot(paired_time,paired_obj,label='paired-dual',color=next(colors))
    plt.plot(sub_time,sub_obj,label='sub-gradient',color=next(colors))
#     
# 
    plt.xlabel('time(seconds)')
    plt.ylabel('objective')
    plt.legend(loc='upper right')
    plt.savefig(folder_name+'/objective')
#     plt.show()

    plt.ylim(0, 10)
    plt.savefig(folder_name+'/zoomed_objective')
    
#     plt.show()

def test_data(weights,height,width,d,num_states,image):
# =====================================
# Testing
# =====================================

    w = weights
    w_unary = np.reshape(w[0:d * num_states],(num_states,d))
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)
    num_pixels = height * width
    
    pixels = Load_Resize_Image(image,height,width).load()
    mn = Create_MarkovNet(height,width,w_unary,w_pair,pixels)
 
    bp = BeliefPropagator(mn)
    bp.runInference(display = "off")
    bp.computeBeliefs()
    bp.computePairwiseBeliefs()
    Z = []
    for i in range(1,num_pixels+1):
        Z.append(np.argmax(bp.varBeliefs[i]))
    return Z    
    
def plot_accuracy(height,width,d,num_states,image,ltnt,folder_name):
# EM ######################
    print 'plot accuracy........'
    f1 = open(folder_name+'/EM_step_weights.txt','r')
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
    
    
    f1 = open(folder_name+'/EM_accuracy_'+str(height)+'_'+str(width)+'ltnt_'+str(ltnt)+'.txt','w')
    pickle.dump(EM_accuracy,f1)
    f1.close()
    
    
    f = open(folder_name+'/EM_time.txt','r')
    EM_time = pickle.load(f)
    EM_time = np.array(EM_time)/1000
#     print '----------------------------'
#     print EM_time
#     print EM_accuracy

# # sub gradient ######################
    f1 = open(folder_name+'/subgrad_step_weights.txt','r')
    subgrad_step_weights = pickle.load(f1)
    subgrad_step_weights = np.array(subgrad_step_weights)
    h = subgrad_step_weights.shape[0]
    subgrad_accuracy = []
    for i in range(h):
        w = subgrad_step_weights[i,:]
        Z = test_data(w,height,width,d,num_states,image)
        Z1 = np.reshape(Z,(height,width))
        subgrad_accuracy.append(sum(sum(true_label == Z1)))
         
    f1 = open(folder_name+'/subgrad_accuracy_'+str(height)+'_'+str(width)+'ltnt_'+str(ltnt)+'.txt','w')
    pickle.dump(subgrad_accuracy,f1)
    f1.close()
    
    f = open(folder_name+'/subgrad_time.txt','r')
    subgrad_time = pickle.load(f)
    subgrad_time = np.array(subgrad_time)/1000
 
     
#     # paired dual ######################
    f1 = open(folder_name+'/paired_step_weights.txt','r')
    paired_step_weights = pickle.load(f1)
    paired_step_weights = np.array(paired_step_weights)
    h = paired_step_weights.shape[0]
    paired_accuracy = []
    
 
    for i in range(h):
        w = paired_step_weights[i,:]
        Z = test_data(w,height,width,d,num_states,image)
        Z1 = np.reshape(Z,(height,width))
        paired_accuracy.append(sum(sum(true_label == Z1)))
         
    
    f1 = open(folder_name+'/paired_accuracy_'+str(height)+'_'+str(width)+'ltnt_'+str(ltnt)+'.txt','w')
    pickle.dump(paired_accuracy,f1)
    f1.close()
    
    f = open(folder_name+'/paired_time.txt','r')
    paired_time = pickle.load(f)
    paired_time = np.array(paired_time)/1000

    plt.clf()


    colors = itertools.cycle(["r", "b", "g"])
      
    plt.scatter(EM_time,EM_accuracy,label='EM',color=next(colors),marker = '*')
    plt.scatter(paired_time,paired_accuracy,label='paired-dual',color=next(colors),marker = 'o')
    plt.scatter(subgrad_time,subgrad_accuracy,label='sub-gradient',color=next(colors),marker = '^')
     

    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
#     plt.savefig(folder_name+'/objective')

    plt.savefig(folder_name+'/'+str(image[7:-4])+'_accuray')
#     plt.show()

def main():
#############
# begin: synthetic data 
#############
#     height = 2
#     width = 2
#     num_states = 3
#     data_type = 'synthetic'
#############
# end: synthetic data 
#############

#############
# begin: real data 
#############
    height = 4
    width = 4
    num_states = 8
    data_type = 'real'
#############
# end: real data 
#############

# number of latent variables
    num_ltnt =2

#     d = 9
    d = 10
#     d = 3
    folder_name = str(height)+'*'+str(width)
    
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
# # # #     # add node weights
# #     bias = np.random.randn(1)
    weights = np.random.randn( num_states * d)
# #     # add edge weights
    weights = np.append(weights, np.random.randn( num_states * num_states))
#     
#     np.savetxt("initial_weights.csv", weights, delimiter=",")
# # # #     
# #     weights = genfromtxt('initial_weights.csv', delimiter=',')
# # #     
    learner = add_train_data(weights,height,width,d,num_ltnt,num_states,data_type)
# # #     joblib.dump(learner, 'learner/learner.pkl')
# #     
# #     learner = joblib.load('learner/learner.pkl')
# # 
#        
    f = open(folder_name+'/reports_'+folder_name+'.txt','a')
    f.write('\n')
    f.write('********************************************')
    f.write('\n')
    f.write('number of hidden variables: '+ str(num_ltnt)+'*'+str(height))
    f.write('\n')
    f.write('number of training data: '+ str(len(learner.labels)))
    f.write('\n')
    f.close()
        
    num_pixels = height*width
# 
# # # #########################EM
    newWeight = train_EM(learner,weights,folder_name,height,width,d,data_type)
    np.savetxt(folder_name+"/EM_final_weights.csv", newWeight, delimiter=",")
#      
    learner.clearRecord()
# # # #########################sub gradient
    newWeight = train_subgrad(learner,weights,folder_name)
    np.savetxt(folder_name+"/subgrad_final_weights.csv", newWeight, delimiter=",")
#  
#  
    learner.clearRecord()
#  # #########################paired Dual
    newWeight = train_pairedDual(learner,weights,folder_name)
    np.savetxt(folder_name+"/pairedDual_final_weights.csv", newWeight, delimiter=",")
# # #  
# #########################plot objective
    plot_objectives(folder_name)
   
# #########################plot training and testing accuracy
#     test_path = "./train/"
#     for file in os.listdir(test_path):
#         if file.endswith('.jpg'):
#             plot_accuracy(height,width,d,num_states,test_path+file,num_ltnt,folder_name)
             

# 
# #########################plot image segmentation
#     newWeight = genfromtxt(folder_name+'/EM_final_weights.csv', delimiter=',')
#     test_path = "./test/"
#     for file in os.listdir(test_path):
#         if file.endswith('.jpg'):
#             Z = test_data(newWeight,height,width,d,num_states,test_path+file)
#             true_label =  Load_Resize_Label(test_path+file[:-4]+"_label.txt", height, width)
#             Z1 = np.reshape(Z,(height,width))
#             Plot_Segmented(Z1,test_path+file,height,width)


            
if __name__ == "__main__":
    main()