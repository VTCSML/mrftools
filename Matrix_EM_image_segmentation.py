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



def add_train_data(weights,height,width,d,ltnt,num_states):
    print '........... adding data..........'
    model = Create_LogLinearModel(height,width,d,num_states)
    learner = MatrixTemplatedLogLinearMLE_EM(model)
    
    train_path = "./train/"
    for file in os.listdir(train_path):
        if file.endswith(".jpg"):
            small_pix = Load_Resize_Image(train_path+file,height,width).load()
#                 print train_path+file
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
                    if lbl_small[i,j] == 255:
                        dic1[k] = -100
                    dic2[k] = get_augmented_pixels(small_pix[j,i],np.true_divide(i,height),np.true_divide(j,width),1)
                    k += 1
            learner.addData(dic1,dic2)
            print ('data is added')
            
            
            
    return learner

def train_EM(learner,weights,folder_name,height,width,d):
# # =====================================
# # E-M
# # =====================================
    print '........... training by EM..........'

    num_pixels = height * width
    maxIter = 100

    t1 = time.clock()

    for i in range(maxIter):
        old_weights  = weights
        # =====================================
        # E-step: inference
        # =====================================
        learner.E_step(weights,'EM')
  
        # =====================================
        # M-step: learning parameters
        # =====================================
        weights = learner.M_step(weights)
        new_weights = weights

        if np.linalg.norm(np.subtract(old_weights,new_weights)) < 0.00001:
            print 'EM has converged after '+ str(i)+' iterations......'
            break;
        

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
    weights = learner.paired_dual_learning(weights)
    t2 = time.clock()
    
    print 'Done in ' + str(t2 - t1)+ ' seconds-----------------'
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
        obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'subgradient'))
    # print learner.subgrad_obj(learner.weight_record[-1,:], 'subgradient')
   
   
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
    print 'plotting objective..........'

    f = open(folder_name+'/EM_obj.txt','r')
    EM_obj = pickle.load(f)

    f = open(folder_name+'/EM_time.txt','r')
    EM_time = pickle.load(f)
    EM_time = np.true_divide(np.array(EM_time),1000)

    f = open(folder_name+'/subgrad_obj.txt','r')
    sub_obj = pickle.load(f)

    f = open(folder_name+'/subgrad_time.txt','r')
    sub_time = pickle.load(f)
    sub_time = np.true_divide(np.array(sub_time),1000)

    f = open(folder_name+'/paired_obj.txt','r')
    paired_obj = pickle.load(f)
        
    f = open(folder_name+'/paired_time.txt','r')
    paired_time = pickle.load(f)
    paired_time = np.true_divide(np.array(paired_time),1000)
      
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

    plt.ylim(0, 60)
    plt.savefig(folder_name+'/zoomed_objective')
    plt.clf()
#     plt.show()

def test_data(weights,height,width,d,num_states,image, messages=None):
# =====================================
# Testing
# =====================================

    w = weights
    w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
    w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
    w_unary = np.array(w_unary,dtype = float)
    w_pair = np.array(w_pair,dtype = float)
    num_pixels = height * width


    pixels = Load_Resize_Image(image, height, width).load()
    mn = Create_MarkovNet(height, width, w_unary, w_pair, pixels)

    bp = MatrixBeliefPropagator(mn)

    if messages is not None:
        bp.set_messages(messages)

    bp.runInference(display = "off")
    bp.computeBeliefs()
    bp.computePairwiseBeliefs()
    bp.load_beliefs()
    Z = []
    for i in range(1,num_pixels+1):
        Z.append(np.argmax(bp.varBeliefs[i]))
    messages = bp.message_mat
    return Z, messages
    
def EM_plot_accuracy(height,width,d,num_states,image,ltnt,folder_name):
# EM ######################
#     print 'plot accuracy........'
    f1 = open(folder_name+'/EM_step_weights.txt','r')
    EM_step_weights = pickle.load(f1)
    EM_step_weights = np.array(EM_step_weights)
    h = EM_step_weights.shape[0]
    true_label =  Load_Resize_Label(image[:-4]+"_label.txt", height, width)

    return compute_accuracies(EM_step_weights, true_label, height, width, d, num_states, image)

def compute_accuracies(weights, true_label, height, width, d, num_states, image):
    num_pixels = height * width
    messages = None
    accuracy = []
    for i in range(weights.shape[0]):
        w = weights[i,:]
        Z, messages = test_data(w, height, width, d, num_states, image, messages)
        Z1 = np.reshape(Z,(height,width))
        accuracy.append(np.true_divide(sum(sum(true_label == Z1)),num_pixels))
    return accuracy


def Subgrad_plot_accuracy(height,width,d,num_states,image,ltnt,folder_name):
# # # sub gradient ######################
    f1 = open(folder_name+'/subgrad_step_weights.txt','r')
    subgrad_step_weights = pickle.load(f1)
    true_label =  Load_Resize_Label(image[:-4]+"_label.txt", height, width)
    subgrad_step_weights = np.array(subgrad_step_weights)

    return compute_accuracies(subgrad_step_weights, true_label, height, width, d, num_states, image)


def Paired_plot_accuracy(height,width,d,num_states,image,ltnt,folder_name):
    # #     # paired dual ######################
    f1 = open(folder_name+'/paired_step_weights.txt','r')
    paired_step_weights = pickle.load(f1)
    paired_step_weights = np.array(paired_step_weights)
    true_label =  Load_Resize_Label(image[:-4]+"_label.txt", height, width)

    return compute_accuracies(paired_step_weights, true_label, height, width, d, num_states, image)


def main():

    height = 10
    width = 12
    num_states = 8
    num_latent = 3
    d = 64
    # d = 3

    folder_name = str(height)+'*'+str(width)
        
        
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
# # # # #     # add node weights
    weights = np.random.randn( num_states * d)
# #     # add edge weights
    weights = np.append(weights, np.random.randn( num_states * num_states))
#     
    np.savetxt("initial_weights.csv", weights, delimiter=",")
# # # #     
    weights = genfromtxt('initial_weights.csv', delimiter=',')
# # #     
    learner = add_train_data(weights,height,width,d,num_latent,num_states)
# # # #     joblib.dump(learner, 'learner/learner.pkl')
# # #     
# # #     learner = joblib.load('learner/learner.pkl')
# # # 
# #      
    learner.setRegularization(0, 0.25)
    f = open(folder_name+'/reports_'+folder_name+'.txt','a')
    f.write('\n')
    f.write('********************************************')
    f.write('\n')
    f.write('number of hidden variables: '+ str(num_latent)+'*'+str(height))
    f.write('\n')
    f.write('number of training data: '+ str(len(learner.labels)))
    f.write('\n')
    f.write('l1 regularizer: '+ str(learner.l1Regularization))
    f.write('\n')
    f.write('l2 regularizer: '+ str(learner.l2Regularization))
    f.write('\n')
    f.close()
            
    num_pixels = height*width

    # train EM
    newWeight = train_EM(learner,weights,folder_name,height,width,d)
    np.savetxt(folder_name+"/EM_final_weights.csv", newWeight, delimiter=",")


    learner.reset()
    # train subgradient
    newWeight = train_subgrad(learner,weights,folder_name)
    np.savetxt(folder_name+"/subgrad_final_weights.csv", newWeight, delimiter=",")

    learner.reset()
    # train PDL
    newWeight = train_pairedDual(learner,weights,folder_name)
    np.savetxt(folder_name+"/pairedDual_final_weights.csv", newWeight, delimiter=",")

    plot_objectives(folder_name)
    
# #########################plot training testing accuracy

    f = open(folder_name+'/EM_time.txt','r')
    EM_time = pickle.load(f)
    EM_time = np.true_divide(np.array(EM_time),1000)

    f = open(folder_name+'/subgrad_time.txt','r')
    subgrad_time = pickle.load(f)
    subgrad_time = np.true_divide(np.array(subgrad_time),1000)

    f = open(folder_name+'/paired_time.txt','r')
    paired_time = pickle.load(f)
    paired_time = np.true_divide(np.array(paired_time),1000)
      
    EM_accuracy_ave_train = np.zeros(len(EM_time))
    Subgrad_accuracy_ave_train = np.zeros(len(subgrad_time))
    Paired_accuracy_ave_train = np.zeros(len(paired_time))
   
    test_path = "./train/"
    counter = 0
    for file in os.listdir(test_path):
        if file.endswith('.jpg'):
            counter+= 1
            EM_accuracy_ave_train = np.add(EM_accuracy_ave_train,EM_plot_accuracy(height,width,d,num_states,test_path+file,num_latent,folder_name))
            Subgrad_accuracy_ave_train = np.add(Subgrad_accuracy_ave_train,Subgrad_plot_accuracy(height,width,d,num_states,test_path+file,num_latent,folder_name))
            Paired_accuracy_ave_train = np.add(Paired_accuracy_ave_train,Paired_plot_accuracy(height,width,d,num_states,test_path+file,num_latent,folder_name))
   
                       
            # print EM_accuracy_ave_train
    EM_accuracy_ave_train = np.true_divide(EM_accuracy_ave_train,counter)
    Subgrad_accuracy_ave_train = np.true_divide(Subgrad_accuracy_ave_train,counter)
    Paired_accuracy_ave_train = np.true_divide(Paired_accuracy_ave_train,counter)
   
       
    colors = itertools.cycle(["r", "b", "g"])
          
    plt.plot(EM_time,EM_accuracy_ave_train,label='EM',color=next(colors),marker = '*')
    plt.plot(paired_time,Paired_accuracy_ave_train,label='paired-dual',color=next(colors),marker = 'o')
    plt.plot(subgrad_time,Subgrad_accuracy_ave_train,label='sub-gradient',color=next(colors),marker = '^')
         
    plt.ylim((0,1))
    plt.xlabel('time')
    plt.ylabel('average accuracy of training data')
    plt.legend(loc='lower right')
#     plt.savefig(folder_name+'/objective')
    
    plt.savefig(folder_name+'/average_train_accuracy')
    # plt.show()

    plt.clf()

#     # #########################plot testing testing accuracy
    EM_accuracy_ave_test = np.zeros(len(EM_time))
    Subgrad_accuracy_ave_test = np.zeros(len(subgrad_time))
    Paired_accuracy_ave_test = np.zeros(len(paired_time))

    test_path = "./test/"
    counter = 0
    for file in os.listdir(test_path):
        if file.endswith('.jpg'):
            counter+= 1
            EM_accuracy_ave_test = np.add(EM_accuracy_ave_test,EM_plot_accuracy(height,width,d,num_states,test_path+file,num_latent,folder_name))
            Subgrad_accuracy_ave_test = np.add(Subgrad_accuracy_ave_test,Subgrad_plot_accuracy(height,width,d,num_states,test_path+file,num_latent,folder_name))
            Paired_accuracy_ave_test = np.add(Paired_accuracy_ave_test,Paired_plot_accuracy(height,width,d,num_states,test_path+file,num_latent,folder_name))


#             print EM_accuracy_ave_train
    EM_accuracy_ave_test = np.true_divide(EM_accuracy_ave_test,counter)
    Subgrad_accuracy_ave_test = np.true_divide(Subgrad_accuracy_ave_test,counter)
    Paired_accuracy_ave_test = np.true_divide(Paired_accuracy_ave_test,counter)


    colors = itertools.cycle(["r", "b", "g"])

    plt.plot(EM_time,EM_accuracy_ave_test,label='EM',color=next(colors),marker = '*')
    plt.plot(paired_time,Paired_accuracy_ave_test,label='paired-dual',color=next(colors),marker = 'o')
    plt.plot(subgrad_time,Subgrad_accuracy_ave_test,label='sub-gradient',color=next(colors),marker = '^')

    plt.ylim((0,1))
    plt.xlabel('time')
    plt.ylabel('average accuracy of test data')
    plt.legend(loc='lower right')

    plt.savefig(folder_name+'/'+'average_testing_accuracy')
    # plt.show()



# 
# #########################plot image segmentation
    newWeight = genfromtxt(folder_name+'/pairedDual_final_weights.csv', delimiter=',')
    test_path = "./train/"
    for file in os.listdir(test_path):
        if file.endswith('.jpg'):
            Z, bp = test_data(newWeight,height,width,d,num_states,test_path+file)
            true_label =  Load_Resize_Label(test_path+file[:-4]+"_label.txt", height, width)
            Z1 = np.reshape(Z,(height,width))
            Plot_Segmented(Z1,test_path+file,height,width)
            plt.savefig(folder_name + '/' + file[:-4] + "result")

if __name__ == "__main__":
    main()