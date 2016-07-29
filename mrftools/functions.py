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
import colorsys
import itertools




# =====================================
# Plot images alog with segmentation
# =====================================
def Plot_Segmented(segment_image,original_image,height,width):
    lbl_small = Load_Resize_Label(original_image[:-4]+"_label.txt",height,width)
    lbl_sm = create_img(lbl_small)
    small_im = Load_Resize_Image(original_image,width,width)
     
    img = matplotlib.image.imread(original_image)
    seg_img = create_img(segment_image)
    fig = plt.figure()
    a=fig.add_subplot(1,4,1)
    a.set_title('original image')
    plt.imshow(img)
    
    a=fig.add_subplot(1,4,2)
    a.set_title('resized image')
    plt.imshow(small_im)

    a=fig.add_subplot(1,4,3)
    a.set_title('resized true label')
    plt.imshow(lbl_sm)

    
    a=fig.add_subplot(1,4,4)
    a.set_title('predicted label')
    plt.imshow(seg_img)

    
#     plt.title('original image','resized image','resized true label','predicted label')
#     plt.show()
#            fig.savefig(file[:-4]+".pdf")

# =====================================
# Create image using pixel labels
# =====================================
color_dic = { 0:[[160,160,160],'gray'],1:[[153,153,0],'dark green'],2:[[102,0,204],'purple'],3:[[0,153,76],'green'],4:[[0,51,102],'blue'],5:[[102,0,0],'dark red'],6:[[153,76,0],'brown'],7:[[255,153,51],'orange']}


def create_img(Z1):

    h = Z1.shape[0]
    w = Z1.shape[1]
    new_Z = np.empty((h,w,3))
    for i in range(0,h):
        for j in range(0,w):
            new_Z[i,j,:] = color_dic[Z1[i,j]][0]
    return new_Z


# =====================================
# Load and Resize image
# =====================================

def rgb2gray(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def Load_Resize_Image(image,height,width):
    imgt = Image.open(image)
    small_img = imgt.resize((width,height),resample = PIL.Image.NEAREST)
    return small_img
#     small_pix =  small_img.load()
#     return small_pix

# =====================================
# Load and Resize Label
# =====================================

def get_augmented_pixels(pixel,vertical_pos,horizontal_pos,augmented):
    if augmented == 1:
        features = []

        px = np.true_divide(np.array(pixel),255)
        vec = np.concatenate((px,[vertical_pos,horizontal_pos]))
        lst = list(itertools.product([0, 1], repeat=len(vec)))
        for i in range(len(lst)):
            features.append(np.sin(np.dot(np.array(lst[i]),vec)))
            features.append(np.cos(np.dot(np.array(lst[i]),vec)))
        features = np.array(features)

        #
        # px = np.true_divide(np.array(pixel),255)
        # vec = np.concatenate((px,[vertical_pos,horizontal_pos, 1.0]))
        # indices = np.triu_indices(len(vec))
        # prod = np.outer(vec, vec)
        # features = np.append(features, prod[indices])

    else:
        features = np.true_divide(np.array(pixel),255)
#         features = np.concatenate((features,[1]))
#         features = np.array(pixel)


    return features

def Load_Resize_Label(label,height,width):


    lbl_file = open(label)
    
    lbl = []
    lbl = np.array(lbl)
    for line in lbl_file:
        if lbl.size == 0:
            lbl = np.array(line.strip().split())
        else:
            lbl = np.vstack((lbl,np.array(line.strip().split())))
    
    
    
    img_lbl = Image.fromarray(lbl.astype(np.uint8))

#     [w,h] = lbl.shape 
#     for i in range(0,h):
#         for j in range(0,w):
#             print lbl[j,i]
#             print img_lbl.getpixel((i,j))
    
    
    
    
    
    small_l = img_lbl.resize((width,height),resample = PIL.Image.NEAREST )
    
    aa = np.array(list(small_l.getdata()))
    lbl_small = np.reshape(aa,(height,width))

    return lbl_small




# =====================================
# Create Log linear model
# =====================================
def Create_LogLinearModel(height,width,d,num_states):
    num_pixels = height * width
    model = LogLinearModel()

    num_edge = 0

    for i in range(1,num_pixels+1):
        model.declare_variable(i, num_states)
        model.set_unary_weights(i, np.random.randn(num_states, d))
        model.set_unary_features(i, np.random.randn(d))


    model.set_all_unary_factors()
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
        
    model.set_edge_factor((1, 2), np.eye(num_states))
    all_edges.add((1,2))
    model.set_edge_factor((1, 1 + width), np.eye(num_states))
    all_edges.add((1,1+width))
        
        
    model.set_edge_factor((width, width - 1), np.eye(num_states))
    all_edges.add((width,width-1))
    model.set_edge_factor((width, width + width), np.eye(num_states))
    all_edges.add((width,width+width))
        
        
    model.set_edge_factor((left_ind, left_ind + 1), np.eye(num_states))
    all_edges.add((left_ind,left_ind +1))
    model.set_edge_factor((left_ind, left_ind - width), np.eye(num_states))
    all_edges.add((left_ind,left_ind - width))
        
        
    model.set_edge_factor((num_pixels, num_pixels - 1), np.eye(num_states))
    all_edges.add((num_pixels,num_pixels - 1))
    model.set_edge_factor((num_pixels, num_pixels - width), np.eye(num_states))
    all_edges.add((num_pixels,num_pixels - width))
        
        
    for i in (left_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges :
            model.set_edge_factor((i, i + 1), np.eye(num_states))
            all_edges.add((i,i+1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.set_edge_factor((i, i - width), np.eye(num_states))
            all_edges.add((i,i-width))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.set_edge_factor((i, i + width), np.eye(num_states))
            all_edges.add((i,i+width))
        
        
    for i in (right_pixels):
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.set_edge_factor((i, i - 1), np.eye(num_states))
            all_edges.add((i,i-1))
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.set_edge_factor((i, i - width), np.eye(num_states))
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.set_edge_factor((i, i + width), np.eye(num_states))
            all_edges.add((i,i+width))
        
        
    for i in  up_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            model.set_edge_factor((i, i + 1), np.eye(num_states))
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.set_edge_factor((i, i - 1), np.eye(num_states))
            all_edges.add((i,i-1))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.set_edge_factor((i, i + width), np.eye(num_states))
            all_edges.add((i,i+width))
        
        
        
    for i in  down_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            model.set_edge_factor((i, i + 1), np.eye(num_states))
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.set_edge_factor((i, i - 1), np.eye(num_states))
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.set_edge_factor((i, i - width), np.eye(num_states))
            all_edges.add((i,i-width))
        
        
    for i in (usual_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            model.set_edge_factor((i, i + 1), np.eye(num_states))
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            model.set_edge_factor((i, i - 1), np.eye(num_states))
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            model.set_edge_factor((i, i - width), np.eye(num_states))
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            model.set_edge_factor((i, i + width), np.eye(num_states))
            all_edges.add((i,i+width))


#    print ('edge factors are done')
    return model

# =====================================
# Create Markov Model
# =====================================
def Create_MarkovNet(height,width,w_unary,w_pair,pixels):

    mn = MarkovNet()
    np.random.seed(1)

    ########Set Unary Factor
    k = 1
    for i in range(0,height):
        for j in range(0,width):
            pxl = get_augmented_pixels(pixels[j,i],np.true_divide(i,height),np.true_divide(j,width),1)

#             pxl = np.array(pixels[j,i])
            mn.set_unary_factor(k, np.dot(w_unary, pxl))

            k += 1

#    print ('set_unary_factor done------')
#     ##########Set Pairwise
# 
    num_pixels = height * width
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
        
        
        
    mn.set_edge_factor((1, 2), w_pair)
    all_edges.add((1,2))
    mn.set_edge_factor((1, 1 + width), w_pair)
    all_edges.add((1,1+width))
        
    mn.set_edge_factor((width, width - 1), w_pair)
    all_edges.add((width,width-1))
    mn.set_edge_factor((width, width + width), w_pair)
    all_edges.add((width,width+width))
        
        
    mn.set_edge_factor((left_ind, left_ind + 1), w_pair)
    all_edges.add((left_ind,left_ind +1))
    mn.set_edge_factor((left_ind, left_ind - width), w_pair)
    all_edges.add((left_ind,left_ind - width))
        
    mn.set_edge_factor((num_pixels, num_pixels - 1), w_pair)
    all_edges.add((num_pixels,num_pixels - 1))
    mn.set_edge_factor((num_pixels, num_pixels - width), w_pair)
    all_edges.add((num_pixels,num_pixels - width))
        
        
    for i in (left_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges :
            mn.set_edge_factor((i, i + 1), w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.set_edge_factor((i, i - width), w_pair)
            all_edges.add((i,i-width))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.set_edge_factor((i, i + width), w_pair)
            all_edges.add((i,i+width))
        
    for i in (right_pixels):
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.set_edge_factor((i, i - 1), w_pair)
            all_edges.add((i,i-1))
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.set_edge_factor((i, i - width), w_pair)
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.set_edge_factor((i, i + width), w_pair)
            all_edges.add((i,i+width))
        
    for i in  up_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            mn.set_edge_factor((i, i + 1), w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.set_edge_factor((i, i - 1), w_pair)
            all_edges.add((i,i-1))
                
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.set_edge_factor((i, i + width), w_pair)
            all_edges.add((i,i+width))
        
    for i in  down_pixels:
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            mn.set_edge_factor((i, i + 1), w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.set_edge_factor((i, i - 1), w_pair)
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.set_edge_factor((i, i - width), w_pair)
            all_edges.add((i,i-width))
        
    for i in (usual_pixels):
        if (i,i+1) not in all_edges and (i+1,i) not in all_edges:
            mn.set_edge_factor((i, i + 1), w_pair)
            all_edges.add((i,i+1))
                
        if (i,i-1) not in all_edges and (i-1,i) not in all_edges:
            mn.set_edge_factor((i, i - 1), w_pair)
            all_edges.add((i,i-1))
                
        if (i,i-width) not in all_edges and (i-width,i) not in all_edges:
            mn.set_edge_factor((i, i - width), w_pair)
            all_edges.add((i,i-width))
        if (i,i+width) not in all_edges and (i+width,i) not in all_edges:
            mn.set_edge_factor((i, i + width), w_pair)
            all_edges.add((i,i+width))


#    print ('set_edge_factor done------')
    return mn
