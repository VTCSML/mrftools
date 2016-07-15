'''
Created on Jul 14, 2016

@author: elahehraisi
'''
import unittest
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from EM import EM
from paired_dual import paired_dual
import numpy as np
from test_functions import *


class TestImageSegmentation(unittest.TestCase):
        
    def test_obj(self):
        d = 3
        num_states = 3
        model = Create_LogLinearModel(2,2,d,num_states)
        
# =====================================
# subgradient
# =====================================
        learner = Learner(model,MatrixBeliefPropagator)
        add_synthetic_data(learner)

        initial_weights = np.random.randn( num_states * d)
        initial_weights = np.append(initial_weights, np.random.randn( num_states * num_states))
        weights = learner.Learn(initial_weights)

        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        subgrad_obj_list = []
        subgrad_time_list = []
        t = learner.time_record[0]
        for i in range(l):
            subgrad_time_list.append(time_record[i] - t)
            subgrad_obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'subgradient'))
            

        
# =====================================
# EM
# =====================================
        learner.reset()
        learner = EM(model,MatrixBeliefPropagator)
        add_synthetic_data(learner)

        weights = learner.Learn(initial_weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        EM_obj_list = []
        EM_time_list = []
        t = learner.time_record[0]
        for i in range(l):
            EM_time_list.append(time_record[i] - t)
            EM_obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'EM'))
         
         
# =====================================
# paired_dual
# =====================================
#         learner.reset()
        learner = paired_dual(model,MatrixBeliefPropagator)
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        paired_obj_list = []
        paired_time_list = []
        t = learner.time_record[0]
        for i in range(l):
            paired_time_list.append(time_record[i] - t)
            paired_obj_list.append(learner.subgrad_obj(learner.weight_record[i,:], 'subgradient'))
         

        colors = itertools.cycle(["r", "b", "g"])
        plt.plot(subgrad_time_list,subgrad_obj_list,label='subgradient',color=next(colors))
        plt.plot(EM_time_list,EM_obj_list,label='EM',color=next(colors))
        plt.plot(paired_time_list,paired_obj_list,label='paired_Dual',color=next(colors))
        plt.xlabel('time(seconds)')
        plt.ylabel('objective')
        plt.legend(loc='upper right')
        plt.savefig('objective')
#         plt.show()   
            
    
    def test_training_accuracy(self):
        d = 3
        num_states = 3
        num_pixels = 4
        model = Create_LogLinearModel(2,2,d,num_states)
        initial_weights = np.random.randn( num_states * d)
        initial_weights = np.append(initial_weights, np.random.randn( num_states * num_states))
# =====================================
# subgradient
# =====================================
        learner = Learner(model,MatrixBeliefPropagator)
        learner.reset()
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
        subgrad_weight_record = learner.weight_record
        subgrad_time_record = learner.time_record
        t = learner.time_record[0]
        subgrad_time = np.true_divide(np.array(subgrad_time_record) - t,1000)
        subgrad_accuracy_ave_train = np.zeros(len(subgrad_time))
        counter = 0
         
        for i in range(learner.num_examples):
            data = train_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(subgrad_weight_record.shape[0]):
                w = subgrad_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display = "off")
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                bp.load_beliefs()
                Z = []
                for ii in range(1,num_pixels+1):
                    Z.append(np.argmax(bp.varBeliefs[ii]))
                cc = 0
                for k in range(len(Z)):
                    if data[2].values()[k] == Z[k]:
                        cc += 1
                 
                accuracy.append(np.true_divide(cc,num_pixels))
#                 accuracy.append(np.true_divide(sum(data[2].values() == Z),num_pixels))
               
            subgrad_accuracy_ave_train = np.add(subgrad_accuracy_ave_train,accuracy)  
             
        subgrad_accuracy_ave_train = np.true_divide(subgrad_accuracy_ave_train,counter)
        
        
# =====================================
# EM
# =====================================
        learner = EM(model,MatrixBeliefPropagator)
        learner.reset()
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
         
        EM_weight_record = learner.weight_record
        EM_time_record = learner.time_record
        t = learner.time_record[0]

        EM_time = np.true_divide(np.array(EM_time_record) - t,1000)
        EM_accuracy_ave_train = np.zeros(len(EM_time))
        counter = 0
         
        for i in range(learner.num_examples):
            data = train_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(EM_weight_record.shape[0]):
                w = EM_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display = "off")
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                bp.load_beliefs()
                Z = []
                for ii in range(1,num_pixels+1):
                    Z.append(np.argmax(bp.varBeliefs[ii]))
                cc = 0
                for k in range(len(Z)):
                    if data[2].values()[k] == Z[k]:
                        cc += 1
                 
                accuracy.append(np.true_divide(cc,num_pixels))
#                 accuracy.append(np.true_divide(sum(data[2].values() == Z),num_pixels))
               
            EM_accuracy_ave_train = np.add(EM_accuracy_ave_train,accuracy)  
             
        EM_accuracy_ave_train = np.true_divide(EM_accuracy_ave_train,counter)
        
        
# =====================================
# paired dual
# =====================================
        learner = paired_dual(model,MatrixBeliefPropagator)
        learner.reset()
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
         
        paired_weight_record = learner.weight_record
        paired_time_record = learner.time_record
        t = learner.time_record[0]

        paired_time = np.true_divide(np.array(paired_time_record) - t,1000)
        paired_accuracy_ave_train = np.zeros(len(paired_time))
        counter = 0
         
        for i in range(learner.num_examples):
            data = train_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(paired_weight_record.shape[0]):
                w = paired_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display = "off")
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                bp.load_beliefs()
                Z = []
                for ii in range(1,num_pixels+1):
                    Z.append(np.argmax(bp.varBeliefs[ii]))
                cc = 0
                for k in range(len(Z)):
                    if data[2].values()[k] == Z[k]:
                        cc += 1
                 
                accuracy.append(np.true_divide(cc,num_pixels))
#                 accuracy.append(np.true_divide(sum(data[2].values() == Z),num_pixels))
               
            paired_accuracy_ave_train = np.add(paired_accuracy_ave_train,accuracy)  
             
        paired_accuracy_ave_train = np.true_divide(paired_accuracy_ave_train,counter)
        
        
        
        
        colors = itertools.cycle(["r", "b", "g"])
              
        plt.clf()
        plt.plot(EM_time,EM_accuracy_ave_train,label='EM',color=next(colors),marker = '*')
        plt.plot(paired_time,paired_accuracy_ave_train,label='paired-dual',color=next(colors),marker = 'o')
        plt.plot(subgrad_time,subgrad_accuracy_ave_train,label='sub-gradient',color=next(colors),marker = '^')
             
        plt.ylim((0,1))
        plt.xlabel('time')
        plt.ylabel('average accuracy of training data')
        plt.legend(loc='lower right')
        plt.savefig('trainingaccturace_time')
#         plt.show()

            
    def test_testing_accuracy(self):
        d = 3
        num_states = 3
        num_pixels = 4
        model = Create_LogLinearModel(2,2,d,num_states)
        initial_weights = np.random.randn( num_states * d)
        initial_weights = np.append(initial_weights, np.random.randn( num_states * num_states))
# =====================================
# subgradient
# =====================================
        learner = Learner(model,MatrixBeliefPropagator)
        learner.reset()
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
        subgrad_weight_record = learner.weight_record
        subgrad_time_record = learner.time_record
        t = learner.time_record[0]
        subgrad_time = np.true_divide(np.array(subgrad_time_record) - t,1000)
        Subgrad_accuracy_ave_test = np.zeros(len(subgrad_time))
        counter = 0
        add_test_data()
        for i in range(len(test_data_dic)):
            data = test_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(subgrad_weight_record.shape[0]):
                w = subgrad_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display = "off")
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                bp.load_beliefs()
                Z = []
                for ii in range(1,num_pixels+1):
                    Z.append(np.argmax(bp.varBeliefs[ii]))
                cc = 0
                for k in range(len(Z)):
                    if data[0].values()[k] == Z[k]:
                        cc += 1
                 
                accuracy.append(np.true_divide(cc,num_pixels))
#                 accuracy.append(np.true_divide(sum(data[2].values() == Z),num_pixels))
               
            Subgrad_accuracy_ave_test = np.add(Subgrad_accuracy_ave_test,accuracy)  
             
        Subgrad_accuracy_ave_test = np.true_divide(Subgrad_accuracy_ave_test,counter)
        
        
# =====================================
# EM
# =====================================
        learner = EM(model,MatrixBeliefPropagator)
        learner.reset()
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
        EM_weight_record = learner.weight_record
        EM_time_record = learner.time_record
        t = learner.time_record[0]
        EM_time = np.true_divide(np.array(EM_time_record) - t,1000)
        EM_accuracy_ave_test = np.zeros(len(EM_time))
        counter = 0
        add_test_data()
        for i in range(len(test_data_dic)):
            data = test_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(EM_weight_record.shape[0]):
                w = EM_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display = "off")
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                bp.load_beliefs()
                Z = []
                for ii in range(1,num_pixels+1):
                    Z.append(np.argmax(bp.varBeliefs[ii]))
                cc = 0
                for k in range(len(Z)):
                    if data[0].values()[k] == Z[k]:
                        cc += 1
                 
                accuracy.append(np.true_divide(cc,num_pixels))
#                 accuracy.append(np.true_divide(sum(data[2].values() == Z),num_pixels))
               
            EM_accuracy_ave_test = np.add(EM_accuracy_ave_test,accuracy)  
             
        EM_accuracy_ave_test = np.true_divide(EM_accuracy_ave_test,counter)
        
        
# =====================================
# paired_dual
# =====================================
        learner = paired_dual(model,MatrixBeliefPropagator)
        learner.reset()
        add_synthetic_data(learner)
        weights = learner.Learn(initial_weights)
        paired_weight_record = learner.weight_record
        paired_time_record = learner.time_record
        t = learner.time_record[0]
        paired_time = np.true_divide(np.array(paired_time_record) - t,1000)
        paired_accuracy_ave_test = np.zeros(len(paired_time))
        counter = 0
        add_test_data()
        for i in range(len(test_data_dic)):
            data = test_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(paired_weight_record.shape[0]):
                w = paired_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
                bp = MatrixBeliefPropagator(mn)
                bp.infer(display = "off")
                bp.computeBeliefs()
                bp.computePairwiseBeliefs()
                bp.load_beliefs()
                Z = []
                for ii in range(1,num_pixels+1):
                    Z.append(np.argmax(bp.varBeliefs[ii]))
                cc = 0
                for k in range(len(Z)):
                    if data[0].values()[k] == Z[k]:
                        cc += 1
                 
                accuracy.append(np.true_divide(cc,num_pixels))
#                 accuracy.append(np.true_divide(sum(data[2].values() == Z),num_pixels))
               
            paired_accuracy_ave_test = np.add(paired_accuracy_ave_test,accuracy)  
             
        paired_accuracy_ave_test = np.true_divide(paired_accuracy_ave_test,counter)  
        
        
        
        colors = itertools.cycle(["r", "b", "g"])

        plt.clf()
        plt.plot(EM_time,EM_accuracy_ave_test,label='EM',color=next(colors),marker = '*')
        plt.plot(paired_time,paired_accuracy_ave_test,label='paired-dual',color=next(colors),marker = 'o')
        plt.plot(subgrad_time,Subgrad_accuracy_ave_test,label='sub-gradient',color=next(colors),marker = '^')
    
        plt.ylim((0,1))
        plt.xlabel('time')
        plt.ylabel('average accuracy of test_functions data')
        plt.legend(loc='lower right')
        plt.savefig('testingaccturace_time')
#         plt.show()
        


if __name__ == '__main__':
    unittest.main()