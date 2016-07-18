'''
Created on Jul 14, 2016

@author: elahehraisi
'''
import unittest
from MatrixBeliefPropagator import MatrixBeliefPropagator
from Learner import Learner
from EM import EM
from PairedDual import PairedDual
import numpy as np
import matplotlib.pyplot as plt
from MarkovNet   import MarkovNet
from LogLinearModel import LogLinearModel
import itertools



class TestImageSegmentation(unittest.TestCase):
        
    def test_obj(self):
        d = 3
        num_states = 3
        model = self.Create_LogLinearModel(2,2,d,num_states)
        
        # =====================================
        # subgradient
        # =====================================
        learner = Learner(model,MatrixBeliefPropagator)
        self.add_synthetic_data(learner)

        initial_weights = np.random.randn( num_states * d)
        initial_weights = np.append(initial_weights, np.random.randn( num_states * num_states))
        weights = learner.learn(initial_weights)

        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        subgrad_obj_list = []
        subgrad_time_list = []
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            subgrad_time_list.append(time_record[i] - t)
            new_obj = learner.subgrad_obj(learner.weight_record[i,:], 'subgradient')
            subgrad_obj_list.append(new_obj)
            assert (new_obj <= old_obj), "subgradient objective is not decreasing"
            old_obj = new_obj
                        
        # =====================================
        # EM
        # =====================================
        learner.reset()
        learner = EM(model,MatrixBeliefPropagator)
        self.add_synthetic_data(learner)
 
        weights = learner.learn(initial_weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        EM_obj_list = []
        EM_time_list = []
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            EM_time_list.append(time_record[i] - t)
            new_obj = learner.subgrad_obj(learner.weight_record[i,:], 'EM')
            EM_obj_list.append(new_obj)
            assert (new_obj <= old_obj), "EM objective is not decreasing"
            old_obj = new_obj
            
         
         
        # =====================================
        # paired_dual
        # =====================================
#         learner.reset()
        learner = PairedDual(model,MatrixBeliefPropagator)
        self.add_synthetic_data(learner)
        weights = learner.learn(initial_weights)
        weight_record = learner.weight_record
        time_record = learner.time_record
        l = weight_record.shape[0]
        paired_obj_list = []
        paired_time_list = []
        t = learner.time_record[0]
        old_obj = np.Inf
        for i in range(l):
            paired_time_list.append(time_record[i] - t)
            new_obj = learner.subgrad_obj(learner.weight_record[i,:], 'subgradient')
            paired_obj_list.append(new_obj)
            assert (new_obj <= old_obj), "paired dual objective is not decreasing"
            old_obj = new_obj           
          
    def test_training_accuracy(self):
        d = 3
        num_states = 3
        num_pixels = 4
        model = self.Create_LogLinearModel(2,2,d,num_states)
        initial_weights = np.random.randn( num_states * d)
        initial_weights = np.append(initial_weights, np.random.randn( num_states * num_states))
        # =====================================
        # subgradient
        # =====================================
        learner = Learner(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        
        weights = learner.learn(initial_weights)
        subgrad_weight_record = learner.weight_record
        subgrad_time_record = learner.time_record
        t = learner.time_record[0]
        subgrad_time = np.true_divide(np.array(subgrad_time_record) - t,1000)
        subgrad_accuracy_ave_train = np.zeros(len(subgrad_time))
        counter = 0
         
        for i in range(learner.num_examples):
            data = self.train_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(subgrad_weight_record.shape[0]):
                w = subgrad_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = self.Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
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
        
        for i in range(len(subgrad_accuracy_ave_train)):
            if i != len(subgrad_accuracy_ave_train) - 1 :
                assert (subgrad_accuracy_ave_train[i]<= subgrad_accuracy_ave_train[i+1]),"subgradient accuracy is not increasing"

        
        
        # =====================================
        # EM
        # =====================================
        learner = EM(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        weights = learner.learn(initial_weights)
         
        EM_weight_record = learner.weight_record
        EM_time_record = learner.time_record
        t = learner.time_record[0]

        EM_time = np.true_divide(np.array(EM_time_record) - t,1000)
        EM_accuracy_ave_train = np.zeros(len(EM_time))
        counter = 0
         
        for i in range(learner.num_examples):
            data = self.train_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(EM_weight_record.shape[0]):
                w = EM_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = self.Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
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
        
        for i in range(len(EM_accuracy_ave_train)):
            if i != len(EM_accuracy_ave_train) - 1 :
                assert (EM_accuracy_ave_train[i]<= EM_accuracy_ave_train[i+1]),"subgradient accuracy is not increasing"

        
        
        # =====================================
        # paired dual
        # =====================================
        learner = PairedDual(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        weights = learner.learn(initial_weights)
         
        paired_weight_record = learner.weight_record
        paired_time_record = learner.time_record
        t = learner.time_record[0]

        paired_time = np.true_divide(np.array(paired_time_record) - t,1000)
        paired_accuracy_ave_train = np.zeros(len(paired_time))
        counter = 0
         
        for i in range(learner.num_examples):
            data = self.train_data_dic[i]
            counter += 1
            accuracy = []
            for j in range(paired_weight_record.shape[0]):
                w = paired_weight_record[j,:]
                w_unary = np.reshape(w[0:d * num_states],(d, num_states)).T
                w_pair = np.reshape(w[num_states * d:],(num_states,num_states))
                w_unary = np.array(w_unary,dtype = float)
                w_pair = np.array(w_pair,dtype = float)
                mn = self.Create_MarkovNet(2, 2, w_unary, w_pair, data[1])
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
        
        for i in range(len(paired_accuracy_ave_train)):
            if i != len(paired_accuracy_ave_train) - 1 :
                assert (paired_accuracy_ave_train[i]<= paired_accuracy_ave_train[i+1]),"paire dual accuracy is not increasing"
        
    def test_precision_difference(self):
    # =====================================
    # first train by subgradient
    # =====================================
        d = 3
        num_states = 3
        num_pixels = 4
        model = self.Create_LogLinearModel(2,2,d,num_states)
        initial_weights = np.random.randn( num_states * d)
        initial_weights = np.append(initial_weights, np.random.randn( num_states * num_states))



        learner = Learner(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        subgrad_weights = learner.learn(initial_weights)
                
        
        learner = EM(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        EM_weights = learner.learn(subgrad_weights)
        
        assert (np.allclose(EM_weights, subgrad_weights)),"Model learned by subgrad is different from EM"

                
        learner = PairedDual(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        paired_weights = learner.learn(subgrad_weights)
        assert (np.allclose(paired_weights, subgrad_weights)),"Model learned by subgrad is different from paired dual"
        
    # =====================================
    # first train by EM
    # =====================================

        print '--------------------'
        learner = EM(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        EM_weights = learner.learn(initial_weights)
        
        
        learner = Learner(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        subgrad_weights = learner.learn(EM_weights)
        
        assert (np.allclose(EM_weights, subgrad_weights)),"Model learned by EM is different from subgrad"

        
        
        learner = PairedDual(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        paired_weights = learner.learn(EM_weights)
        assert (np.allclose(EM_weights, paired_weights)),"Model learned by EM is different from paired dual"

        
        # =====================================
        # first train by paired dual
        # =====================================

        print '--------------------'
        learner = PairedDual(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        paired_weights = learner.learn(initial_weights)
        
        
        
        learner = EM(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        EM_weights = learner.learn(paired_weights)
        assert (np.allclose(EM_weights, paired_weights)),"Model learned by paired dual is different from EM"

        
        learner = Learner(model,MatrixBeliefPropagator)
        learner.reset()
        self.add_synthetic_data(learner)
        subgrad_weights = learner.learn(paired_weights)
        assert (np.allclose(EM_weights, paired_weights)),"Model learned by paired dual is different from subgrad"

      
    train_data_dic = {}
    test_data_dic = {}
    
    def add_test_data(self):
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
        self.test_data_dic[0] = [dic1,dic2]
        
        
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
        self.test_data_dic[1] = [dic1,dic2]
        
        
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
        self.test_data_dic[2] = [dic1,dic2]
    
    def add_synthetic_data(self,learner):
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
        self.train_data_dic[0] = [dic1,dic2,dic3]
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
        self.train_data_dic[1] = [dic1,dic2,dic3]
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
        self.train_data_dic[2] = [dic1,dic2,dic3]
             
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
        self.train_data_dic[3] = [dic1,dic2,dic3]
    #     print ('data is added')
        learner.setRegularization(0, 0.25)
    
    
    
    # =====================================
    # Create Log linear model
    # =====================================
    
    def Create_LogLinearModel(self,height,width,d,numStates):
        num_pixels = height * width
        model = LogLinearModel()
    
        for i in range(1,num_pixels+1):
            model.declareVariable(i, numStates)
            model.setUnaryWeights(i,np.random.randn(numStates, d))
            model.setUnaryFeatures(i, np.random.randn(d))
    
    
        model.setAllUnaryFactors()
    
    #     ########### Set Edge Factor
        south_west_ind = num_pixels - width +1
            
        left_pixels = [i for i in range(1,num_pixels+1) if (i % width) == 1 and i != south_west_ind]
        down_pixels = range(south_west_ind+1 , num_pixels+1)
        usual_pixels = [i for i in range(1,num_pixels+1) if i not in left_pixels and i not in down_pixels and i not in [south_west_ind]]
        
        
        ##### set south neighbour for left pixels
        for i in (left_pixels):
            model.setEdgeFactor((i,i+width),np.eye(numStates))
             
    ##### set left neighbour for sought pixels
        for i in (down_pixels):
            model.setEdgeFactor((i-1,i),np.eye(numStates))
    
    ##### set south and left neighbour for all the remaining pixels
        for i in (usual_pixels):
            model.setEdgeFactor((i,i+width),np.eye(numStates))
            model.setEdgeFactor((i-1,i),np.eye(numStates))
            
        return model
        
    # =====================================
    # Create Markov Model
    # =====================================
    
    def Create_MarkovNet(self,height,width,w_unary,w_pair,pixels):
        mn = MarkovNet()
        np.random.seed(1)
        num_pixels = height * width
    
        ########Set Unary Factor
        for i in range(1,num_pixels+1):
            mn.setUnaryFactor(i,np.dot(w_unary,pixels[i]))
        
    #     ##########Set Pairwise
        south_west_ind = num_pixels - width +1
            
        left_pixels = [i for i in range(1,num_pixels+1) if (i % width) == 1 and i != south_west_ind]
        down_pixels = range(south_west_ind+1 , num_pixels+1)
        usual_pixels = [i for i in range(1,num_pixels+1) if i not in left_pixels and i not in down_pixels and i not in [south_west_ind]]
        
    ##### set south neighbour for left pixels
        for i in (left_pixels):
            mn.setEdgeFactor((i,i+width),w_pair)
             
    ##### set left neighbour for sought pixels
        for i in (down_pixels):
            mn.setEdgeFactor((i-1,i),w_pair)
    
    ##### set south and left neighbour for all the remaining pixels
        for i in (usual_pixels):
            mn.setEdgeFactor((i,i+width),w_pair)
            mn.setEdgeFactor((i-1,i),w_pair)
            
        return mn
            
            

if __name__ == '__main__':
    unittest.main()