import copy
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
import math
from opt import ada_grad, ada_grad_1
import time
import lbfgs
import random


class Learner(object):
    def __init__(self, inference_type):
        self.tau_q = None
        self.tau_p = None
        self.weight_record = np.array([])
        self.time_record = np.array([])
        self.inference_type = inference_type
        self.num_examples = 0
        self.models = []
        self.models_q = []
        self.belief_propagators_q = []
        self.belief_propagators = []
        self.l1_regularization = 0.00
        self.l2_regularization = 0
        self.weight_dim = None
        self.fully_observed = True
        self.initialization_flag = False
        self.edges_group_regularizers = 0
        self.var_group_regularizers = 0
        self.edge_regularizers = dict()
        self.var_regularizers = dict()
        self.graft = False
        self.sufficent_stats = None
        self.is_initialize_grad_sum = True


    def set_regularization(self, l1, l2, var_reg, edge_reg):
        """Set the regularization parameters."""
        self.l1_regularization = l1
        self.l2_regularization = l2
        self.edges_group_regularizers = edge_reg
        self.var_group_regularizers = var_reg

    def init_model(self, model):

        self.models.append(model)
        self.belief_propagators.append(self.inference_type(model))
        if self.weight_dim == None:
            self.weight_dim = model.weight_dim
        else:
            assert self.weight_dim == model.weight_dim, "Parameter dimensionality did not match"
        model_q = copy.deepcopy(model)
        bp_q = self.inference_type(model_q)
        bp_q.graft_condition()

        self.belief_propagators_q.append(bp_q)



    def add_data(self, labels, model):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the
         unary viables. Features should be a dictionary containing the feature vectors for the unary variables."""
        self.models.append(model)
        self.belief_propagators.append(self.inference_type(model))

        if self.weight_dim == None:
            self.weight_dim = model.weight_dim
        else:
            assert self.weight_dim == model.weight_dim, "Parameter dimensionality did not match"

        model_q = copy.deepcopy(model)
        self.models_q.append(model_q)

        bp_q = self.inference_type(model_q)
        for (var, state) in labels.items():
            bp_q.condition(var, state)

        for var in model_q.variables:
            if var not in labels.keys():
                self.fully_observed = False

        self.belief_propagators_q.append(bp_q)

        self.num_examples += 1

    def _set_initialization_flag(self, flag):
        self.initialization_flag = flag

    def do_inference(self, belief_propagators):
        for bp in belief_propagators:
            if self.initialization_flag == True:
                bp.initialize_messages()
            bp.infer(display = 'off')

    def set_inference_truncation(self, bp_iter):
        for bp in self.belief_propagators + self.belief_propagators_q:
            bp.set_max_iter(bp_iter)

    def get_feature_expectations(self, belief_propagators):
        """Run inference and return the marginal in vector form using the order of self.potentials.
        """
        marginal_sum = 0
        for bp in belief_propagators:
            marginal_sum += bp.get_feature_expectations()
        return marginal_sum / len(belief_propagators)

    def get_bethe_entropy(self, belief_propagators):
        bethe = 0
        for bp in belief_propagators:
            bp.compute_beliefs()
            bp.compute_pairwise_beliefs()
            bethe += bp.compute_bethe_entropy()

        bethe = bethe / self.num_examples
        return bethe

    def subgrad_obj(self, weights, options=None):
        # if self.feature_graft:
        #     weights[self.zero_feature_indices] = 0
        if (self.tau_q is None or not self.fully_observed) and not (self.graft):
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        return self.objective(weights)

    def subgrad_obj_dual(self, weights, options=None):
        # if self.feature_graft:
        #     weights[self.zero_feature_indices] = 0
        if (self.tau_q is None or not self.fully_observed) and not (self.graft):
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        return self.objective_dual(weights)

    def subgrad_grad(self, weights, options=None):
        # if self.feature_graft:
            # weights[self.zero_feature_indices] = 0
        if (self.tau_q is None or not self.fully_observed) and not (self.graft):
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, False)
        return self.gradient(weights)

    def function_owlqn(self, x ,g):
        obj = self.objective(x, options=None)
        g[:] = self.subgrad_grad(x)
        return obj

    def learn(self, weights, max_iter, edge_regularizers, var_regularizers, data, verbose=False, callback_f=None, loss=None):
        self.edge_regularizers = edge_regularizers
        self.var_regularizers = var_regularizers
        self.data = data
        self.len_data = len(data)

        if self.is_initialize_grad_sum == True:
            self.grad_sum = np.zeros(len(weights))
            self.is_initialize_grad_sum = False

        res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=self.subgrad_grad, callback=callback_f, options={'maxiter': max_iter, 'gtol': 1e-15, 'ftol': 1e-15})
        new_weights = res.x
        if loss is not None:
            loss.append(self.objec)
        print(self.objec)
        t = 0
        if verbose:
            print('Iter')
            print(res.nit)
            print('opt. message')
            print(res.message)
            print(np.sqrt(res.jac.dot(res.jac)))
            print('obj')
            print(res.fun)

        return new_weights, t

    def reset(self):
        self.weight_record =  np.array([])
        self.time_record = np.array([])
        for bp in self.belief_propagators + self.belief_propagators_q:
            bp.initialize_messages()

    def set_weights(self, weight_vector, belief_propagators):
        """Set weights of Markov net from vector using the order in self.potentials."""
        for bp in belief_propagators:
            bp.mn.set_weights(weight_vector)

    def calculate_tau(self, weights, belief_propagators, should_infer=True):
        self.set_weights(weights, belief_propagators)
        if should_infer:
            self.do_inference(belief_propagators)

        return self.get_feature_expectations(belief_propagators)

    def objective(self, weights, options=None):
        # if self.feature_graft:
        #     weights[self.zero_feature_indices] = 0
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)
        term_p = sum([x.compute_energy_functional() for x in self.belief_propagators]) / len(self.belief_propagators)
        if not self.fully_observed:
            # recompute energy functional for label distributions only in latent variable case
            self.set_weights(weights, self.belief_propagators_q)
            term_q = sum([x.compute_energy_functional() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        else:
            term_q = np.dot(self.tau_q, weights)

        self.term_q_p = term_p - term_q # tau_q : DATA EXPECTATION
        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        for edge in self.edge_regularizers.keys():
            curr_reg = np.zeros(len(weights))
            curr_reg[self.edge_regularizers[edge]] = weights[self.edge_regularizers[edge]]

            length_normalizer = np.sqrt(len(self.edge_regularizers[edge]))

            objec += length_normalizer * self.edges_group_regularizers * np.sqrt(curr_reg.dot(curr_reg))

        for var in self.var_regularizers.keys():
            curr_reg = np.zeros(len(weights))
            curr_reg[self.var_regularizers[var]] = weights[self.var_regularizers[var]]

            length_normalizer = np.sqrt(len(self.var_regularizers[var]))

            objec += length_normalizer * self.var_group_regularizers * np.sqrt(curr_reg.dot(curr_reg))

        self.objec = objec
        return objec

    def gradient(self, weights, options=None):

        # if self.feature_graft:
        #     weights[self.zero_feature_indices] = 0
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, False) 
        grad = np.zeros(len(weights))
        grad += self.l1_regularization * np.sign(weights)
        grad += self.l2_regularization * weights
        grad -= np.squeeze(self.tau_q)
        grad += np.squeeze(self.tau_p)
        grad_reg = np.zeros(len(weights))

        for edge in self.edge_regularizers.keys():
            curr_reg = np.zeros(len(weights))
            curr_reg[self.edge_regularizers[edge]] = weights[self.edge_regularizers[edge]]
            length_normalizer = np.sqrt(len(self.edge_regularizers[edge]))
            edge_norm = np.sqrt(curr_reg.dot(curr_reg))
            edge_grad = copy.deepcopy(grad[list(self.edge_regularizers[edge].flatten())])
            edge_grad_norm = np.sqrt(edge_grad.dot(edge_grad))
            edge_reg = length_normalizer * self.edges_group_regularizers * (curr_reg / (edge_norm + 1e-10))
            grad_reg += edge_reg


        for var in self.var_regularizers.keys():
            curr_reg = np.zeros(len(weights))
            curr_reg[self.var_regularizers[var]] = weights[self.var_regularizers[var]]
            length_normalizer = np.sqrt(len(self.var_regularizers[var]))
            var_norm = np.sqrt(curr_reg.dot(curr_reg))
            var_grad = copy.deepcopy(grad[self.var_regularizers[var]])
            var_grad_norm = np.sqrt(var_grad.dot(var_grad))
            grad_reg += length_normalizer * self.var_group_regularizers * (curr_reg / (var_norm + 1e-10)) #IF NORM IS ZERO

        grad += grad_reg
        self.grad = grad

        return grad

    def dual_obj(self, weights, options=None):
        # if self.feature_graft:
        #     weights[self.zero_feature_indices] = 0
        self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        if self.tau_q == None or not self.fully_observed:
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)

        term_p = sum([x.compute_dual_objective() for x in self.belief_propagators]) / len(self.belief_propagators)
        term_q = sum([x.compute_dual_objective() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        term_q = np.dot(self.tau_q, weights)

        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        for edge in self.edge_regularizers.keys():
            curr_reg = np.zeros(len(weights))
            curr_reg[self.edge_regularizers[edge]] = weights[self.edge_regularizers[edge]]
            # length_normalizer = float(1) / ( len(self.belief_propagators[0].mn.unary_potentials[edge[0]]) * len(self.belief_propagators[0].mn.unary_potentials[edge[1]] ))
            length_normalizer = np.sqrt(len(self.belief_propagators[0].mn.unary_potentials[edge[0]]) * len(self.belief_propagators[0].mn.unary_potentials[edge[1]]))
            objec += length_normalizer * self.edges_group_regularizers * np.sqrt(curr_reg.dot(curr_reg))

        for var in self.var_regularizers.keys():
            curr_reg = np.zeros(len(weights))
            curr_reg[self.var_regularizers[var]] = weights[self.var_regularizers[var]]
            # length_normalizer = float(1) / len(self.belief_propagators[0].mn.unary_potentials[var])
            length_normalizer = np.sqrt(len(self.belief_propagators[0].mn.unary_potentials[var]))
            objec += length_normalizer * self.var_group_regularizers * np.sqrt(curr_reg.dot(curr_reg))

        return objec

    def set_sufficient_stats(self, stats):
        self.graft = True
        # self.set_weights(weights, belief_propagators)
        # self.do_inference(self.belief_propagators)
        self.tau_q = stats


    def logsumexp(self, matrix, dim = None):
        """Compute log(sum(exp(matrix), dim)) in a numerically stable way."""
        try:
            with np.errstate(over='raise', under='raise'):
                return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
        except:
            max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
            with np.errstate(under='ignore', divide='ignore'):
                return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val
