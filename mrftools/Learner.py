import copy
import time
from _hashlib import new
import numpy as np
from scipy.optimize import minimize, check_grad
from LogLinearModel import LogLinearModel
from MatrixBeliefPropagator import MatrixBeliefPropagator
from ConvexBeliefPropagator import ConvexBeliefPropagator
from opt import *


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
        self.l2_regularization = 1
        self.weight_dim = None
        self.fully_observed = True
        self.initialization_flag = False
        self.loss_augmented = False
        self.MAP_Convex_inference = False

    def set_regularization(self, l1, l2):
        """Set the regularization parameters."""
        self.l1_regularization = l1
        self.l2_regularization = l2

    def instantiate(self, model):
        default_counting_numbers = dict ( )
        for var in model.variables:
            default_counting_numbers[var] = 0.15
            for neighbor in model.neighbors[var]:
                if var < neighbor:
                    default_counting_numbers[(var, neighbor)] = 0.15

        bp = ConvexBeliefPropagator ( model, default_counting_numbers )
        return bp

    def add_data(self, labels, model):
        """Add data example to training set. The states variable should be a dictionary containing all the states of the
         unary variables. Features should be a dictionary containing the feature vectors for the unary variables."""
        self.models.append(model)
        if self.MAP_Convex_inference == True:
            bp = self.instantiate(model)
        else:
            bp = self.inference_type(model)

        if self.loss_augmented == True:
            for (var, state) in labels.items ( ):
                bp.augment_loss(var, state)

        self.belief_propagators.append(bp)

        if self.weight_dim is None:
            self.weight_dim = model.weight_dim
        else:
            assert self.weight_dim == model.weight_dim, "Parameter dimensionality did not match"

        model_q = copy.deepcopy(model)
        self.models_q.append(model_q)

        if self.MAP_Convex_inference == True:
            bp_q = self.instantiate(model_q)
        else:
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
        if self.tau_q is None or not self.fully_observed:
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        return self.objective(weights)

    def subgrad_grad(self, weights, options=None):
        if self.tau_q is None or not self.fully_observed:
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, False)
        return self.gradient(weights)

    def learn(self, weights, callback_f=None):
        self.start = time.time()
        # res = adam(self.subgrad_obj, self.subgrad_grad, weights, args=None, callback=callback_f)
        res = ada_grad ( self.subgrad_obj, self.subgrad_grad, weights, args=None, callback=callback_f )
        new_weights = res
        # res = minimize(self.subgrad_obj, weights, method='L-BFGS-B', jac=self.subgrad_grad, callback=callback_f)
        # new_weights = res.x
        return new_weights

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

        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)

        term_p = sum([x.compute_energy_functional() for x in self.belief_propagators]) / len(self.belief_propagators)

        if not self.fully_observed:
            # recompute energy functional for label distributions only in latent variable case
            self.set_weights(weights, self.belief_propagators_q)
            term_q = sum([x.compute_energy_functional() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        else:
            term_q = np.dot(self.tau_q, weights)

        self.term_q_p = term_p - term_q


        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        return objec

    def gradient(self, weights, options=None):

        if time.time() - self.start > 60:
            print 'more than 60 sec'
            grad = np.zeros ( len ( weights ) )
            return grad
        else:

            self.tau_p = self.calculate_tau(weights, self.belief_propagators, False)

            grad = np.zeros(len(weights))

            # add regularization penalties
            grad += self.l1_regularization * np.sign(weights)
            grad += self.l2_regularization * weights

            grad -= np.squeeze(self.tau_q)
            grad += np.squeeze(self.tau_p)

            return grad

    def dual_obj(self, weights, options=None):
        if self.tau_q is None or not self.fully_observed:
            self.tau_q = self.calculate_tau(weights, self.belief_propagators_q, True)
        self.tau_p = self.calculate_tau(weights, self.belief_propagators, True)

        term_p = sum([x.compute_dual_objective() for x in self.belief_propagators]) / len(self.belief_propagators)
        term_q = sum([x.compute_dual_objective() for x in self.belief_propagators_q]) / len(self.belief_propagators_q)
        # term_q = np.dot(self.tau_q, weights)

        self.term_q_p = term_p - term_q

        objec = 0.0
        # add regularization penalties
        objec += self.l1_regularization * np.sum(np.abs(weights))
        objec += 0.5 * self.l2_regularization * weights.dot(weights)
        objec += self.term_q_p

        return objec