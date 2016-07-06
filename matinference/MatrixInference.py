import numpy as np
import logging
from scipy.sparse import dok_matrix, csc_matrix
from Model import Model

logger = logging.getLogger(__name__)


class MatrixInference(object):
    """
    Class that performs dual inference with overcounting-number entropies using matrix operations whenever possible.
    """

    def __init__(self, model):
        self.model = model
        self.norm_mat = None
        self.cons_left = None
        self.cons_right = None
        self.cons_dual_vars = None
        self.norm_dual_vars = None
        self.log_beliefs = np.zeros((model.num_marginals, 1))
        self.counting_nums = None
        self.counting_nums_norm = None
        self.counting_nums_left = None
        self.counting_nums_right = None
        self.counting_nums_marg = None
        self.create_matrices()
        self.init_variables()

    def create_matrices(self):
        """
        Creates sparse constraint matrices.
        :return: None
        """
        num_tables = len(self.model.vars) + len(self.model.edges)
        self.norm_mat = dok_matrix((num_tables, self.model.num_marginals))
        row = 0
        for factor in self.model.vars + self.model.edges:
            self.norm_mat[row, self.model.belief_indices[factor]] = 1
            row += 1

        num_cons = len(self.model.edges) * self.model.num_states * 2

        if num_cons > 0:
            self.cons_left = dok_matrix((num_cons, self.model.num_marginals))
            self.cons_right = dok_matrix((num_cons, self.model.num_marginals))
        else:
            self.cons_left = np.zeros((0, self.model.num_marginals))
            self.cons_right = np.zeros((0, self.model.num_marginals))

        row = 0
        for edge in self.model.edges:
            (u, v) = edge
            pair_index_mat = np.reshape(self.model.belief_indices[edge], (self.model.num_states, self.model.num_states),
                                        order='F')
            u_indices = self.model.belief_indices[u]
            v_indices = self.model.belief_indices[v]
            for i in range(self.model.num_states):
                self.cons_right[row, u_indices[i]] = 1
                self.cons_left[row, pair_index_mat[i, :]] = 1
                row += 1

            for i in range(self.model.num_states):
                self.cons_right[row, v_indices[i]] = 1
                self.cons_left[row, pair_index_mat[:, i]] = 1
                row += 1

        if num_cons > 0:
            self.cons_left = csc_matrix(self.cons_left)
            self.cons_right = csc_matrix(self.cons_right)
        self.norm_mat = csc_matrix(self.norm_mat)

    def init_variables(self):
        self.norm_dual_vars = np.zeros((self.norm_mat.shape[0], 1))
        self.cons_dual_vars = np.zeros((self.cons_left.shape[0], 1))

    def set_counting_nums(self, counting_nums=None):
        """
        Sets counting numbers for each factor. If counting_nums are not provided, sets the Bethe counting numbers
        :param counting_nums: Dictionary containing a counting number for each factor (variables and edges) in model
        :return: None
        """

        if counting_nums is None:
            counting_nums = self.bethe_counting_nums()

        assert len(counting_nums) == len(self.model.vars) + len(self.model.edges)

        self.counting_nums = np.zeros((len(counting_nums), 1))

        i = 0
        for key in self.model.vars + self.model.edges:
            self.counting_nums[i] = counting_nums[key]
            i += 1

        self.counting_nums_marg = np.asarray(self.model.factor_mat.dot(self.counting_nums))
        self.counting_nums_norm = np.asarray(self.norm_mat.dot(self.counting_nums_marg) / self.norm_mat.sum(1))
        self.counting_nums_left = np.asarray(self.cons_left.dot(self.counting_nums_marg) / self.cons_left.sum(1))
        self.counting_nums_right = np.asarray(self.cons_right.dot(self.counting_nums_marg) / self.cons_right.sum(1))

    def bethe_counting_nums(self):
        counting_nums = dict()
        for edge in self.model.edges:
            counting_nums[edge] = 1

        for var in self.model.vars:
            degree = len(self.model.neighbors[var])
            counting_nums[var] = 1# - degree
        return counting_nums

    def update_beliefs(self):
        """
        Updates beliefs to closed form solution given current constraint dual variables. Also updates normalization
        dual variables so beliefs are normalized
        :return: None
        """
        self.log_beliefs = (self.model.template_mat.T.dot(self.model.weights) +
                            self.norm_mat.T.dot(self.norm_dual_vars) +
                            (self.cons_left - self.cons_right).T.dot(self.cons_dual_vars)) / self.counting_nums_marg - 1
        self.log_beliefs = np.asarray(self.log_beliefs)

    def update_dual_vars(self):
        """
        Perform a block update on all dual variables
        :return: None
        """
        if self.cons_dual_vars.size > 0:
            right_coeff = self.cons_right.copy()
            adjusted_dual_vars = self.cons_dual_vars / self.counting_nums_right
            right_coeff.data = np.exp(right_coeff.data * adjusted_dual_vars.take(right_coeff.indices))

            left_coeff = self.cons_left.copy()
            adjusted_dual_vars = self.cons_dual_vars / self.counting_nums_left
            left_coeff.data = np.exp(-left_coeff.data * adjusted_dual_vars.take(left_coeff.indices))

            self.cons_dual_vars = (logdotexp(right_coeff, self.log_beliefs) - logdotexp(left_coeff, self.log_beliefs)) \
                / (1 / self.counting_nums_left + 1 / self.counting_nums_right)

        unnormalized_beliefs = (self.model.template_mat.T.dot(self.model.weights) +
                                (self.cons_left - self.cons_right).T.dot(
                                    self.cons_dual_vars)) / self.counting_nums_marg - 1
        self.norm_dual_vars = - self.counting_nums_norm * logdotexp(self.norm_mat, np.asarray(unnormalized_beliefs))
        self.update_beliefs()

    def dual_objective(self):
        """
        :return: current dual objective, with closed-form values for beliefs and norm_dual_vars
        """
        return self.energy() + self.entropy() + self.cons_dual_vars.T.dot(self.calibration_vec()).sum()

    def entropy(self):
        """
        :return: current entropy, with closed-form values for beliefs and norm_dual_vars
        """
        return - np.exp(self.log_beliefs).T.dot(self.counting_nums_marg * self.log_beliefs)

    def energy(self):
        """
        :return: current energy, with closed-form values for beliefs and norm_dual_vars
        """
        return self.model.template_mat.T.dot(self.model.weights).T.dot(np.exp(self.log_beliefs))

    def calibration_vec(self):
        return (self.cons_left - self.cons_right).dot(np.exp(self.log_beliefs))

    def infer(self, max_iter=300, tolerance=1e-6, damping=0.0):
        """
        Run iterative inference optimization
        :param max_iter: maximum number of dual updates to run before giving up
        :param tolerance: absolute change in dual variables required to consider converged
        :param damping: ratio of previous dual variables to use for constraint variables
        :return: None
        """
        iter = 0
        change = np.inf
        while iter < max_iter and change >= tolerance:
            old_dual_vars = self.cons_dual_vars
            self.update_dual_vars()
            self.cons_dual_vars = damping * old_dual_vars + (1 - damping) * self.cons_dual_vars

            change = np.sum(np.abs(old_dual_vars - self.cons_dual_vars))

            logger.debug("Inference iteration %d. Dual objective %f (energy %f, entropy %f)" % \
                         (iter, self.dual_objective(), self.energy(), self.entropy()))
            iter += 1


    def get_beliefs(self, key, state=None):
        """
        Gets the current estimated beliefs for either a variable or an edge
        :param key: variable name or edge pair.
        :param state: Variable integer state or edge state pair, or None to get full belief vector or matrix
        :return: belief of the variable or edge at the state, or the vector or matrix if state is not provided
        """

        if key in self.model.var_index:
            # unary marginal
            belief_vec = self.log_beliefs[self.model.belief_indices[key]]
            if state is None:
                return belief_vec
            else:
                return belief_vec[state]
        else:
            # pairwise marginal
            if key not in self.model.edge_index:
                belief_vec = self.log_beliefs[self.model.belief_indices[(key[1], key[0])]]
                transpose = True
            else:
                belief_vec = self.log_beliefs[self.model.belief_indices[key]]
                transpose = False

            belief_mat = belief_vec.reshape((self.model.num_states, self.model.num_states), order='F')

            if transpose:
                belief_mat = belief_mat.T

            if state is None:
                return belief_mat
            else:
                return belief_mat[state[0], state[1]]


    def get_unary_mat(self):
        """
        :return: num_states by num_vars unary belief matrix
        """
        return self.log_beliefs[:self.model.num_states * len(self.model.vars)].reshape(
            (self.model.num_states, len(self.model.vars)), order='F')

    def get_pair_tensor(self):
        """
        :return: num_states by num_states by num_edges pair belief tensor
        """
        return self.log_beliefs[self.model.num_states * len(self.model.vars):].reshape(
            (self.model.num_states, self.model.num_states, len(self.model.edges)), order='F')

    def get_expectations(self):
        """
        :return: returns the current expectations of the aggregated feature functions
        """
        return self.model.template_mat.dot(self.log_beliefs)

def logdotexp(coeff, exponent):
    max_val = exponent.max(0, keepdims=True)
    return (np.log(coeff.dot(np.exp(exponent - max_val))) + max_val)

