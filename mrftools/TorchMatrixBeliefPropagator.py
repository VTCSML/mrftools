"""BeliefPropagator class."""
# import numpy as np
from Inference import Inference
import torch as t
from torch.autograd import Variable
'''
https://github.com/pytorch/pytorch/issues/1668
if torch.cuda.available():
    import torch.cuda as t
else:
    import torch as t
'''

class TorchMatrixBeliefPropagator(Inference):
    """
    Object that can run belief propagation on a MarkovNet. Uses sparse matrices to encode the
    indexing underlying belief propagation.
    """

    def __init__(self, markov_net, is_cuda, var_on):
        """
        Initialize belief propagator for markov_net.

        :param markov_net: Markov net
        :type markov_net: MarkovNet object encoding the probability distribution
        """
        self.mn = markov_net
        self.is_cuda = is_cuda
        self.var_on = var_on
        self.var_beliefs = dict()
        self.pair_beliefs = dict()

        # if the MarkovNet has not created its matrix representation of structure, create it
        if not self.mn.matrix_mode:
            self.mn.create_matrices()

        self.previously_initialized = False
        self.message_mat = None
        self.orig_message_mat = None
        self.initialize_messages()

        self.belief_mat = t.DoubleTensor(self.mn.max_states, len(self.mn.variables)).zero_()
        if self.var_on:
            self.belief_mat = Variable(t.DoubleTensor(self.mn.max_states, len(self.mn.variables)).zero_(),
                                       requires_grad=self.var_on)
        if self.is_cuda:
            self.belief_mat = self.belief_mat.cuda()


        self.pair_belief_tensor = t.DoubleTensor(self.mn.max_states, self.mn.max_states, self.mn.num_edges).zero_()
        if self.var_on:
            self.pair_belief_tensor = Variable(
                t.DoubleTensor(self.mn.max_states, self.mn.max_states, self.mn.num_edges).zero_(),
                requires_grad=self.var_on)
        if self.is_cuda:
            self.pair_belief_tensor = self.pair_belief_tensor.cuda()

        self.max_iter = 300  # default maximum iterations

        # the augmented_mat is used to condition variables or for loss-augmented inference for max-margin learning
        self.augmented_mat = t.DoubleTensor(self.mn.max_states, len(self.mn.variables)).zero_()
        if self.var_on:
            self.augmented_mat = Variable(t.DoubleTensor(self.mn.max_states, len(self.mn.variables)).zero_(),
                                          requires_grad=self.var_on)
        if self.is_cuda:
            self.augmented_mat = self.augmented_mat.cuda()

        self.fully_conditioned = False  # true if every variable has been conditioned

        # conditioned stores the indices of variables that have been conditioned, initialized to all False
        self.conditioned = t.ByteTensor(len(self.mn.variables)).zero_()
        if self.is_cuda:
            self.conditioned = self.conditioned.cuda()

        # condition variables so they can't be in states greater than their cardinality
        self.disallow_impossible_states()

    def set_max_iter(self, max_iter):
        """
        Set the maximum iterations of belief propagation to run before early stopping
        :param max_iter: integer maximum iterations
        :return: None
        """
        self.max_iter = max_iter

    def initialize_messages(self):
        """
        Initialize messages to default initialization (set to zeros).

        :return: None
        """
        self.message_mat = t.DoubleTensor(self.mn.max_states, 2 * self.mn.num_edges).zero_()
        if self.var_on:
            self.message_mat = Variable(t.DoubleTensor(self.mn.max_states, 2 * self.mn.num_edges).zero_(),
                                        requires_grad=self.var_on)
        if self.is_cuda:
            self.message_mat = self.message_mat.cuda()

    def augment_loss(self, var, state):
        """
        Adds a loss penalty to the MRF energy function. Used to create loss-augmented inference for max-margin learning

        :param var: variable to add loss to
        :type var: object
        :param state: state of variable in ground truth labels
        :type state: int
        :return: None
        """
        i = self.mn.var_index[var]
        self.augmented_mat[:][i] = 1
        for s in state:
            self.augmented_mat[s, i] = 0

    def condition(self, var, state):
        """
        Condition a variable, usually because of statistical evidence, to be in a subset of its possible states

        :param var: variable to condition
        :type var: object
        :param state: state to condition to or array of states that the variable may be in
        :type state: int or array
        :return: None
        """
        i = self.mn.var_index[var]
        self.augmented_mat[:, i] = -float('inf')

        if not hasattr(state, "__iter__"):
            state = [state]
        if self.is_cuda:
            for s in t.LongTensor(state).cuda():
                self.augmented_mat[s, i] = 0
        else:
            for s in state:
                self.augmented_mat[s, i] = 0
        if isinstance(state, int):
            # only if the variable is fully conditioned to be in a single state, mark that the variable is conditioned
            self.conditioned[i] = True

        if self.conditioned.all():
            # compute beliefs and set flag to never recompute them
            self.compute_beliefs()
            self.compute_pairwise_beliefs()
            self.fully_conditioned = True

    def disallow_impossible_states(self):
        """
        Force variables to only allow nonzero probability on their possible states.

        :return: None
        """
        for var, num_states in self.mn.num_states.items():
            self.condition(var, range(num_states))

    def compute_beliefs(self):
        """
        Compute unary log beliefs based on current messages and store them in belief_mat

        :return: None
        """
        if not self.fully_conditioned:
            self.belief_mat = self.mn.unary_mat + self.augmented_mat
            self.belief_mat += sparse_dot(self.message_mat, self.mn.message_to_map)

            self.belief_mat -= torch_logsumexp(self.belief_mat, 0)

    def compute_pairwise_beliefs(self):
        """
        Compute pairwise log beliefs based on current messages, and stores them in pair_belief_tensor

        :return: None
        """
        if not self.fully_conditioned:
            adjusted_message_prod = self.belief_mat[:, self.mn.message_from] \
                                    - t.cat((self.message_mat[:, self.mn.num_edges:],
                                                 self.message_mat[:, :self.mn.num_edges]), 1)

            # Have to convert NaNs to negative infinity at this point
            adjusted_message_prod = torch_nan_to_neginf(adjusted_message_prod)

            # Have to make contiguous before reshaping
            to_messages = adjusted_message_prod[:, :self.mn.num_edges].contiguous().view(
                (self.mn.max_states, 1, self.mn.num_edges))
            from_messages = adjusted_message_prod[:, self.mn.num_edges:].contiguous().view(
                (1, self.mn.max_states, self.mn.num_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, self.mn.num_edges:] + to_messages + from_messages

            beliefs -= torch_logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor = beliefs

    def update_messages(self):
        """
        Update all messages between variables and store them in message_mat

        :return: the float change in messages from previous iteration.
        """
        self.compute_beliefs()

        # Using the beliefs as the sum of all incoming log messages, subtract the outgoing messages and add the edge
        # potential.
        adjusted_message_prod = self.mn.edge_pot_tensor + self.belief_mat[:, self.mn.message_from] \
                                - t.cat((self.message_mat[:, self.mn.num_edges:],
                                             self.message_mat[:, :self.mn.num_edges]), 1)

        # Have to convert NaNs to negative infinity at this point
        adjusted_message_prod = torch_nan_to_neginf(adjusted_message_prod)

        messages = t.squeeze(torch_logsumexp(adjusted_message_prod, 1))
        messages = torch_nan_to_zero(messages - t.max(messages, 0)[0])

        change = t.sum(torch_nan_to_zero(t.abs(messages - self.message_mat)))

        self.message_mat = messages

        return change

    def _compute_inconsistency_vector(self):
        """
        Compute the vector of inconsistencies between unary beliefs and pairwise beliefs
        :return: Vector of inconsistencies
        :rtype: array
        """
        expanded_beliefs = t.exp(self.belief_mat[:, self.mn.message_to])

        pairwise_beliefs = t.cat((t.sum(t.exp(self.pair_belief_tensor), 0),
                                      t.sum(t.exp(self.pair_belief_tensor), 1)), 1)

        return expanded_beliefs - pairwise_beliefs

    def compute_inconsistency(self):
        """
        Return the total disagreement between each unary belief and its pairwise beliefs.
        When message passing converges, the inconsistency should be within numerical error.

        :return: the total absolute disagreement between each unary belief and its pairwise beliefs
        """
        disagreement = t.sum(t.abs(self._compute_inconsistency_vector()))

        return disagreement

    # WARNING: Torch has some weird significant figure issues not going past about 1e-6
    #   If left at 1e-8 otherwise, it will go for max iterations despite being 'done'
    def infer(self, tolerance=1e-8, display='iter'):
        """
        Run belief propagation until messages change less than tolerance.

        :param tolerance: the minimum amount that the messages can change while message passing can be considered not
                            converged
        :param display: string parameter indicating how much to display. Options are 'full' and 'iter'
                        'full' prints the energy functional and dual objective each iteration,
                                which requires extra computation
                        'iter' prints just the change in messages each iteration
        :return: None
        """
        change = float('inf')
        iteration = 0
        while change > tolerance and iteration < self.max_iter:
            change = self.update_messages()
            if display == "full":
                energy_func = self.compute_energy_functional()
                disagreement = self.compute_inconsistency()
                dual_obj = self.compute_dual_objective()
                print("Iteration %d, change in messages %f. Calibration disagreement: %f, "
                      "energy functional: %f, dual obj: %f" % (iteration, change, disagreement, energy_func, dual_obj))
            elif display == "iter":
                print("Iteration %d, change in messages %f." % (iteration, change))
            iteration += 1
        if display == 'final' or display == 'full' or display == 'iter':
            print("Belief propagation finished in %d iterations." % iteration)

    def load_beliefs(self):
        """
        Update the belief dictionaries var_beliefs and pair_beliefs using the current messages.

        :return: None
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        for (var, i) in self.mn.var_index.items():
            self.var_beliefs[var] = self.belief_mat[:len(self.mn.unary_potentials[var]), i]

        for edge, i in self.mn.message_index.items():
            (var, neighbor) = edge

            belief = self.pair_belief_tensor[:len(self.mn.unary_potentials[var]),
                     :len(self.mn.unary_potentials[neighbor]), i]

            self.pair_beliefs[(var, neighbor)] = belief

            self.pair_beliefs[(neighbor, var)] = belief.t()

    def compute_bethe_entropy(self):
        """
        Compute Bethe entropy from current beliefs.
        This method assumes that the beliefs have been computed and are fresh.

        :return: computed Bethe entropy
        """
        if self.fully_conditioned:
            entropy = 0
        else:
            entropy = - t.sum(torch_nan_to_zero(torch_nan_to_zero(self.pair_belief_tensor) * t.exp(self.pair_belief_tensor))) \
                      - t.sum(torch_nan_to_zero(t.mul(t.mul((1 - self.mn.degrees), torch_nan_to_zero(self.belief_mat)), t.exp(self.belief_mat))))

        return entropy

    def compute_energy(self):
        """
        Compute the log-linear energy. Assume that the beliefs have been computed and are fresh.

        :return: computed energy
        """
        energy = t.sum(torch_nan_to_zero(
            t.mul(torch_nan_to_zero(self.mn.edge_pot_tensor[:, :, self.mn.num_edges:]), t.exp(self.pair_belief_tensor))
        )) + t.sum(torch_nan_to_zero(
            t.mul(torch_nan_to_zero(self.mn.unary_mat), t.exp(self.belief_mat))
        ))

        return energy

    # Currently don't care about this since it's a LogLinearModel
    '''
    def get_feature_expectations(self):
        """
        Computes the feature expectations under the currently estimated marginal probabilities. Only works when the
        model is a LogLinearModel class with features for edges.

        :return: vector of the marginals in order of the flattened unary features first, then the flattened pairwise
                    features
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()

        summed_features = self.mn.unary_feature_mat.dot(np.exp(self.belief_mat).T)

        summed_pair_features = self.mn.edge_feature_mat.dot(np.exp(self.pair_belief_tensor).reshape(
            (self.mn.max_states ** 2, self.mn.num_edges)).T)

        marginals = np.append(summed_features.reshape(-1), summed_pair_features.reshape(-1))

        return marginals
    '''

    def compute_energy_functional(self):
        """
        Compute the energy functional, which is the variational approximation of the log-partition function.

        :return: computed energy functional
        """
        self.compute_beliefs()
        self.compute_pairwise_beliefs()
        return self.compute_energy() + self.compute_bethe_entropy()

    def compute_dual_objective(self):
        """
        Compute the value of the BP Lagrangian.

        :return: Lagrangian objective function
        """
        objective = self.compute_energy_functional() + \
                    t.sum(torch_nan_to_zero(t.mul(self.message_mat, self._compute_inconsistency_vector())))

        return objective

    def set_messages(self, messages):
        """
        Set the message vector. Useful for warm-starting inference if a previously computed message matrix is available.

        :param messages: message matrix
        :type messages: ndarray
        :return: None
        """
        assert t.eq(self.message_mat.size(), messages.size())
        self.message_mat = messages

    def autograd_unary(self):
        """
        Computes the backward pass over the network and returns the gradient of the unary beliefs
        :return: gradient of unary beliefs
        """
        if self.var_on:
            self.belief_mat.backward()
            return self.belief_mat.grad
        else:
            print("Variables flag (var_on) not set to True to allow for autograd")
            return

    def autograd_pairwise(self):
        """
        Computes the backward pass over the network and returns the gradient of the pairwise beliefs
        :return: gradient of pairwise beliefs
        """
        if self.var_on:
            self.pair_beliefs.backward()
            return self.pair_beliefs.grad
        else:
            print("Variables flag (var_on) not set to True to allow for autograd")
            return

def torch_logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.

    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of log(sum(exp(matrix), dim)))
    :rtype: ndarray
    """
    # max_val = torch_nan_to_num(matrix.max(dim=dim[0], keepdim=True)[0].max(dim=dim[1], keepdim=True)[0])
    # t.log(t.exp(matrix - max_val).sum(dim=dim[0], keepdim=True).sum(dim=dim[1], keepdim=True)) + max_val
    if not hasattr(dim, "__iter__"):
        max_val = torch_nan_to_num(t.max(matrix, dim=dim, keepdim=True)[0])
        ret_val = t.log(t.sum(t.exp(matrix - max_val), dim=dim, keepdim=True)) + max_val
        return ret_val
    else:
        max_val = matrix
        for d in dim:
            max_val = max_val.max(dim=d, keepdim=True)[0]
        exp_val = t.exp(matrix - max_val)
        for d in dim:
            exp_val = exp_val.sum(dim=d, keepdim=True)
        return t.log(exp_val) + max_val


def sparse_dot(full_matrix, sparse_matrix):
    """
    Convenience function to compute the dot product of a full matrix and a sparse matrix.
    Useful to avoid writing code with a lot of transposes.

    ISSUE: Sparse operations on Variables not yet supported (is supported for Tensors)
    https://github.com/pytorch/pytorch/issues/2389

    :param full_matrix: dense matrix
    :type full_matrix: ndarray
    :param sparse_matrix: sparse matrix
    :type sparse_matrix: t.sparse.DoubleTensor
    :return: full_matrix.dot(sparse_matrix)
    :rtype: ndarray
    """
    return t.mm(sparse_matrix.t(), full_matrix.t()).t()


def torch_nan_to_zero(mat):
    """
    Replaces all NaNs in a Tensor with 0's since Torch doesn't have such a function. This is done easily because NaN != NaN
    :param mat: matrix to replace NaN in
    :type mat: t.Tensor
    :return: matrix with replaced 0's
    :rtype: t.Tensor
    """
    new_mat = mat
    new_mat[new_mat != new_mat] = 0
    return new_mat

def torch_nan_to_neginf(mat):
    """
    Replaces all NaNs in a Tensor with -infinity since Torch doesn't have such a function. This is done easily because NaN != NaN
    :param mat: matrix to replace NaN in
    :type mat: t.Tensor
    :return: matrix with replaced -infinity's
    :rtype: t.Tensor
    """
    new_mat = mat
    new_mat[new_mat != new_mat] = -float('inf')
    return new_mat

def torch_nan_to_num(mat):
    """
    Replaces all NaNs in a Tensor with -infinity since Torch doesn't have such a function. This is done easily because NaN != NaN
    :param mat: matrix to replace NaN in
    :type mat: t.Tensor
    :return: matrix with replaced -infinity's
    :rtype: t.Tensor
    """
    new_mat = mat
    new_mat[new_mat != new_mat] = 0
    new_mat = t.clamp(new_mat, min=-1.79769313e+308, max=1.79769313e+308)
    return new_mat
