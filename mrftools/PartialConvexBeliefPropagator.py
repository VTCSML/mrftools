import numpy as np
from scipy.sparse import csc_matrix
import random

from .MatrixBeliefPropagator import MatrixBeliefPropagator, logsumexp, sparse_dot
from .ConvexBeliefPropagator import ConvexBeliefPropagator

class PartialConvexBeliefPropagator(ConvexBeliefPropagator):

    def __init__(self, markov_net, counting_numbers=None):

        super(PartialConvexBeliefPropagator, self).__init__(markov_net, counting_numbers)
        self.initialize_mats()

        # selected_nodes are nodes selected by the algorithm and we need to update messages from it or to it
        self._selected_nodes_ids = None
        self._selected_nodes = None

        # update_nodes are nodes which we need to update their beliefs. They are selected nodes and their neighbors
        self._update_nodes = None
        self._update_nodes_ids = None

        # update_messages are  messages we need to update.
        # reversed_messages are messages we need to use for updating update_messages.
        # update_messages[e] = (i,j) then reversed_messages[e] = (j,i)
        # from_nodes are used to update messages

        self._update_messages_ids = None
        self._reversed_messages_ids = None
        self._needed_from_nodes_ids = None

        # needed_edges are edges we need when we update the update_nodes

        self._needed_messages_ids = None
        self._needed_messages = None

        # partial_to_map is the T in note. It is a len(self._needed_edges) * len(self._update_nodes) matrix
        self._partial_to_map = None

    def initialize_mats(self):
        self.belief_mat = self.mn.unary_mat + self.augmented_mat
        self.belief_mat += sparse_dot(self.message_mat, self.mn.message_to_map)
        self.belief_mat -= logsumexp(self.belief_mat, 0)

        # update messages
        adjusted_message_prod = self.mn.edge_pot_tensor + self.belief_mat[:, self.mn.message_from] \
                                - np.hstack((self.message_mat[:, self.mn.num_edges:],
                                             self.message_mat[:, :self.mn.num_edges]))

        messages = np.squeeze(logsumexp(adjusted_message_prod, 1))
        messages = np.nan_to_num(messages - messages.max(0))
        self.message_mat = messages

    def select_update_part(self, N):
        selected_nodes, selected_nodes_ids = self.select_nodes(N)
        self._selected_nodes_ids = selected_nodes_ids
        self._selected_nodes = selected_nodes
        update_nodes, update_nodes_ids = self.select_update_nodes()
        self._update_nodes = update_nodes
        self._update_nodes_ids = update_nodes_ids
        update_messages_ids, reversed_messages_ids, needed_from_nodes_ids = self.get_update_messages()
        self._update_messages_ids = update_messages_ids
        self._reversed_messages_ids = reversed_messages_ids
        self._needed_from_nodes_ids = needed_from_nodes_ids
        needed_messages, needed_messages_ids = self.get_needed_messages()
        self._needed_messages_ids = needed_messages_ids
        self._needed_messages = needed_messages
        self._partial_to_map = self.create_edge_index_mat()

    def select_nodes(self, N):
        selected_nodes = set()
        selected_nodes_ids = list()
        ids = random.sample(range(len(self.mn.variables)), N)
        for id in ids:
            node = self.mn.var_list[id]
            selected_nodes.add(node)
            neighbors = self.mn.neighbors[node]
            for neighbor in neighbors:
                selected_nodes.add(neighbor)
        selected_nodes = list(selected_nodes)
        for node in selected_nodes:
            node_id = self.mn.var_index[node]
            selected_nodes_ids.append(node_id)
        return selected_nodes, selected_nodes_ids

    def select_update_nodes(self):
        update_nodes = set()
        update_nodes_ids = list()
        for node in self._selected_nodes:
            neighbors = self.mn.neighbors[node]
            for neighbor in neighbors:
                update_nodes.add(neighbor)
        update_nodes = list(update_nodes)
        for node in update_nodes:
            node_id = self.mn.var_index[node]
            update_nodes_ids.append(node_id)
        return update_nodes, update_nodes_ids

    def get_update_messages(self):
        update_edges = set()
        for node in self._selected_nodes:
            neighbors = self.mn.neighbors[node]
            for neighbor in neighbors:
                #if self.mn.var_index[node] < self.mn.var_index[neighbor]:
                if node < neighbor:
                    update_edges.add((node, neighbor))
                else:
                    update_edges.add((neighbor, node))
        update_edges = list(update_edges)
        num_edges = len(update_edges)
        update_messages_ids = [1]*(2 * num_edges) # messages we need to update
        reversed_messages_ids = [1]*(2 * num_edges) # messages we need to use for updating messages, which is reversed to the update_messages
        needed_from_nodes_ids = [1]*(2 * num_edges) # the start nodes of each edges, which is used to update messages
        for i in range(0, num_edges):
            message_id = self.mn.message_index[update_edges[i]]
            reversed_messages_id = message_id + self.mn.num_edges
            message_from_node_id = self.mn.var_index[update_edges[i][0]]
            reversed_message_from_node_id = self.mn.var_index[update_edges[i][1]]
            update_messages_ids[i] = message_id
            update_messages_ids[i + num_edges] = reversed_messages_id
            reversed_messages_ids[i] = reversed_messages_id
            reversed_messages_ids[i + num_edges] = message_id
            needed_from_nodes_ids[i] = message_from_node_id
            needed_from_nodes_ids[i + num_edges] = reversed_message_from_node_id
        return update_messages_ids, reversed_messages_ids, needed_from_nodes_ids

    def get_needed_messages(self):
        needed_messages = set()
        for node in self._update_nodes:
            neighbors = self.mn.neighbors[node]
            for neighbor in neighbors:
                    needed_messages.add((neighbor, node))
        needed_messages = list(needed_messages)
        needed_messages_ids = list() #messages we need to use to update unary beliefs
        for message in needed_messages:
            #if self.mn.var_index[message[0]] < self.mn.var_index[message[1]]:
            if message[0] < message[1]:
                needed_messages_ids.append(self.mn.message_index[message])
            else:
                needed_messages_ids.append(self.mn.message_index[(message[1],message[0])] + self.mn.num_edges)
        return needed_messages, needed_messages_ids

    def create_edge_index_mat(self):
        to_rows = np.ones(len(self._needed_messages_ids),dtype = np.int)
        to_cols = np.zeros(len(self._needed_messages_ids),dtype = np.int)
        to_rows = np.cumsum(to_rows) - 1
        for i in range(0, len(to_cols)):
            to_id = self.mn.var_index[self._needed_messages[i][1]]
            to_cols[i] = self._update_nodes_ids.index(to_id)
        data = np.ones(len(self._needed_messages_ids))
        indices = (to_rows, to_cols)
        matrix_shape = (len(self._needed_messages_ids), len(self._update_nodes_ids))
        partial_to_map = csc_matrix((data, indices), matrix_shape)
        return partial_to_map

    def partial_compute_beliefs(self):
        if not self.fully_conditioned:
            partial_unary_coefficients = self.unary_coefficients[self._update_nodes_ids].T
            partial_belief_mat = self.mn.unary_mat[:, self._update_nodes_ids] + self.augmented_mat[:, self._update_nodes_ids]
            partial_belief_mat += sparse_dot(self.message_mat[:, self._needed_messages_ids], self._partial_to_map)
            partial_belief_mat /= partial_unary_coefficients
            partial_belief_mat -= logsumexp(partial_belief_mat, 0)
            self.belief_mat[:, self._update_nodes_ids] = partial_belief_mat

    def partial_update_messages(self):
        self.partial_compute_beliefs()
        partial_edge_counting_numbers = self.edge_counting_numbers[self._update_messages_ids]
        adjusted_message_prod = self.mn.edge_pot_tensor[:,:,self._update_messages_ids] \
        - self.message_mat[:, self._reversed_messages_ids]
        adjusted_message_prod /= partial_edge_counting_numbers
        adjusted_message_prod += self.belief_mat[:, self._needed_from_nodes_ids]
        messages = np.squeeze(logsumexp(adjusted_message_prod, 1)) * partial_edge_counting_numbers
        messages = np.nan_to_num(messages - messages.max(0))
        change = np.sum(np.abs(messages - self.message_mat[:, self._update_messages_ids]))
        self.message_mat[:, self._update_messages_ids] = messages

        return change

    def partial_infer(self,  N, tolerance=1e-8):
        self.select_update_part(N)
        change = np.inf
        iteration = 0
        for it in range(0, self.max_iter):
            change = self.partial_update_messages()
            #print("Iteration %d, change in messages %f." % (it, change))
            if change < tolerance:
                break



