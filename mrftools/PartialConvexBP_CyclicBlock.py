import numpy as np
from scipy.sparse import csc_matrix
import random

from .MatrixBeliefPropagator import MatrixBeliefPropagator, logsumexp, sparse_dot
from .ConvexBeliefPropagator import ConvexBeliefPropagator

class PartialConvexBP_CyclicBolck(ConvexBeliefPropagator):

    def __init__(self, markov_net, counting_numbers=None):

        super(PartialConvexBP_CyclicBolck, self).__init__(markov_net, counting_numbers)
        #self.initialize_mats()

        # # selected_nodes are nodes selected by the algorithm and we need to update messages from it or to it
        # self._selected_nodes_ids = None
        # self._selected_nodes = None

        # update_nodes are nodes which we need to update their beliefs. They are selected nodes and their neighbors
        self._update_nodes_list = None
        self._update_nodes_ids_list = None

        # update_messages are  messages we need to update.
        # reversed_messages are messages we need to use for updating update_messages.
        # update_messages[e] = (i,j) then reversed_messages[e] = (j,i)
        # from_nodes are used to update messages

        self._update_messages_ids_list = list()
        self._reversed_messages_ids_list = list()
        self._needed_from_nodes_ids_list = list()

        # messages we need to use for updating pairwise_potential, which is reversed
        # the start nodes of each edges, which is used to update pairwise_potential

        self._update_pot_messages_ids_list = list()
        self._update_pot_reversed_messages_ids_list = list()
        self._update_pot_needed_from_nodes_ids_list = list()

        # needed_edges are edges we need when we update the update_nodes

        self._needed_messages_ids_list = list()
        self._needed_messages_list = list()

        # partial_to_map is the T in note. It is a len(self._needed_edges) * len(self._update_nodes) matrix
        self._partial_to_map_list = list()

        self._subgraph_index = random.randint(0, 100)
        self._num_subsets = 0

    def separate_nodes(self, num_R, num_C):
        size = list(max(self.mn.variables))
        size[0] = size[0] + 1
        size[1] = size[1] + 1
        updated_node_list = list()
        for r in range(1,num_R+1):
            for c in range(1,num_C+1):
                left = float(c - 1.0)/num_C * size[1]
                right = float(c)/num_C * size[1]
                top = float(r - 1.0)/num_R * size[0]
                bottom = float(r)/num_R * size[0]
                node_subset = list()
                for variable in self.mn.variables:
                    if top <= variable[0] < bottom and left <= variable[1] < right:
                        node_subset.append(variable)
                if node_subset != []:
                    updated_node_list.append(node_subset)
        updated_node_id_list = list()
        for node_subset in updated_node_list:
            node_subset_id = list()
            for node in node_subset:
                node_id = self.mn.var_index[node]
                node_subset_id.append(node_id)
            updated_node_id_list.append(node_subset_id)
        return updated_node_list, updated_node_id_list

    def separate_graph(self, num_R, num_C):
        updated_node_list, updated_node_ids_list = self.separate_nodes(num_R, num_C)
        self._num_subsets = len(updated_node_list)
        self._update_nodes_list = updated_node_list
        self._update_nodes_ids_list = updated_node_ids_list
        for i in range(0, self._num_subsets):
            update_nodes = self._update_nodes_list[i]
            update_nodes_ids = self._update_nodes_ids_list[i]
            update_messages_ids, reversed_messages_ids, needed_from_nodes_ids = self.get_update_messages(update_nodes)
            needed_messages, needed_messages_ids = self.get_needed_messages(update_nodes)
            partial_to_map = self.create_edge_index_mat(needed_messages_ids, needed_messages, update_nodes_ids)
            self._update_messages_ids_list.append(update_messages_ids)
            self._reversed_messages_ids_list.append(reversed_messages_ids)
            self._needed_from_nodes_ids_list.append(needed_from_nodes_ids)
            self._needed_messages_list.append(needed_messages)
            self._needed_messages_ids_list.append(needed_messages_ids)
            self._partial_to_map_list.append(partial_to_map)
            self._update_pot_messages_ids_list.append(update_messages_ids)
            self._update_pot_reversed_messages_ids_list.append(reversed_messages_ids)
            self._update_pot_needed_from_nodes_ids_list.append(needed_from_nodes_ids)


    def get_update_messages(self, update_nodes):
        update_edges = set()
        #for node in self._selected_nodes:
        for node in update_nodes: #self._update_nodes:
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

    def get_needed_messages(self, update_nodes):
        needed_messages = set()
        for node in update_nodes: #self._update_nodes:
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

    def create_edge_index_mat(self, needed_messages_ids, needed_messages, update_nodes_ids):
        to_rows = np.ones(len(needed_messages_ids),dtype = np.int)
        to_cols = np.zeros(len(needed_messages_ids),dtype = np.int)
        to_rows = np.cumsum(to_rows) - 1
        for i in range(0, len(to_cols)):
            to_id = self.mn.var_index[needed_messages[i][1]]
            to_cols[i] = update_nodes_ids.index(to_id)
        data = np.ones(len(needed_messages_ids))
        indices = (to_rows, to_cols)
        matrix_shape = (len(needed_messages_ids), len(update_nodes_ids))
        partial_to_map = csc_matrix((data, indices), matrix_shape)
        return partial_to_map

    def partial_compute_beliefs(self):
        if not self.fully_conditioned:
            f = self._subgraph_index % self._num_subsets
            update_nodes_ids = self._update_nodes_ids_list[f]
            needed_messages_ids = self._needed_messages_ids_list[f]
            partial_to_map = self._partial_to_map_list[f]

            partial_unary_coefficients = self.unary_coefficients[update_nodes_ids].T
            partial_belief_mat = self.mn.unary_mat[:, update_nodes_ids] + self.augmented_mat[:, update_nodes_ids]
            partial_belief_mat += sparse_dot(self.message_mat[:, needed_messages_ids], partial_to_map)
            partial_belief_mat /= partial_unary_coefficients
            partial_belief_mat -= logsumexp(partial_belief_mat, 0)
            self.belief_mat[:, update_nodes_ids] = partial_belief_mat

    def partial_update_messages(self):
        self.partial_compute_beliefs()
        f = self._subgraph_index % self._num_subsets
        update_messages_ids = self._update_messages_ids_list[f]
        reversed_messages_ids = self._reversed_messages_ids_list[f]
        needed_from_nodes_ids = self._needed_from_nodes_ids_list[f]

        partial_edge_counting_numbers = self.edge_counting_numbers[update_messages_ids]
        adjusted_message_prod = self.mn.edge_pot_tensor[:,:,update_messages_ids] \
        - self.message_mat[:, reversed_messages_ids]
        adjusted_message_prod /= partial_edge_counting_numbers
        adjusted_message_prod += self.belief_mat[:, needed_from_nodes_ids]
        messages = np.squeeze(logsumexp(adjusted_message_prod, 1)) * partial_edge_counting_numbers
        messages = np.nan_to_num(messages - messages.max(0))
        change = np.sum(np.abs(messages - self.message_mat[:, update_messages_ids]))
        self.message_mat[:, update_messages_ids] = messages

        return change

    def partial_infer(self, tolerance=1e-8):
        #print "partial_infer"
        change = np.inf
        iteration = 0
        self._subgraph_index = self._subgraph_index + 1
        #print self._subgraph_index % self._num_subsets
        for it in range(0, self.max_iter):
            change = self.partial_update_messages()
            #print("Iteration %d, change in messages %f." % (it, change))
            if change < tolerance:
                break

    def partial_compute_pairwise_beliefs(self):
        if not self.fully_conditioned:
            f = self._subgraph_index % self._num_subsets
            update_pot_needed_from_nodes_ids = self._update_pot_needed_from_nodes_ids_list[f]
            update_pot_reversed_messages_ids = self._update_pot_reversed_messages_ids_list[f]
            update_pot_messages_ids = self._update_pot_messages_ids_list[f]

            num_update_edges = len(update_pot_needed_from_nodes_ids) / 2

            adjusted_message_prod = self.belief_mat[:, update_pot_needed_from_nodes_ids] \
                                - np.nan_to_num(self.message_mat[:, update_pot_reversed_messages_ids] \
                                                / self.edge_counting_numbers[update_pot_reversed_messages_ids])

            to_messages = adjusted_message_prod[:, :num_update_edges].reshape(
                (self.mn.max_states, 1, num_update_edges))

            from_messages = adjusted_message_prod[:, num_update_edges:].reshape(
                (1, self.mn.max_states, num_update_edges))

            beliefs = self.mn.edge_pot_tensor[:, :, update_pot_messages_ids[num_update_edges:]] \
                  / self.edge_counting_numbers[update_pot_messages_ids[num_update_edges:]] \
                  + to_messages + from_messages

            beliefs -= logsumexp(beliefs, (0, 1))

            self.pair_belief_tensor[:, :, update_pot_messages_ids[:num_update_edges]] = beliefs




    def get_feature_expectations(self):
        """
        Computes the feature expectations under the currently estimated marginal probabilities. Only works when the
        model is a LogLinearModel class with features for edges.

        :return: vector of the marginals in order of the flattened unary features first, then the flattened pairwise
                    features
        """

        if self._update_nodes_list == None:
            self.compute_beliefs()
            self.compute_pairwise_beliefs()
        else:
            self.partial_compute_beliefs()
            self.partial_compute_pairwise_beliefs()






        summed_features = self.mn.unary_feature_mat.dot(np.exp(self.belief_mat).T)

        summed_pair_features = self.mn.edge_feature_mat.dot(np.exp(self.pair_belief_tensor).reshape(
            (self.mn.max_states ** 2, self.mn.num_edges)).T)

        marginals = np.append(summed_features.reshape(-1), summed_pair_features.reshape(-1))

        return marginals
