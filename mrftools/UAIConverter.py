from mrftools import *
import torch

class UAIConverter(object):

    def __init__(self, filename, is_cuda):
        self.uai_file = open(filename, "r")
        self.mn = TorchMarkovNet(is_cuda=is_cuda)
        self.variables = 0
        self.cardinalities = []
        self.clique_num = 0
        self.cliques = dict()
        self.clique_table = dict()

    def convert(self):
        func_table = True
        error_out = False
        table_counter = 0
        # Iteratre through each line in the file, stripping the whitespace along the way
        for line_num, line in enumerate(self.uai_file):
            stripped_line = line.strip()

            if line_num == 0:
                assert stripped_line == "MARKOV"

            if line_num == 1:
                self.variables = int(stripped_line)

            if line_num == 2:
                for token in stripped_line.split():
                    self.cardinalities.append(int(token))

            if line_num == 3:
                self.clique_num = int(stripped_line)

            if line_num >= 4 and func_table:
                # Once a newline is found, begin the Function Tables
                if stripped_line == "":
                    func_table = False
                    continue

                split_tokens = stripped_line.split()
                del split_tokens[0]
                split_tokens = [int(i) for i in split_tokens]
                self.cliques[int(line_num) - 4] = split_tokens

                # The UAI Converter can only convert pairwise graphs (limitation on our own MarkovNet and BP)
                if len(split_tokens) > 2:
                    error_out = True
                    break

            if line_num >= 4 and not func_table:
                if stripped_line == "":
                    table_counter += 1
                    continue
                if len(stripped_line) == 1:
                    continue
                split_tokens = stripped_line.split()
                split_tokens = [float(i) for i in split_tokens]
                if table_counter not in self.clique_table:
                    self.clique_table[table_counter] = [split_tokens]
                else:
                    cur_list = self.clique_table[table_counter]
                    cur_list.append(split_tokens)
                    self.clique_table[table_counter] = cur_list

        # Finished reading through file (or errored out)
        if error_out:
            raise ValueError("This UAI Converter can only be used for pairwise graphs (the number of variables in a clique can be no more than 2)")

        # Confirm simple checks on variables and cliques, required in the UAI format
        assert len(self.cardinalities) == self.variables
        assert len(self.cliques) == self.clique_num and len(self.clique_table) == self.clique_num

        # Initialize all unary factors to 0
        for v in range(self.variables):
            self.mn.set_unary_factor(v, torch.from_numpy(np.zeros(self.cardinalities[v])))

        # Update all of the unary factors and edges provided in the Function Tables
        for c in range(self.clique_num):
            if len(self.cliques[c]) == 1:
                vals = self.clique_table[c]
                self.mn.set_unary_factor(self.cliques[c][0], torch.from_numpy(np.array(vals[0])))
            elif len(self.cliques[c]) == 2:
                vals = self.clique_table[c]
                self.mn.set_edge_factor((self.cliques[c][0], self.cliques[c][1]),
                                        torch.from_numpy(np.array(vals)
                                                         .reshape(self.cardinalities[self.cliques[c][0]],
                                                                  self.cardinalities[self.cliques[c][1]])))
            else:
                raise ValueError("Somehow a clique of more than 2 got through")

        # Create the TorchMarkovNet and return
        self.mn.create_matrices()
        return self.mn