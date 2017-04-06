import csv
import time

import random
random.seed(0)
def read_plants():

    locations = dict()

    all_locations = set()


    with open('../../../plants/plants.data') as data_file:
        for line in data_file:
            tokens = line.strip().split(',')

            locations[tokens[0]] = set(tokens[1:])

            for loc in tokens[1:]:
                all_locations.add(loc)

    location_abbr = dict()

    num_states = dict()

    with open('../../../plants/stateabbr.txt') as data_file:
        for line in data_file:
            abbr, _, rest = line.partition(' ')

            if abbr in all_locations:
                # print "Found %s" % rest.strip()
                if rest[:2] == 'Qu':
                    location_abbr['Quebec'] = abbr
                else:
                    location_abbr[rest.strip()] = abbr
    variables = []
    for key in location_abbr:
        variables.append(key)
        num_states[key] = 2

    max_num_states = 2

    data = []

    for _, present in locations.items():
        example = dict()

        for variable in variables:
            if location_abbr[variable] in present:
                example[variable] = 0
            else:
                example[variable] = 1

        data.append(example)

    random.shuffle(data)

    print "Loaded %d plants" % len(data)

    return data, num_states, max_num_states, variables



if __name__ == "__main__":

    read_plants()
