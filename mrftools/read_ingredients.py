import json
import time

import random
random.seed(0)
def read_ingredients():

    start_time = time.time()

    with open('../data/food.json') as data_file:
        recipes = json.load(data_file)

    # get variable values
    ingredients = dict()
    # cuisines = set()

    for recipe in recipes:
        # cuisines.add(recipe[u'cuisine'])
        for ingredient in recipe[u'ingredients']:
            if ingredient.lower() not in ingredients:
                ingredients[ingredient.lower()] = 0
            ingredients[ingredient.lower()] += 1

    print "Total recipes: %d" % len(recipes)

    threshold = 500

    to_delete = set()

    count = 0
    for ingredient in ingredients:
        if ingredients[ingredient] >= threshold:
            count += 1
        else:
            to_delete.add(ingredient)

    print "Number with more than %d recipes: %d" % (threshold, count)


    for ingredient in to_delete:
        del ingredients[ingredient]

    # cuisine_index = dict()
    # for i, cuisine in enumerate(sorted([x for x in cuisines])):
    #     cuisine_index[cuisine] = i

    data = []

    default_dict = dict()
    for ingredient in ingredients:
        default_dict[ingredient] = 0

    for recipe in recipes:
        example = default_dict.copy()
        for ingredient in recipe[u'ingredients']:
            if ingredient.lower() not in to_delete:
                example[ingredient.lower()] = 1

        # example[u'cuisine'] = cuisine_index[recipe[u'cuisine']]

        data.append(example)

    max_num_states = 2 #len(cuisines)
    num_states = dict()
    for ingredient in ingredients:
        num_states[ingredient] = 2
    # num_states[u'cuisine'] = len(cuisines)

    print "Data all loaded in %f seconds. %d entries" % (time.time() - start_time, len(data))

    variables = sorted(ingredients) #+ [u'cuisine']

    print variables

    random.shuffle(data)

    data = data[:10000]


    return data, num_states, max_num_states, variables

