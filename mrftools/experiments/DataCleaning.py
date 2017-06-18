'''
import math
import operator
from collections import defaultdict, Counter
'''

def load_instances(filename, filter_missing_values=False, missing_value='?'):
    '''Returns a list of instances stored in a file'''
    instances = []
    with open(filename, 'r') as f:
        for line in f:
            new_instance = line.strip().split(',')
            if not filter_missing_values or missing_value not in new_instance:
                instances.append(new_instance)
    return instances


def load_attribute_names(filename, separator=':'):
    '''Returns a list of attribute names in a file'''
    with open(filename, 'r') as f:
        attribute_names = [line.strip().split(separator)[0] for line in f]
    return attribute_names


def load_attribute_values(attribute_filename):
    '''Returns a list of attribute values in filename.'''
    attribute_values = []
    with open(attribute_filename) as f:
        for line in f:
            attribute_name_and_value_string_list = line.strip().split(':')
            attribute_name = attribute_name_and_value_string_list[0]
            if len(attribute_name_and_value_string_list) < 2:
                attribute_values.append({}) # no values for this attribute
            else:
                value_abbreviation_description_dict = {}
                description_and_abbreviation_string_list = attribute_name_and_value_string_list[1].strip().split(',')
                for description_and_abbreviation_string in description_and_abbreviation_string_list:
                    description_and_abbreviation = description_and_abbreviation_string.strip().split('=')
                    description = description_and_abbreviation[0]
                    if len(description_and_abbreviation) < 2: # assumption: no more than 1 value is missing an abbreviation
                        value_abbreviation_description_dict[None] = description
                    else:
                        abbreviation = description_and_abbreviation[1]
                        value_abbreviation_description_dict[abbreviation] = description
                attribute_values.append(value_abbreviation_description_dict)
    return attribute_values
