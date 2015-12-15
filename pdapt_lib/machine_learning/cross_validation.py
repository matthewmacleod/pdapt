""" cv module

various routines for splitting up the data set

"""
import math, random

def create_data_partition(data, fraction):
    """ input data vector, fraction (eg 0.75)
     where fraction represents the amount to allocate
     to the training set.
    >>> create_data_partition([1,2,3,4,5,6,7,8,9,10],.8)
    ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])
    """
    p_index = round(fraction*len(data))
    training, testing = data[0:p_index], data[p_index:]
    return training, testing


