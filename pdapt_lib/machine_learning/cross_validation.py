""" cv module

various routines for splitting up the data set

"""
import math, random
import numpy as np

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


def random_split(data, fraction):
    """ input data vector, fraction (eg 0.75)
    here we split on fraction but the data for training
    and test splits will be selected at random.
    """
    train_size = round(fraction*len(data))
    length = len(data)
    all_indices = np.array(range(length))
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    training, testing = [],[]
    for i in all_indices:
        if i in train_indices:
            training.append(data[i])
        else:
            testing.append(data[i])
    return training, testing


