#  cross_validation.py
#
#  Author: Matthew K. MacLeod
#
#  For license information see license.txt

""" cv module

various routines for splitting up the data set

"""
import math, random
import numpy as np

def create_train_test_partition(data, fraction):
    """ input: data vector, fraction (eg 0.75)
     where fraction represents the amount to allocate
     to the training set.
    output: train, test
    >>> create_train_test_partition([1,2,3,4,5,6,7,8,9,10],.8)
    ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10])
    """
    p_index = round(fraction*len(data))
    training, testing = data[0:p_index], data[p_index:]
    return training, testing


def create_train_validation_test_partition(data, train_fraction, test_fraction):
    """ input: data vector, train fraction (eg 0.80), test fraction (eg 0.10)
     where fraction represents the amount to allocate
     to the training set.
    output: train, validation, test
    >>> create_train_validation_test_partition([1,2,3,4,5,6,7,8,9,10],.8,.1)
    ([1, 2, 3, 4, 5, 6, 7, 8], [9], [10])
    """
    train_index = round(train_fraction*len(data))
    test_index = round((1.0-test_fraction)*len(data))
    train, validation, test = data[0:train_index], data[train_index:test_index], data[test_index:]
    return train, validation, test



def random_split(data, fraction):
    """ input: data vector, fraction (eg 0.75)
    here we split on fraction but the data for training
    and test splits will be selected at random.
    output: train, test, train_indices
    NB 1: train indices are also included in output for more complicated splits
    NB 2: for more splits, simply call multiple times, eg
       training_and_validation, testing = random_split(data, 0.9)
       training, validation = random_split(training_and_validation, 0.5)
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
    return training, testing, train_indices


def make_df_split(data_df, test_portion):
    """ returns randomized split of actual pandas dataframes, not slices
    Input: pandas dataframe, test portion eg 0.2 for 80-20 split
    Output: train and test data frames
    """
    train = df.sample(frac=(1.0-test_portion),random_state=200)
    test = df.drop(train.index)
    return train, test


