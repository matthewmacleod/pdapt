#  classification.py
#
#  Author: Matthew K. MacLeod
#
#  For license information see license.txt

""" classifiction module


"""

import math, random
from collections import defaultdict
from pdapt_lib.machine_learning.maths import sum_of_squares, dot
import numpy as np
from math import sqrt



def softmax(xs):
    """ allow for conversion from scores to probabilities
    Input: xs which are numpy array with 1 row for each score and any number of columns
    Output: array of probabilities

    NB useful to convert array from general numbers to those between 0 and 1
    NB probabilities will also sum to 1

    >>> softmax(np.array([3.0, 1.0, 0.2]))
    array([ 0.8360188 ,  0.11314284,  0.05083836])
    >>> softmax(np.array([30.0, 10.0, 2.0]))
    array([  9.99999998e-01,   2.06115362e-09,   6.91440009e-13])
    >>> softmax(np.array([0.30, 0.1, 0.02]))
    array([ 0.38842275,  0.31801365,  0.2935636 ])
    >>> sum(softmax(np.array([0.30, 0.1, 0.02])))
    1.0
    """
    smax = np.exp(xs) / np.sum(np.exp(xs), axis=0)
    return smax


def logistic(xs):
    """
    """
    return 1.0 / (1.0 + math.exp(-xs))


def logistic_derivative(xs):
    """
    """
    return logistic(xs) * (1.0 - logistic(xs))
