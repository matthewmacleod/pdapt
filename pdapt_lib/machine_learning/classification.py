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

    >>> softmax(np.array([3.0, 1.0, 0.2]))
    array([ 0.8360188 ,  0.11314284,  0.05083836])
    """
    smax = np.exp(xs) / sum(np.exp(xs))
    return smax

