""" statistics tools


"""
import math
from collections import defaultdict
from pdapt_lib.machine_learning.maths import sum_of_squares, dot

def mean(x):
    """ this is the average
    # example:
    >>> mean([1,2,3,4,5,6])
    3.5
    """
    return sum(x) / len(x)

def median(x):
    """ middle of data, aka Q2
    >>> median([1,2,5,4,3])
    3
    >>> median([1,2,6,4,5,3])
    3.5
    """
    x = sorted(x)
    if len(x) % 2 == 1: # odd length return midpoint
        return x[len(x) // 2]
    else:
        low = x[len(x) // 2 - 1]
        high = x[len(x) // 2]
        return (low + high) / 2

def quantile(x,p):
    """ return pth percentile in x
    >>> quantile([1,2,3,4,5,6,7,8,9,10],0.9)
    9
    """
    pth_index = int(p*len(x))-1
    return sorted(x)[pth_index]

def mode(x):
    """ most common value
    todo this probably could be faster
    returns list since might be more than one
    >>> mode([0,0,1,2,2])
    [0, 2]
    """
    counts = defaultdict(int) # int yields 0 as default value
    for i in x:
        counts[i] += 1
    max_count = max(counts.values())
    modes = []
    for i in x:
        if counts[i] == max_count and i not in modes:
            modes.append(i)
    return modes

def data_range(x):
    """ measure dispersion in most simplistic way
    >>> data_range([1,2,6,4,5,3])
    5
    """
    return max(x) - min(x)

def from_mean(x):
    """ measure dispersion in another way
    >>> from_mean([1,2,3,4,5,6])
    [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    """
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    """ measure dispersion in another way
    >>> variance([1,2,3,4,5,6])
    3.5
    """
    deviations = from_mean(x)
    return sum_of_squares(deviations) / (len(x)-1)

def covariance(x,y):
    """ measure dispersion in another way
    >>> covariance([1,2,3,4,5,6],[1,2,3,4,5,7])
    4.0
    """
    return dot(from_mean(x), from_mean(y)) / (len(x) - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def correlation(x,y):
    """ measure dispersion in another way
    >>> correlation([1,2,3,4,5,6],[1,2,3,4,5,7])
    0.989743318610787
    """
    std_dev_x = standard_deviation(x)
    std_dev_y = standard_deviation(y)
    if std_dev_x > 0 and std_dev_y > 0:
        return covariance(x,y) / std_dev_x / std_dev_y
    else:
        return 0












