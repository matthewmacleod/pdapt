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

def interquartile_range(x):
    """ this is Q3-Q1, hinges on box plots are 1.5 x this amount
    >>> interquartile_range([1,2,3,4,5,6,7,8,9,10])
    5
    """
    return quantile(x,0.75) - quantile(x,0.25)

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
    n = len(x)
    deviations = from_mean(x)
    # note that almost average squared deviation from mean, but dont divide by
    # n but n-1 this is because x_bar is only an estimate since assuming are sampling
    # from a larger population so the squared deviations are an underestimate thus correct
    # for this by dividing by n-1 instead of n
    return sum_of_squares(deviations) / (n-1)

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

def outliers(x):
    """ ouliers are defined as +- 1.5 x IQR """
    iqr = interquartile_range(x)
    min_x = median(x) - 1.5 * iqr
    max_x = median(x) + 1.5 * iqr
    return [ i for i in x if i < min_x or i > max_x]

def standardize(x):
    """
    standardize a vector
    """
    m = mean(x)
    s = standard_deviation(x)
    return [(i-m)/s for i in x]

def unstandardize(z,m,s):
    """
    return original vector
    """
    return [i*s+m for i in z]

def winsorise(v, limit):
    """ https://en.wikipedia.org/wiki/Winsorising
    input: v (vector), limit is percent cutoff on edges of distribution ..to reduce *possible* spurious effects of outliers
    output: winsorized vector (same length but outliers have been replaced with closest values
    NB a 0.05 percent limit would alter 10% of the data
    NB sometimes outliers should not be removed!
    >>> winsorise([92, 19, 101, 58, 1053, 91, 26, 78, 10, 13, -40, 101, 86, 85, 15, 89, 89, 28, -5, 41], 0.05)
    [92, 19, 101, 58, 101, 91, 26, 78, 10, 13, -5, 101, 86, 85, 15, 89, 89, 28, -5, 41]
    >>> mean(winsorise([92, 19, 101, 58, 1053, 91, 26, 78, 10, 13, -40, 101, 86, 85, 15, 89, 89, 28, -5, 41], 0.05))
    55.65
    """
    vs = sorted(v)
    low_quantile_limit, upper_quantile_limit  = quantile(vs,limit), quantile(vs,1.0-limit)
    lower_replacement = vs[vs.index(low_quantile_limit)+1]
    upper_replacement = vs[vs.index(upper_quantile_limit)]
    w = []
    for i in v:
        if i < lower_replacement:
            w.append(lower_replacement)
        elif i > upper_replacement:
            w.append(upper_replacement)
        else:
            w.append(i)
    return w

def summary(x):
    """
    print out summary statistics
    for now set up to work on a list x
    """
    print('{0:10} {1:5f}'.format('Minimum:', min(x)))
    print('{0:10} {1:5f}'.format('Q1:', quantile(x,0.25)))
    print('{0:10} {1:5f}'.format('Median:', median(x)))
    print('{0:10} {1:5f}'.format('Q3:', quantile(x,0.75)))
    print('{0:10} {1:5f}'.format('Maximum:', max(x)))
    print('{0:10} {1:5f}'.format('IQR:', interquartile_range(x)))
    print('Outliers +/- 1.5 IQR:', outliers(x))
    print()
    print('{0:10} {1:5f}'.format('Mean:', mean(x)))
    print('{0:10} {1:5f}'.format('Std dev:', standard_deviation(x)))










