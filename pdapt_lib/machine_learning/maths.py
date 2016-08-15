#  maths.py
#
#  Author: Matthew K. MacLeod
#
#  For license information see license.txt

import math, sys, os

from pdapt_lib.basics.tco import TailCaller, TailCall, tailcall

import numpy as np
from functools import reduce


""" Maths module

    basic linear algebra
     vector
     matrix
"""

@TailCaller
def factorial(n, acc=1):
    """ simple factorial
    Args: just n as using default args for accumulator.
    here's a doc test:
    >>> factorial(5)
    120
    """
    if n == 1: return acc
    else: return tailcall(factorial)(n-1, n*acc)


def vector_add(v,w):
    """ add corresponding elements
    # example:
    >>> vector_add([1,2,3],[1,2,3])
    [2, 4, 6]
    """
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v,w):
    """ subtract corresponding elements """
    return [v_i - w_i for v_i, w_i in zip(v,w)]


def dot(a,b):
    """ Dot product aka inner product

    .. math::
            \mathbf{A} \cdot \mathbf{B} = A^\dag B = \sum_i^n A_i * B_i

    # Doctest Example:
    >>> dot([0,2.0,4],[0,1,1])
    6.0
    """
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def vector_sum(vectors):
    """ sum componentwise
    >>> vector_sum([[1,2,3],[1,2,3]])
    12
    """
    summed = vectors[0]
    vectors = vectors[1:]
    for v in vectors:
        summed = vector_add(summed,v)
    return reduce(lambda acc,i: acc+i, summed)


def scalar_multiply(c, v):
    """ scale each component of vector
    # example
    >>> scalar_multiply(10.0, [1,2,3])
    [10.0, 20.0, 30.0]
    """
    return list(map(lambda x: c*x, v))


def vector_mean(vectors):
    """ compute the vector whose ith element is mean of the
    ith elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def sum_of_squares(v):
    """ v_1 * v_1 + ... + v_n * v_n """
    return dot(v,v)


def magnitude(v):
    """ this is the l2 norm """
    return math.sqrt(sum_of_squares(v))

def lp_norm(v,p):
    """ l_p norm
    input: vector, p (order of the norm)
    output: norm
    NB for more info see
    https://en.wikipedia.org/wiki/Lp_space
    >>> lp_norm(range(1,11),2)
    19.621416870348583
    >>> lp_norm(list(map(lambda x: abs(x),list(range(1,11)))),1)
    55.0
    >>> lp_norm(range(1,11),1)
    55.0
    >>> lp_norm(range(1,11),0.5)
    504.82352465265495
    """
    return (sum(map(lambda x: abs(x)**p,v)))**(1.0/p)


def squared_distance(v,w):
    """ (v_1-w1)**2 + ... + (v_n - w_n)**2 """
    return sum_of_squares(vector_subtract(v,w))


def distance(v,w):
    """
    >>> distance([1.2, 2.2, 3.8], [1.0, 2.0, 8.8])
    5.007993610219566
    """
    diff = vector_subtract(v,w)
    return magnitude(diff)


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


def get_row(A,i): return A[i]


def get_column(A,j): return [A_i[j] for A_i in A]


def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]


def is_diagonal(i,j): return 1 if i == j else 0


## geometry specific stuff ##

# function distance will work for points

def angle(v,w):
    return math.acos( dot(v,w)/(magnitude(v)*magnitude(w)) ) * 180.0/math.pi


def point_angle(a, b, c):
    """ a b c are xyz points, in degrees
    # example:
    >>> abs(point_angle([-2.975941,3.026175,-1.069039],[-3.318522,2.808196,0.810145],[-3.627889,4.656182,1.240712]) - 97.966613497642) < 0.00000000001
    True
    """
    ba = vector_subtract(a,b) # vector pointing from b to a is a-b
    bc = vector_subtract(c,b) # vector pointing from b to c is c-b
    return angle(ba,bc)


def cross(v,w):
    a = np.array(v)
    b = np.array(w)
    return np.cross(a,b)


def dihedral(a, b, c, d):
   """  get dihedral angle in degrees from points
   # example:
   >>> dihedral([-2.975941,3.026175,-1.069039],[-3.318522,2.808196,0.810145],[-3.627889,4.656182,1.240712],[-4.122107,4.795455,2.203724])
   163.75385970522458
   """
   # vectors connecting points
   v1 = vector_subtract(b,a)
   v2 = vector_subtract(b,c) # notice the direction is from c to b
   v3 = vector_subtract(d,c)
   # form normals
   n1 = cross(v1,v2) / magnitude(cross(v1,v2))
   n2 = cross(v2,v3) / magnitude(cross(v2,v3))
   # form orthogonal frame
   x = dot(n1,n2)
   v2n = scalar_multiply(1.0/magnitude(v2),v2)
   m1 = cross(n1,v2n)
   y = dot(m1,n2)
   return math.atan2(y,x) * 180.0/math.pi


def norm(x):
    """ returns the norm of a vector
    expects np ararry
    >>> norm(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]))
    3.3166247903553998
    """
    sum_of_squares = x.dot(x.T)
    return np.sqrt(sum_of_squares)


def cosine_distance(x,y):
    """ cosine distance metric
    nb: not a proper distance metric but useful
    >>> cosine_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.5648058601107554
    >>> cosine_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.32159947470003181
    >>> cosine_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.00024025948370620664
    """
    numerator =  x.dot(y.T)
    denominator = norm(x) * norm(y)
    return 1.0 - numerator / denominator


def euclidean_distance(x,y):
    """ assumptions:
    The Euclidean distance assumes that clusters have identity covariances,
        ie the dimensions are statistically independent and the variance of along each dimension (column) is one.
    input: expecting np arrays
    nb: not scaled
    >>> euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    3.6055512754639891
    >>> euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    3.0
    >>> euclidean_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.10000000000000009
    """
    difference = x - y
    return norm(difference)


def braycurtis_distance(x,y):
    """ bray curtis distance
    input: expecting np arrays
    >>> braycurtis_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.52941176470588236
    """
    return np.abs(x - y).sum() / np.abs(x + y).sum()


def cityblock_distance(x,y):
    """ expecting np arrays
    input: expecting np arrays
    >>> cityblock_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    9.0
    """
    differences = np.abs(x-y)
    return float(differences.sum())


def canberra_distance(x,y):
    """ weighted cityblock
    input: expecting np arrays
    >>> canberra_distance(np.array([ 3, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    6.3333333333333339
    """
    numerator =  np.abs(x-y)
    denominator = np.abs(x) + np.abs(y)
    return (numerator / denominator).sum()


def jaccard_distance(x,y):
    """ sets are convenient, since defined as
       D_j =  | intersection | / | union |
    input: arrays or lists of some sort, converted within to sets
    >>> jaccard_distance(np.array([ 2.0, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0.0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.0
    >>> jaccard_distance(np.array([ 2.0, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2.0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.0
    >>> jaccard_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2.0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.25
    """
    x, y = set(x), set(y)
    return 1.0 - len(x & y)  / float(len(x | y))


def sorensen_distance(x,y):
    """ Sorensen distance
        D_s = 2 intersection  /  (union + intersection)
    input: arrays or lists of some sort, converted within to sets
    >>> sorensen_distance(np.array([ 2.0, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0.0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.0
    >>> sorensen_distance(np.array([ 2.0, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2.0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.0
    >>> sorensen_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2.0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.1428571428571429
    """
    x, y = set(x), set(y)
    intersection = float(len(x & y))
    union = float(len(x | y))
    return 1.0 - (2*intersection) / (intersection + union)


### custom distances


def normalized_sigmoid(x):
    return 2.0/(1.0 + np.exp(-0.5*x))-1.0


def sigmoid_euclidean_distance(x,y):
    """
    >>> sigmoid_euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.71697295177260845
    >>> sigmoid_euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.6351489523872873
    >>> sigmoid_euclidean_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.024994792968420665
    """
    dist = euclidean_distance(x,y)
    return normalized_sigmoid(dist)


def alternative_euclidean_distance(x,y):
    """ assumptions:
    input: expecting np arrays
    nb: closer vectors should be closer to zero
    >>> alternative_euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.78287072704466754
    >>> alternative_euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.75
    >>> alternative_euclidean_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.090909090909090939
    """
    difference = x - y
    return 1.0 - (1.0 / (1.0 + norm(difference)))


def cosine_sigmoid_euclidean_distance(x,y):
    """ mean between cosine and sigmoid euclidean distance
    >>> cosine_sigmoid_euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.64088940594168187
    >>> cosine_sigmoid_euclidean_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.47837421354365955
    >>> cosine_sigmoid_euclidean_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.012617526226063436
    """
    cosine_dist = cosine_distance(x,y)
    sigmoid_dist = sigmoid_euclidean_distance(x,y)
    return (cosine_dist + sigmoid_dist) / 2.0


def pearson_distance(x,y):
    """ 1 minus pearson
    >>> pearson_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 0, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.77296041684200578
    >>> pearson_distance(np.array([ 2, 0, 1, 1, 1, 1, 1, 1, 1, 0]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.5
    >>> pearson_distance(np.array([ 1.9, 2, 2, 1, 1, 0, 0,0, 1, 1]), np.array([ 2, 2, 2, 1, 1, 0, 0,0, 1, 1]))
    0.00031570189411622707
    """
    if len(x) < 3 : return 0.0
    pearson =  0.5 + 0.5 * np.corrcoef(x, y, rowvar = 0)[0][1]
    return 1.0 - pearson




if __name__ == "__main__":
    import doctest
    doctest.testmod()

