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
    >>> point_angle([-2.975941,3.026175,-1.069039],[-3.318522,2.808196,0.810145],[-3.627889,4.656182,1.240712])
    97.96661349764267
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()

