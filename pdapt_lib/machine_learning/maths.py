import math, sys, os

from pdapt_lib.basics.tco import TailCaller, TailCall, tailcall


""" Maths module
    basic linear algebra
    statistics
    probability
    machine learning
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


def test_func(x):
    """This function will try to calculate:

    .. math::
              \sum_{i=1}^{\\infty} x_{i}

    good luck!
    """
    pass


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

def dot(v,w):
    """ Dot product aka inner product

    .. math::
            \Gamma(z) = \int_0^\infty x^{z-1}e^{-x}\,dx

    """
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return map(lambda x: c*x, v)

def vector_mean(vectors):
    """ compute the vector whose ith element is mean of the
    ith elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def sum_of_squares(v):
    """ v_1 * v_1 + ... + v_n * v_n """
    return dot(v,v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))


def squared_distance(v,w):
    """ (v_1-w1)**2 + ... + (v_n - w_n)**2 """
    return sum_of_squares(vector_subtract(v,w))

def distance(v,w):
    return math.sqrt(vector_subtract(v,w))

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A,i):
    return A[i]

def get_column(A,j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def is_diagonal(i,j):
    return 1 if i == j else 0

## geometry specific stuff ##



### Statistics ###



### Probability ####



### Machine Learning ###


### run doctests ###

if __name__ == "__main__":
    import doctest
    doctest.testmod()

