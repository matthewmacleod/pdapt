import math, sys, os


def vector_add(v,w):
    """ add corresponding elements """
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

def test_func(x):
    """This function will try to calculate:

    .. math::
              \sum_{i=1}^{\\infty} x_{i}

    good luck!
    """
    pass

