#  probs.py
#
#  Author: Matthew K. MacLeod
#
#  For license information see license.txt


""" Probability

    basics

"""
from pdapt_lib.machine_learning.maths import sum_of_squares, dot, factorial


def choose(n,k):
    """ n choose k
    # example, say want to know ways to form committee of 3 students from 20 total students
    >>> choose(20,3)
    1140.0
    """
    return factorial(n) / (factorial(k)*factorial(n-k))

