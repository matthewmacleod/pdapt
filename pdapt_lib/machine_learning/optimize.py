#  optimization.py
#
#  Author: Matthew K. MacLeod
#
#  For license information see license.txt

""" optimize module

gradient descent
stochastic gradient descent

"""
import math, random
from collections import defaultdict
from pdapt_lib.machine_learning.maths import sum_of_squares, dot


def solve_simple_regression(x,y):
    """ solve simple regression coefficients
    returns w_0 (slope) and w_1 (intercept)
    >>> solve_simple_regression(range(9),[19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24])
    (0.7166666666666667, 19.18888888888889)
    """
    n = float(len(x))
    slope = (dot(y,x) - ((sum(y)*sum(x))/n))/(dot(x,x)-((sum(x)*sum(x))/n))
    intercept = sum(y)/n - slope*sum(x)/n
    return slope, intercept

def difference_quotient(f, x, h):
    return (f(x+h)-f(x))/h

def partial_difference_quotien(f, v, i, h):
    """ compute ith partial difference quotient of f at v """
    w = [ v_j + (h if j == i else 0) #adding h only the ith element of v
            for j, v_j in enumerate(v)]
    return (f(w)-f(v))/h

def estimate_gradient(f, v, h=0.0001):
    return [partial_difference_quotient(f,v,i,h)
            for i, _ in enumerate(v)]

def step(v, direction, step_size):
    """ move step size in direction from v """
    return [v_i + step_size * direction_i
            for v_i, direction_i in list(zip(v,direction))]

def safe(f):
    """ return a new function that is same as f except when f produces error, then return inf """
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
        return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """ use gradient descent to find theta that minimizes target function"""
    step_sizes = [100,10,1,0.1,0.01,0.001,0.0001,0.00001]
    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        if abs(value-next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    return lambda *args, **kwargs: -f(*args,**kwargs)

def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn), negate_all(gradient_fn), theta_0, tolerance)

def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = list(zip(x,y))
    theta, alpha = theta_0, alpha_0
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0

    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn), negate_all(gradient_fn), x, y, theta_0, alpha_0)







