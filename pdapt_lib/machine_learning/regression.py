""" regression module

simple linear regression
general regression

"""
import math, random
from collections import defaultdict
from pdapt_lib.machine_learning.maths import sum_of_squares, dot
import numpy as np
from math import sqrt


def solve_simple_regression(x,y):
    """ solve simple regression coefficients
    input: x and y from following
    y = w_0 x  + w_1
    output: w_0 (slope) and w_1 (intercept)
    >>> solve_simple_regression(range(9),[19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24])
    (0.7166666666666667, 19.18888888888889)
    """
    n = float(len(x))
    slope = (dot(y,x) - ((sum(y)*sum(x))/n))/(dot(x,x)-((sum(x)*sum(x))/n))
    intercept = sum(y)/n - slope*sum(x)/n
    return slope, intercept


def get_regression_predictions(input_feature, slope, intercept):
    """ simple regression model based on input coefficients
    input: input_feature (x vector), slope(w_0), intercept (w_1)
    output: y hat
    """
    predicted_values = intercept + slope*input_feature
    return predicted_values


def residual_sum_of_squares(input_feature, output, slope, intercept):
    """ simple regression model based on input coefficients
    input: input_feature (x vector), output (y vector), slope(w_0), intercept (w_1)
    output: rss
    """
    predictions = get_regression_predictions(input_feature, slope, intercept)
    residuals = output - predictions
    rss = dot(residuals, residuals)
    return(rss)


def inverse_regression_predictions(output, slope, intercept):
    """ simple regression model based on input coefficients
    input: output (y vector), slope(w_0), intercept (w_1)
    output: x hat
    """
    estimated_feature = (output - intercept)/slope
    return estimated_feature

# more general regression
def predict_output(feature_matrix, weights):
    """ assume feature_matrix is a numpy matrix containing the features
    as columns and weights is a corresponding numpy array
    """
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


def partial_derivative(errors, feature):
    """ assume that errors and feature are both numpy arrays of the same length
    compute twice the dot product of these vectors as 'derivative' and return the value
    """
    derivative = 2.0 * np.dot(errors, feature)
    return(derivative)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    """ gradient descent for muliple linear regression
        w_t+1 <- w_t - \eta \nabla
    """
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute predictions based on feature_matrix and weights using
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        gradient_sum_squares = 0.0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # compute the partial derivative for weight[i]:
            derivative_i = partial_derivative(errors, feature_matrix[:,i])
            partial_i_dotted = np.dot(derivative_i, derivative_i)
            gradient_sum_squares += partial_i_dotted
            # update the weight based on step size and derivative
            weights[i] = weights[i] - step_size*derivative_i
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)




