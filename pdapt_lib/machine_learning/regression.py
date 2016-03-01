#  regression.py
#
#  Author: Matthew K. MacLeod
#
#  For license information see license.txt

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

def get_numpy_data(data_frame, features, output):
    """
    Input: pandas data frame,  a list of feature names (e.g. ['sqft_living', 'bedrooms']) and an target feature e.g. ('price')
    Output: A numpy matrix whose columns are the desired features plus a constant column (this is how we create an 'intercept')
    """
    data_frame['constant'] = 1
    features = ['constant'] + features
    features_frame = data_frame[features]
    feature_matrix = np.mat(features_frame)
    output_array = np.array(data_frame[output])
    return(feature_matrix, output_array)


def predict_output(feature_matrix, weights):
    """ create the predictions vector by using np.dot()
    Input: feature matrix, numpy matrix numpy matrix containing the features as columns and weights is a corresponding numpy array
    Output: predictions vector
    """
    predictions = np.dot(feature_matrix, weights)
    pt = predictions.T
    preds = np.squeeze(np.asarray(pt)) # turn matrix into array
    return(preds)

def feature_derivative(errors, feature):
    """
    Input: Assume that errors and feature are both numpy arrays of the same length (number of data points)
    Output:   compute twice the dot product of these vectors as 'derivative' and return the value
    """
    derivative = 2.0 * np.dot(errors, feature)
    return(derivative)


def rss(test, out):
    """
    Input:  functions output (test), and correct answers (out)
    Output: residual sum of square errors
    """
    residuals = test - out
    rss = np.dot(residuals, residuals)
    return rss


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


def polynomial_frame(feature, degree):
    """
    Input: numpy feature array, powers (degrees) to raise initial feature matrix
    Output: pandas dataframe with feature^n, feature^(n+1),... feature^degree
    """
    poly_frame = pd.DataFrame()
    poly_frame['power_1'] = feature
    if degree > 1:
        for power in range(2, degree + 1):
            name = 'power_' + str(power)
            poly_frame[name] =  np.array([x**power for x in feature])
    return poly_frame


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    """
    Input:  If feature_is_constant is True, derivative is twice the dot product of errors and feature
    Output:
    """
    if feature_is_constant:
        return  2.0 * np.dot(errors, feature)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        return 2.0 * np.dot(errors, feature) + 2.0 * l2_penalty * weight


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    """ gradient descent for l2 regularization
    """
    weights = np.array(initial_weights)
    iterations = 0
    while iterations < max_iterations:
        # compute the predictions based on feature_matrix and weights using predict_output() function
        preds = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = preds - output
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # (Remember: when i=0, computing the derivative of the constant!)
            derivative_i = 0.0
            if i == 0:
                derivative_i = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, True)
            else:
                derivative_i = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, False)
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size*derivative_i
        iterations += 1
    return weights


def normalize_features(feature_matrix):
    """ normalizes columns of a given feature matrix
    Input: feature matrix
    Output: a pair (normalized_features, norms), where the second item contains the norms of original features
    """
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)


def rescale_weights(weights, norms):
    """ normalize weights so can use these weights with unnormalized test data """
    return weights / norms


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    """ lasso coordinate descent algorithm
    """
    # compute prediction
    prediction = predict_output(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    rho_i = sum(feature_matrix[:,i] * ((output - prediction)+ weights[i]*feature_matrix[:,i]))
    new_weight_i = 0.0
    if i == 0: # intercept -- do not regularize
        new_weight_i = rho_i
    elif rho_i < -l1_penalty/2.:
        new_weight_i = rho_i + l1_penalty/2.0
    elif rho_i > l1_penalty/2.:
        new_weight_i = rho_i - l1_penalty/2.0
    else:
        new_weight_i = 0.
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    """
     cyclical coordinate descent where we optimize coordinates 0, 1, ..., (d-1) in order and repeat
    """
    weights = np.array(initial_weights)
    change = np.array(initial_weights) * 0.0
    converged = False
    while not converged:
        for i in range(len(weights)):
            new_weight = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            # compute change in weight for feature
            change[i] = np.abs(new_weight - weights[i])
            # assign new weight
            weights[i] = new_weight
        # maximum change in weight, after all changes have been computed
        max_change = max(change)
        if max_change < tolerance:
            converged = True
    return weights





