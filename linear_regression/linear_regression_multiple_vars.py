import numpy as np
import pandas as pd


def main_two():
    data = pd.read_csv('ex1data2.txt', sep=',', header=None)
    X = data.iloc[:, 0:2]  # read first two columns into X
    y = data.iloc[:, 2]  # read third column into y
    m = len(y)  # num of training examples
    print(data.head())

    '''
    Feature scaling + adding intercept term and initializing parameters
    '''
    X = (X - np.mean(X)) / np.std(X)  # feature scaleing
    ones = np.ones((m, 1))
    X = np.hstack((ones, X))  # adding column of 1's for x0
    alpha = 0.01
    iterations = 400
    theta = np.zeros((3, 1))  # set thetas to zero
    y = y[:, np.newaxis]  # make y into a column vector by inserting an axis along second dimension

    '''
    Computing cost function
    '''
    J = compute_cost_multi(X, y, theta)
    print(J)

    '''
    Attempt to minimize J by gradient descent
    '''
    theta = gradient_descent_multi(X, y, theta, alpha, iterations)
    print(theta)
    theta = normal_equation(X, y)
    print(theta)


def compute_cost_multi(X, y, theta):
    m = len(y)
    offsets = np.dot(X, theta) - y
    return np.sum(np.power(offsets, 2)) / (2 * m)


def gradient_descent_multi(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        offsets = np.dot(X, theta) - y
        partials = np.dot(X.T, offsets)
        theta = theta - (alpha / m) * partials
        print(compute_cost_multi(X, y, theta))
    return theta


def normal_equation(X, y):
    temp = np.dot(X.T, X)
    temp = np.linalg.inv(temp)
    temp = np.dot(temp, X.T)
    return np.dot(temp, y)
