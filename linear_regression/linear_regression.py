import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression_multiple_vars import main_two


def main():
    '''
    Before starting on any task, it is often useful to understand the data by visualizing it.
    For this dataset, you can use a scatter plot to visualize the data, since it has only two
    properties to plot (profit and population). (Many other problems that you will encounter
    in real life are multi-dimensional and can’t be plotted on a 2-d plot.)
    '''
    data = pd.read_csv("ex1data1.txt", header=None)  # read from dataset
    # print("Type of pands.read_csv(): ", type(data))
    assert isinstance(data, object)
    # print(data)
    X = data.iloc[:, 0]  # read first column
    # print("Type of pands.read_csv().iloc[]: ", type(X))
    y = data.iloc[:, 1]  # read second column
    m = len(y)  # number of training example (same as X)
    data.head()  # view first few rows of the data
    plt.scatter(X, y)  # create scatter plot of X vs y
    plt.xlabel(" Population of City in 10,000s ")
    plt.ylabel(" Profit in $10,000s ")
    # plt.show()

    '''
    Adding the intercept term
    '''
    X = X[:, np.newaxis]  # make X into a column vector by inserting an axis along second dimension
    y = y[:, np.newaxis]  # make y into a column vector by inserting an axis along second dimension
    theta = np.zeros([2, 1])
    iterations = 1500
    alpha = 0.01  # set learning rate
    ones = np.ones([m, 1])  # create vector with m 1's
    X = np.hstack((ones, X))  # add column of 1's to X to make it a mx2 matrix

    '''
    Computing cost function J(theta)
    '''
    J = compute_cost(X, y, theta)
    print(J)

    '''
    Attempt to minimize J by gradient descent
    '''
    theta = gradient_descent(X, y, theta, alpha, iterations)
    print(theta)

    '''
    Plot showing the best fit line
    '''
    plt.scatter(X[:, 1], y)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.plot(X[:, 1], np.dot(X, theta))
    plt.show()

    main_two()

def compute_cost(X, y, theta):
    m = len(y)
    offsets = np.dot(X, theta) - y  # h(xi) - yi (i = 1 to m) in mx1 vector
    return np.sum(np.power(offsets, 2)) / (2 * m)  # return sum of elements in vector / 2m


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        offsets = np.dot(X, theta) - y  # h(xi) - yi (i = 1 to m) in mx1 vector
        partials = np.dot(X.T, offsets)  # partial derivatives w.r.t theta-0 and theta-1 in 2x1 vector
        theta = theta - (alpha / m) * partials  # adjust theta
        print(compute_cost(X, y, theta))
    return theta


if __name__ == '__main__':
    main()
