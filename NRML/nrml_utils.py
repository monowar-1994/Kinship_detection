"""
This file contains the neighborhood repulsed metric learning utilties
Author: Md. Monowar Anjum
Email: monowaranjum@gmail.com
"""

import numpy as np
import math


def euclidian_distance(a, b):
    """
    Measures squared distance between two given numpy array
    :param a: input array a
    :param b: input array b
    :return: euclidian distance between two arrays
    """
    val = np.sum((a-b)*(a-b))
    return math.sqrt(val)


def column_to_row_vector_multiplication(X,Y):
    """
    :param X: comes in a numpy array format . Transpose this to make a column vector. Make sure its 2D
    :param Y: comes in a numpy array format . Keep it as it is . But make it 2D
    :return: returns the numpy dot product . Basically its a matrix multiplication . Returns a n*n numpy 2D array
    """

    x_2d = np.zeros((1, X.shape[0]))
    y_2d = np.zeros((1, Y.shape[0]))

    x_2d[0] = X
    y_2d[0] = Y

    return x_2d.T.dot(y_2d)
