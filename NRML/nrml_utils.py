"""
This file contains the neighborhood repulsed metric learning utilties
Author: Md. Monowar Anjum
Email: monowaranjum@gmail.com
"""

import numpy as np
import math
from numpy.core.multiarray import dtype


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


def get_w_matrix(input_matrix, dimension, number_of_eigenvectors):
    """
    Construct the W matrix here which conforms the condition W*(W_transpose) = I
    :param dimension: dimension of eigenvectors , equal to feature dimension as per formulation
    :param input_matrix: The matrix for eigendecomposition. Must be a square matrix
    :param number_of_eigenvectors: the parameter "l" as specified in the paper
    :return: returns the W amtrix for the current iteration on which it is called.
    """
    eig_values, eig_vectors = np.linalg.eig(input_matrix)
    eig_values = np.absolute(eig_values)
    eig_list = []
    for k in range(eig_values.shape[0]):
        eig_list.append((eig_values[k], eig_vectors[:, k]))
    eig_list.sort(key=lambda tup: tup[0], reverse = True)
    w_matrix = np.zeros((dimension, number_of_eigenvectors), dtype= np.complex)
    for i in range(number_of_eigenvectors):
        w_matrix[:, i] = eig_list[i][1]

    return w_matrix


def get_mahalanbish_distance_variant(f1,f2, w):
    """
    This function calculates the mahalanbish distance given the feature map of two images
    :param f1: Feature map from image1
    :param f2: Feature map from image 2
    :param w: The W marix in the formula of Mahalanbish distance
    :return: The distance between two feature maps
    """

    u = (w.T).dot(f1.T)
    v = (w.T).dot(f2.T)
    dist = ((u-v).T).dot(u-v)
    return math.sqrt(np.absolute(dist))