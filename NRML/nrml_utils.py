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
    :return: euclidain distance between two arrays
    """
    val = np.sum((a-b)*(a-b))
    return math.sqrt(val)