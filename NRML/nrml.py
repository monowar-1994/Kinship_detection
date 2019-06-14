"""
This file contains the neighborhood repulsed metric learning implementation as described in https://ieeexplore.ieee.org/document/6562692
Author: Md. Monowar Anjum
Email: monowaranjum@gmail.com
"""

import numpy as np
from NRML import nrml_utils as nutils


class NRML:
    def __init__(self, feature_dimension, number_of_samples):
        self.parent_data = np.zeros((number_of_samples, feature_dimension))
        self.child_data = np.zeros((number_of_samples, feature_dimension))
        self.dimension = feature_dimension
        self.sample_num = number_of_samples
        self.PARENT_CODE = 0
        self.CHILD_CODE = 1
        self.parent_data_distance_list = []
        self.child_data_distance_list = []

    def set_data(self, x_i, y_i):
        self.parent_data = x_i
        self.child_data = y_i

    def initialize_data(self):
        # For every x_i find the distance based ranking

        for i in range(self.sample_num):
            __temp_p = self.parent_data[i]
            __temp_c = self.child_data[i]

            __temp_p_distance_list = []
            __temp_c_distance_list = []

            for k in range(self.sample_num):

                t_dist = nutils.euclidian_distance(__temp_p, self.parent_data[k])
                __temp_p_distance_list.append((self.PARENT_CODE, k, t_dist))

                t_dist = nutils.euclidian_distance(__temp_p, self.child_data[k])
                __temp_p_distance_list.append((self.CHILD_CODE, k, t_dist))

                t_dist = nutils.euclidian_distance(__temp_c, self.parent_data[k])
                __temp_c_distance_list.append((self.PARENT_CODE, k, t_dist))

                t_dist = nutils.euclidian_distance(__temp_c, self.child_data[k])
                __temp_c_distance_list.append((self.CHILD_CODE, k, t_dist))

            self.parent_data_distance_list.append(__temp_p_distance_list)
            self.child_data_distance_list.append(__temp_c_distance_list)

        print("Data processing Done, will start sorting now.")

        for distance_list in self.parent_data_distance_list:
            distance_list.sort(key=lambda tup: tup[2])
        for distance_list in self.child_data_distance_list:
            distance_list.sort(key=lambda tup: tup[2])

        print("data sorting done")

        print("initializing done")

    def get_present_distance_list(self):
        return self.parent_data_distance_list, self.child_data_distance_list