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

    def _process_(self,iteration_count, epsilon):
        for k in range(iteration_count):
            #Calculating H1, H2 and H3
            h1_temp = self.calculate_h1(self.sample_num, self.sample_num//2)
            h2_temp = self.calculate_h2(self.sample_num, self.sample_num//2)
            h3_temp = self.calculate_h3(self.sample_num)

    def calculate_h1(self, N, K):
        h1 = 0
        for i in range(N):
            x_i = self.parent_data[i]
            y_it1_list = self.child_data_distance_list[i]

            for j in range(K):
                distance_element = y_it1_list[j]

                if distance_element[0] == self.PARENT_CODE:
                    y_it1 = self.parent_data[distance_element[1]]
                elif distance_element[1] == self.CHILD_CODE:
                    y_it1 = self.child_data[distance_element[1]]

                h1 += ((nutils.euclidian_distance(x_i, y_it1))**2)

        return h1/(N*K)

    def calculate_h2(self, N, K):
        h2 = 0
        for i in range(N):
            y_i = self.child_data[i]
            x_it2_list = self.parent_data_distance_list[i]

            for j in range(K):
                distance_element = x_it2_list[j]

                if distance_element[0] == self.PARENT_CODE:
                    x_it2 = self.parent_data[distance_element[1]]
                elif distance_element[0] == self.CHILD_CODE:
                    x_it2 = self.child_data[distance_element[1]]

                h2 += ((nutils.euclidian_distance(x_it2, y_i))**2)

            return h2/(N*K)

    def calculate_h3(self, N):
        h3 = 0
        for i in range(N):
            h3 += nutils.euclidian_distance(self.parent_data[i], self.child_data[i])
        return h3
