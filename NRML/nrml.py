"""
This file contains the neighborhood repulsed metric learning implementation as described in https://ieeexplore.ieee.org/document/6562692
Author: Md. Monowar Anjum
Email: monowaranjum@gmail.com
"""

import numpy as np
from numpy.core.multiarray import ndarray

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

        print("data sorting and initialization done")

    def get_present_distance_list(self):
        return self.parent_data_distance_list, self.child_data_distance_list

    def _process_(self, iteration_count, epsilon):
        w_prev = np.zeros((self.dimension, self.dimension // 2))
        w_now = np.zeros((self.dimension, self.dimension // 2))
        diff_array = np.zeros(iteration_count)
        for k in range(iteration_count):
            # Calculating H1, H2 and H3

            h1_temp = self.calculate_h1(self.sample_num, self.sample_num//2)
            h2_temp = self.calculate_h2(self.sample_num, self.sample_num//2)
            h3_temp = self.calculate_h3(self.sample_num)

            h_sum = h1_temp + h2_temp - h3_temp

            # Eigenvector decomposition part
            w_now = nutils.get_w_matrix(h_sum, self.dimension, self.dimension//2)
            self.update_distance_list(w_now)

            w_diff = w_now-w_prev
            abs_diff = np.sum(np.absolute(w_diff))
            diff_array[k] = abs_diff
            if abs_diff < epsilon:
                break
            w_prev = w_now
            print("Iteration done: "+str(k))
        return w_now, diff_array

    def calculate_h1(self, N, K):
        h1 = np.zeros((self.dimension, self.dimension))
        for i in range(N):
            x_i = self.parent_data[i]
            y_it1_list = self.child_data_distance_list[i]

            for j in range(K):
                distance_element = y_it1_list[j]

                if distance_element[0] == self.PARENT_CODE:
                    y_it1 = self.parent_data[distance_element[1]]
                elif distance_element[0] == self.CHILD_CODE:
                    y_it1 = self.child_data[distance_element[1]]
                else:
                    # If you find the next print statement executed , then something went wrong
                    print("Error: In H1 parent or child data not found.", str(distance_element[0]))
                    y_it1 = np.zeros(self.dimension)
                h1 += (nutils.column_to_row_vector_multiplication(x_i, y_it1))

        return h1/(N*K)

    def calculate_h2(self, N, K):
        h2 = np.zeros((self.dimension, self.dimension))
        for i in range(N):
            y_i = self.child_data[i]
            x_it2_list = self.parent_data_distance_list[i]

            for j in range(K):
                distance_element = x_it2_list[j]

                if distance_element[0] == self.PARENT_CODE:
                    x_it2 = self.parent_data[distance_element[1]]
                elif distance_element[0] == self.CHILD_CODE:
                    x_it2 = self.child_data[distance_element[1]]
                else:
                    print("Error: In H2 parent or child data not found. ", str(distance_element[0]))
                    x_it2 = np.zeros(self.dimension)

                h2 += (nutils.column_to_row_vector_multiplication(x_it2, y_i))

            return h2/(N*K)

    def calculate_h3(self, N):
        h3 = np.zeros((self.dimension, self.dimension))
        for i in range(N):
            h3 += nutils.column_to_row_vector_multiplication(self.parent_data[i], self.child_data[i])
        return h3/N

    def update_distance_list(self, updated_w):
        """
        This function updates the distance list after every iteration with the new output w matrix
        :param updated_w: The w matrix calculated from every iteration
        :return: Nothing. Just updates the list in place.
        """
        self.parent_data_distance_list.clear()
        self.child_data_distance_list.clear()

        for i in range(self.sample_num):
            __temp_p = self.parent_data[i]
            __temp_c = self.child_data[i]

            __temp_p_distance_list = []
            __temp_c_distance_list = []

            for k in range(self.sample_num):

                t_dist = nutils.get_mahalanbish_distance_variant(__temp_p, self.parent_data[k], updated_w)
                __temp_p_distance_list.append((self.PARENT_CODE, k, t_dist))

                t_dist = nutils.get_mahalanbish_distance_variant(__temp_p, self.child_data[k], updated_w)
                __temp_p_distance_list.append((self.CHILD_CODE, k, t_dist))

                t_dist = nutils.get_mahalanbish_distance_variant(__temp_c, self.parent_data[k], updated_w)
                __temp_c_distance_list.append((self.PARENT_CODE, k, t_dist))

                t_dist = nutils.get_mahalanbish_distance_variant(__temp_c, self.child_data[k], updated_w)
                __temp_c_distance_list.append((self.CHILD_CODE, k, t_dist))

            self.parent_data_distance_list.append(__temp_p_distance_list)
            self.child_data_distance_list.append(__temp_c_distance_list)

        for distance_list in self.parent_data_distance_list:
            distance_list.sort(key=lambda tup: tup[2])
        for distance_list in self.child_data_distance_list:
            distance_list.sort(key=lambda tup: tup[2])

        print("data processing done for updated w matrix.")

