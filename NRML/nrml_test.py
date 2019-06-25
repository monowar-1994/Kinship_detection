from NRML.nrml import NRML
import numpy as np

a = np.random.rand(256,256)
b = np.random.rand(256,256)

obj = NRML(256, 256)
obj.set_data(a, b)
obj.initialize_data()
r1, r2 = obj._process_(10, 0.0001)
print(r2)


# x_d, y_d = obj.get_present_distance_list()
#
# h3_array = obj.calculate_h3(256)
# print("H3 array done")
#
# h1_array = obj.calculate_h1(256,128)
# print("H1 array done")
#
# h2_array = obj.calculate_h2(256,128)
# print("H2 array done")
#
# h = h1_array + h2_array - h3_array
# w = nutils.get_w_matrix(h, 256, 20)
#
# f1 = np.random.rand(256)
# f2 = np.random.rand(256)
#
# dist = nutils.get_mahalanbish_distance_variant(f2,f2,w)
# print(dist)