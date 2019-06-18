from NRML.nrml import NRML
import numpy as np
from NRML import  nrml_utils as nutils

a = np.random.rand(256,256)
b = np.random.rand(256,256)

obj = NRML(256, 256)
obj.set_data(a,b)
obj.initialize_data()

x_d, y_d = obj.get_present_distance_list()

h3_array = obj.calculate_h3(256)
print("H3 array done")

h1_array = obj.calculate_h1(256,128)
print("H1 array done")

h2_array = obj.calculate_h2(256,128)
print("H2 array done")

h = h1_array + h2_array - h3_array
print(h[0:10, 10:20])

w = nutils.get_w_matrix(h, 256, 20)
print(w[0:10, 0:10])
