from NRML.nrml import NRML
import numpy as np
from NRML import  nrml_utils as nutils

a = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
b = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])

obj = NRML(2,5)
obj.set_data(a,b)
obj.initialize_data()

x_d, y_d = obj.get_present_distance_list()

h3_array = obj.calculate_h3(5)
print(h3_array)

h1_array = obj.calculate_h1(5,3)
print(h1_array)

h2_array = obj.calculate_h2(5,3)
print(h2_array)

h = h1_array + h2_array - h3_array
print(h)

e_val, e_vec = np.linalg.eig(h)

print(e_val)
print(e_vec)

# a = np.array([[0,1,2,3,4,5]])
# b = np.array([[2,8,6,9,7,14]])
# c = (a.T).dot(b)
# d = nutils.column_to_row_vector_multiplication(a[0],b[0])
