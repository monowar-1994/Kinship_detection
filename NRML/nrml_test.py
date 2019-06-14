from NRML.nrml import NRML
import numpy as np

a = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
b = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]])

obj = NRML(2,5)
obj.set_data(a,b)
obj.initialize_data()

x_d, y_d = obj.get_present_distance_list()

for list_item in x_d:
    print(list_item)
    print('\n\n')

for list_item in y_d:
    print(list_item)
    print('\n\n')
