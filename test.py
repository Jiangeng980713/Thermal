import numpy as np
a = np.zeros((4, 4))
print(a)
temp = a
temp[:, 1] = 1
print(temp)
print(a)
