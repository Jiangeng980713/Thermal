import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

T = np.load("thermal_T_110.npy")

plt.imshow(T, cmap='jet')
plt.colorbar(label='Temperature (K)')
plt.xlabel('Y index')
plt.ylabel('X index')
plt.title('Temperature Field at Step 1410')

plt.show()