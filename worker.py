import numpy as np
import parameter
from functions import *

thermal = Thermal()
heat_loc = [0, 0]
Layer_NUM = 0
P = 300
V_total = Optimize()
V = V_total[0]

# one layer
for episode in range(len(LAYER_HEIGHT)):
    # test
    for step in range(len(CELL_SIZE * CELL_SIZE)):
        # load V

        # Thermal diffusion
        heat_matrix = thermal.Diffusion(P, V, heat_loc)

        # Update Location
        heat_loc[0] += 1
        if heat_loc[0] == 9:
            heat_loc[1] += 1
            heat_loc[0] = 0
            V = V_total[episode * CELL_SIZE + step]

    # finish one layer
    Layer_NUM += 1
    P -= 20

Cost = Optimize.Cost_function(history)
