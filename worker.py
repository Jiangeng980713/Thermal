import numpy as np
import parameter
from functions import *
from optimize import *

thermal = Thermal()
V_total = Optimize()

def Exceute(V_total):
    # inti
    heat_loc = [0, 0, 0]
    P = 300  # size: CELL_SIZE * LAYER_HEIGHT
    V = V_total[0]
    Thermal.reset()
    Cost = 0

    # one layer
    for episode in range(len(LAYER_HEIGHT)):
        for step in range(len(CELL_SIZE * CELL_SIZE)):

            # One step
            thermal.Step(P, V, heat_loc)

            """ 计算一种 cost function"""

            # Update Location
            heat_loc[0] += 1

            # one stripe is over
            if heat_loc[0] == 9:
                heat_loc[1] += 1
                heat_loc[0] = 0
                V = V_total[episode * CELL_SIZE + step]

        # one layer is done
        heat_loc[2] += 1

        """ test for determine"""
        P -= 20


    return Cost
