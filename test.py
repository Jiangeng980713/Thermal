import numpy as np
import parameter
from functions import *
from optimize import *

thermal = Thermal()
# V_total = Optimize()

def Exceute(V_total):
    # inti
    heat_loc = [0, 0, 0]
    P = 300  # size: CELL_SIZE * LAYER_HEIGHT
    V = V_total[0]
    thermal.reset()
    Cost = 0

    # one layer
    for episode in range(LAYER_HEIGHT):
        # one stripe
        for stripe in range(STRIPE_NUM):
            # one step
            for step in range(CELL_SIZE):

                thermal.Step(P, V, heat_loc)

                """ 计算一种 cost function"""

                # Update Location
                heat_loc[0] += 1

                # one stripe is over
                if heat_loc[1] == CELL_SIZE:
                    heat_loc[1] += 1
                    heat_loc[0] = 0

        print('episode', episode)

        # one layer is done
        heat_loc[0] = 0
        heat_loc[1] = 0
        heat_loc[2] += 1

        """ test for determine"""
        P -= 20

    return Cost

V = VS * np.ones(CELL_SIZE*LAYER_HEIGHT)

Cost = Exceute(V_total=V)
