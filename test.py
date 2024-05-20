import numpy as np
import parameter
from functions import *
from optimize import *

thermal = Thermal()
# V_total = Optimize()


def Execute(V_total):

    # init
    heat_loc = [0, 0, 0]  # STEP, STRIPE, LAYER
    P = P_Max
    V = V_total[0]
    thermal.reset()
    Cost = 0

    # one layer
    for episode in range(LAYER_HEIGHT):
        # one stripe
        heat_loc[0], heat_loc[1] = 0, 0
        for stripe in range(STRIPE_NUM):
            # one step
            heat_loc[0] = 0
            for step in range(CELL_SIZE_X):
                # Execute One Step
                thermal.Step(P, V, heat_loc)

                # cost function
                Cost += thermal.Cost_function(heat_loc)

                # Update Location
                heat_loc[0] += 1

                # one stripe is over
                if heat_loc[0] == CELL_SIZE_X:
                    heat_loc[1] += INTERVAL
                    continue

        if heat_loc[1] == STRIPE_NUM:
            # increase one layer
            heat_loc[2] += 1
            continue

        """ test for determine"""
        P -= (P_Max-P_Min)/STRIPE_NUM

    return Cost


V = VS * np.ones(STRIPE_NUM * LAYER_HEIGHT)
Cost = Execute(V_total=V)
