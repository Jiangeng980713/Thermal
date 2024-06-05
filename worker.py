import time

import numpy as np
import parameter
from functions import *
from optimize import *


def Execute(V_total):
    thermal = Thermal()
    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER
    P = P_Max
    V = V_total[0]
    thermal.Reset()
    Cost = 0

    for layer in range(LAYER_HEIGHT):
        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y
        time1 = time.time()
        for stripe in range(STRIPE_NUM):
            # stripe begin
            heat_loc[0] = 0

            for step in range(CELL_SIZE_X):

                # Execute One Step
                thermal.Step(P, V, heat_loc)
                # print('stripe, step', stripe, step)

                # if step - (step//70)*90 == 0:
                #      print('step', step, step//10)
                #      print('stripe, step', stripe, step)
                #      thermal.Display(thermal.previous_T)

                # Update Location
                heat_loc[0] += 1

            # one stripe is done
            heat_loc[1] += INTERVAL_Y

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

        """ test for determine"""
        P -= (P_Max - P_Min) / STRIPE_NUM

    return Cost

V = VS * np.ones(STRIPE_NUM * LAYER_HEIGHT)
Cost = Execute(V_total=V)

