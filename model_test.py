import numpy as np
from functions import *


def Execute(input_vector):
    thermal = Thermal()
    # init
    heat_loc = [INIT_X, INIT_Y, 0]  # STEP, STRIPE, LAYER
    P = P_Max

    thermal.Reset()
    Cost = 0
    count = 0

    assert len(input_vector) == LAYER_HEIGHT * STRIPE_NUM, " V num do not match stripe num"

    for layer in range(LAYER_HEIGHT):

        # layer begin
        heat_loc[0], heat_loc[1] = INIT_X, INIT_Y
        for stripe in range(STRIPE_NUM):

            # stripe begin
            heat_loc[0] = 0

            for step in range(CELL_SIZE_X):
                # Execute One Step
                thermal.Step(P, input_vector[count], heat_loc)

                cost_function = thermal.Cost_function(heat_loc)  # determine the cost function
                # thermal.Display(thermal.current_T)

                # Update Location
                heat_loc[0] += 1

                # Add cost function
                Cost += cost_function

            # one stripe is done
            heat_loc[1] += INTERVAL_Y  # 加上层间的距离，由道宽以及重叠率决定
            count += 1

        # one layer is done
        heat_loc[2] += 1
        thermal.reset()

        """ test for determine """
        # 300-800
        P -= (P_Max - P_Min) / STRIPE_NUM

    return Cost


if __name__ == "__main__":
    vector = np.random.uniform(V_MIN, V_MAX, 35)
    time1 = time.time()
    cost = Execute(vector)
    time2 = time.time()
    print(time2 - time1)
