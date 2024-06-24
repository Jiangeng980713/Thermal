from functions import *
import multiprocessing


def Execute(id, V_total, q):
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
                cost_function = thermal.Cost_function(heat_loc)

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

    # output the cost_function to the list
    q.put({id: cost_function})


if __name__ == '__main__':
    q = multiprocessing.Queue()
    V = VS * np.ones(STRIPE_NUM * LAYER_HEIGHT)
    id = 0
    p = multiprocessing.Process(target=Execute, args=(id, V, q))
    p.start()
    p.join()
    results = [q.get()]
    print(results)

