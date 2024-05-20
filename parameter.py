# simulation length and height
SIMU_L = 0.04
SIMU_W = 0.02
SIMU_H = 0.02

# cell number
CELL_SIZE_X = 200
CELL_SIZE_Y = 100
LAYER_HEIGHT = 10  # length of height
STRIPE_NUM = 10
INTERVAL = 9

# cell size
DELTA_X = SIMU_L / CELL_SIZE_X
DELTA_Y = SIMU_W / CELL_SIZE_Y
DELTA_Z = SIMU_H / LAYER_HEIGHT

# heater parameter
LAMDA = 0.37
Rb = 0.0015   # 0.0015m
P_Max = 300
P_Min = 200

# physical entity
Tm = 600
Ta = 25
"""看一下对不对"""
Kt = 22.5E3       # W/K
""""""
h = 25            # W/(m^2 * K)
ALPHA = 5.632E-6  # m^2/s
VS = 0.2          # m/s
t = SIMU_L / (VS * CELL_SIZE_X)   # m/s
