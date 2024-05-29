# simulation length and height
SIMU_L = 0.04
SIMU_W = 0.02
SIMU_H = 0.02

# cell number
CELL_SIZE_X = 200
CELL_SIZE_Y = 100
LAYER_HEIGHT = 10  # length of height
STRIPE_NUM = 10

# heater parameter
# HEATER_ROW, HEATER_COLUMN, HEATER_STD = 15, 15, 5
HEATER_ROW, HEATER_COLUMN = 15, 15

# Starter location
INIT_X = 0
INIT_Y = 4  # 5 in the real world, in matrix 5-1
INTERVAL_X = 1  # distance between step
INTERVAL_Y = 10  # distance between stripe

# cell size
DELTA_X = SIMU_L / CELL_SIZE_X
DELTA_Y = SIMU_W / CELL_SIZE_Y
DELTA_Z = SIMU_H / LAYER_HEIGHT

# heater parameter
LAMDA = 0.37
Rb = 0.0015  # 0.0015m
P_Max = 300
P_Min = 200

# physical entity
Tm = 1600  # melting temperature for 316L
Ta = 300

Kt = 22.5  # W/mK
h = 25  # W/(m^2 * K)
ALPHA = 5.632E-6  # m^2/s
VS = 5E-3  # m/s

t = SIMU_L / (VS * CELL_SIZE_X)  # 4E-2
TIME_SCALE = 100
