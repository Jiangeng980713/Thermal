# simulation length and height
SIMU_L = 0.04
SIMU_W = 0.02
SIMU_H = 0.02

# cell number  150 x 75
CELL_SIZE_X = 150
CELL_SIZE_Y = 75
LAYER_HEIGHT = 10  # length of height
STRIPE_NUM = 10

# heater parameter
# HEATER_ROW, HEATER_COLUMN, HEATER_STD = 15, 15, 5
HEATER_ROW, HEATER_COLUMN = 10, 10    # 10 * 0.26um = 2.6 mm

# Starter location
INIT_X = 0
INIT_Y = 7       # 5 in the real world, in matrix 5-1
INTERVAL_X = 1   # distance between step
INTERVAL_Y = 10  # distance between stripe
GRADIENT_LENGTH = 3   # gradient regions about the optimization

# cell size
DELTA_X = SIMU_L / CELL_SIZE_X
DELTA_Y = SIMU_W / CELL_SIZE_Y
DELTA_Z = SIMU_H / LAYER_HEIGHT

# heater parameter
LAMDA = 0.5
Rb = 0.0015  # 0.0015m
P_Max = 600
P_Min = 300

# require for further update
V_MAX = 15E-3   # m/s -> 900 mm/min
V_MIN = 2.5E-3  # m/s -> 150 mm/min

# physical entity
Tm = 1600  # melting temperature for 316L
Ta = 300

Kt = 22.5  # W/mK
h = 25  # W/(m^2 * K)
ALPHA = 5.632E-6  # m^2/s
VS = 5E-3  # m/s -> 300 mm/min

t = SIMU_L / (VS * CELL_SIZE_X)  # 4E-2
TIME_SCALE = 75
