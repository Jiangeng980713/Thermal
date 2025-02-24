# simulation length and height
SIMU_L = 0.03
SIMU_W = 0.0092
SIMU_H = 0.005
LAYER_HEIGHT = 5  # length of height
STRIPE_NUM = 7
MIDDLE_LAYER = 3

# cell size 0.2mm / cell
CELL_SIZE_X = 175
CELL_SIZE_Y = 46  # 10 + 6 * 6

# heater parameter
HEATER_ROW, HEATER_COLUMN = 11, 11  # 11 * 0.2mm = 2.2 mm

# Starter location
INIT_X = 0
INIT_Y = 5  # 5 in the real world, in matrix 5-1
INTERVAL_X = 1  # distance between step
INTERVAL_Y = 6  # distance between stripe

""" further determination depends on optimization scene """
# GRADIENT_LENGTH = 3   # gradient regions about the optimization

# cell size
DELTA_X = SIMU_L / CELL_SIZE_X
DELTA_Y = SIMU_W / CELL_SIZE_Y
DELTA_Z = SIMU_H / LAYER_HEIGHT

P_Max = 700
P_Min = 400

# time between the heater actuated
TIME_SLEEP = 58

# require for further update
V_MAX = 10E-3  # m/s -> 600 mm/min -> 10mm/s -> 20ms (0.2mm per cell) -> 50 fps
V_MIN = 2.5E-3  # m/s -> 150 mm/min -> 2.5mm/s -> 80ms (0.2mm per cell) -> 12.5 fps

# physical entity
Tm = 2150  # melting temperature for 316L
Ta = 300

# heater parameter
LAMDA = 0.5
Rb = 0.0012  # Rb 越小，温度越高

# parameters for materials
Kt = 22.5  # W/mK
h = 25  # W/(m^2 * K)
ALPHA = 5.632E-6  # m^2/s

VS = 6E-3  # m/s -> 360 mm/min

# coverage speed
t = SIMU_L / (VS * CELL_SIZE_X)  # 4E-2
TIME_SCALE = 75  # 30+ is converged

# optimizer for PSO
THREAD_NUM = 12

# hand-pick starters
HEAT_STARTER = [157, 232, 307, 382, 457, 533, 608, 684, 756, 829, 903, 977, 1052, 1126, 1200, 1275, 1348, 1420, 1494, 1567, 1641, 1715, 1788, 1860, 1934, 2005, 2074, 2146, 2212, 2281, 2350, 2419, 2488, 2559, 2632, 2704]
COLD_STARTER = [211, 287, 362, 437, 513, 587, 663, 738, 809, 883, 957, 1032, 1106, 1180, 1255, 1328, 1401, 1475, 1548, 1621, 1695, 1770, 1841, 1915, 1986, 2056, 2127, 2194, 2265, 2330, 2401, 2471, 2543, 2614, 2686]

# calibration parameter
X_TRANS = 1
Y_TRANS = 1
Z_TRANS = 0.1
BODY_OFFSET = 50
NOTE = None