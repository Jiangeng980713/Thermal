# simulation length and height
SIMU_L = 0.03
SIMU_W = 0.0092
SIMU_H = 0.005
LAYER_HEIGHT = 5  # length of height
STRIPE_NUM = 7

# cell size 0.2mm / cell
CELL_SIZE_X = 175
CELL_SIZE_Y = 46    # 10 + 6 * 6

# heater parameter
HEATER_ROW, HEATER_COLUMN = 11, 11    # 11 * 0.2mm = 2.2 mm

# Starter location
INIT_X = 0
INIT_Y = 5       # 5 in the real world, in matrix 5-1
INTERVAL_X = 1   # distance between step
INTERVAL_Y = 3   # distance between stripe

""" further determination depends on optimization scene """
# GRADIENT_LENGTH = 3   # gradient regions about the optimization

# cell size
DELTA_X = SIMU_L / CELL_SIZE_X
DELTA_Y = SIMU_W / CELL_SIZE_Y
DELTA_Z = SIMU_H / LAYER_HEIGHT

P_Max = 600
P_Min = 300

# time between the heater actuated
TIME_SLEEP = 58

# require for further update
V_MAX = 10E-3   # m/s -> 600 mm/min -> 10mm/s -> 20ms (0.2mm per cell) -> 50 fps
V_MIN = 2.5E-3  # m/s -> 150 mm/min -> 2.5mm/s -> 80ms (0.2mm per cell) -> 12.5 fps

# physical entity
Tm = 2150       # melting temperature for 316L
Ta = 300

# heater parameter （变化这些需要在文章中说明，需要cite其他工作，或者在代码中直接改掉）
LAMDA = 0.5
Rb = 0.0012  # Rb 越小，温度越高
# parameters for materials （变化这些需要在文章中说明，需要cite其他工作，或者在代码中直接改掉）
Kt = 22.5  # W/mK
h = 25  # W/(m^2 * K)
ALPHA = 5.632E-6  # m^2/s

VS = 6E-3  # m/s -> 360 mm/min

# coverage speed
t = SIMU_L / (VS * CELL_SIZE_X)  # 4E-2
TIME_SCALE = 75    # 30+ is converged

# optimizer for PSO
THREAD_NUM = 12