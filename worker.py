import numpy as np
import parameter
from functions import *



thermal = Thermal()



for episode in range(len(LAYER_HEIGHT)):

    # start one step
    for step in range(len(CELL_SIZE * CELL_SIZE)):
        print('a')
