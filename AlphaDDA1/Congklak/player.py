#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np

class Random_player():
    def __init__(self):
        pass
    def action(self, g):
        valid_moves = g.Get_valid_moves()
        return np.random.choice(valid_moves)
