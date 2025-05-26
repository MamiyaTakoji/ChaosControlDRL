# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:09:46 2025

@author: Mamiya
"""
import numpy as np

from gym.spaces import MultiDiscrete

M = MultiDiscrete([3,4,5])
e = 0.3
TotalAction = []
for subActionSpace in M.nvec:
    
    print(subActionSpace)


















