# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:51:36 2022

all functions, that are handy to externalize

@author: Till
"""
import numpy as np


#round a to the next power of two if a != 0
#return 0 if a == 0
def env_pot2(a):
    
    a = int(a)
    if a == 0:
        return 0
    
    elif a > 0:
        i = -1
        while ( np.bitwise_and(i, a) ) !=0:
            #print ('grow')
            i *= 2    
        return -i     

    else: # a < 0:
        return -env_pot2(-a)

