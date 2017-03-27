# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:44:53 2017

@author: dcric
"""

import math
import cmath
import numpy as np
import qutip

def visualise_qubit(s):
    
    alpha = cmath.phase(s[0])

    A = s[0].real/np.cos(alpha)
    
    theta = 2*math.acos(A)
    
    sinThetaTwo = np.sin(theta/2.0)
    
    
    if (np.isclose(sinThetaTwo,s[1])):
        #   This means phi == 0
        phi = 0.0
    else:
        #   Do some stuff to find phi
        beta = cmath.phase(s[1])
        phi = beta - alpha
    
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    vec = [x, y, z]
    
    b = qutip.Bloch()
    b.add_vectors(vec)
    
    b.show()
    b.clear()
    return

def visualise(state, qubit):
    
    ### Do some stuff to isolate the qubit into a new State called s ###
    s = [0,1] ### Remove This ###
    visualise(s)