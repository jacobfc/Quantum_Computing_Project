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
    """
    This methods uses the qutip module to visualise a qubit's superposition
    on a bloch sphere.
    
    This comes from the fact we can write any qubit in the form:
        
        |phi> = e^ia*(cos(x/2)|0> + (e^iy)*sin(x/2)|1> )
        
    and ignore the global phase e^ia, giving two angles (x,y) that we
    can use to visualise any qubit uniquely
    
    :param s: The state object depicting the qubit we want to visualise
    """
    
    #   Find the global phase to remove
    alpha = cmath.phase(s[0])

    A = s[0].real/np.cos(alpha)
    
    #   find the first angle
    theta = 2*math.acos(A)
    
    sinThetaTwo = np.sin(theta/2.0)
    
    #   Use this to find the second
    if (np.isclose(sinThetaTwo,s[1])):
        #   This means phi == 0
        phi = 0.0
    else:
        #   Do some stuff to find phi
        beta = cmath.phase(s[1])
        phi = beta - alpha
    
    #   Convert these angles to a unit vector using spherical coordinates
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    vec = [x, y, z]
    
    #   Plot this on a bloch sphere
    b = qutip.Bloch()
    b.add_vectors(vec)
    
    b.show()
    b.clear()
    return
