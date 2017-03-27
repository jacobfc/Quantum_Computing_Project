# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 19:45:00 2017

@author: dcric
"""

from quantum_circuit.state import State
import quantum_circuit.gates_library as g_lib
import quantum_circuit.bloch_sphere as bloch
from quantum_circuit.gates import MatrixGate
import numpy as np
import random

def main():
    """
    |abc>
    
    |a> = |0> --<produce state to send>------ ... -----c-H-M----c------------------
    |b> = |0> ---H-c------------------------- ... -----X---M-c---------------------
    |c> = |0> -----X------------------------- ... -----------X--Z--<observe state>-
                  ^
            ^     |
            |     Take them any distance away
            |
            Creating a pair of Bell States
    """
    
    # Keep the qubits in seperate State objects so we can measure and plot them ?? #
    
    # Create a state to teleport
    s = State.from_basis_state(1,0)
    s = g_lib.hadamard()(s)
    s = g_lib.phase()(s)
    s = g_lib.create_rotation_gate(random.uniform(0,np.pi))(s)
    
    bloch.visualise_qubit(s)
    
    initial = [s[0], s[1]]
    
    s_full = State([s[0],0,0,0,s[1],0,0,0])
    #print(s_full)
    
    #           |     0             0              0      >
    #                 ^             ^              ^
    #                 |             |              |
    #                 |             |              |
    g_list = [g_lib.identity(), g_lib.hadamard(), g_lib.identity()]
    bell1 = MatrixGate.join_gates(3, g_list)
    bell2 = MatrixGate.controlled_gate(3, g_lib.pauli_x(), [0], [1])
    bell = bell2 * bell1
    
    s_full = bell(s_full)

    
    invBell1 = MatrixGate.controlled_gate(3, g_lib.pauli_x(), [1], [2])
    g_list = [g_lib.identity(), g_lib.identity(), g_lib.hadamard()]
    invBell2 = MatrixGate.join_gates(3, g_list)
    invBell = invBell2 * invBell1
    
    
    s_full = invBell(s_full)
    
    
    m = s_full.random_measure_bs()
    
    s_new = [0.,0.]
    if m==0 or m==1:
        do = 0
        
        for i in range(s_full.basis_size):
            if i==0:
                s_new[0] = s_full[0]
            elif i==1:
                s_new[1] = s_full[1]
            else:
                s_full[i] = 0.
    
    elif m==2 or m==3:
        do = 1
        
        for i in range(s_full.basis_size):
            if i==2:
                s_new[0] = s_full[2]
            elif i==3:
                s_new[1] = s_full[3]
            else:
                s_full[i] = 0.
    
    elif m==4 or m==5:
        do = 2
        
        for i in range(s_full.basis_size):
            if i==4:
                s_new[0] = s_full[4]
            elif i==5:
                s_new[1] = s_full[5]
            else:
                s_full[i] = 0.
        
    else:
        do = 3
        
        for i in range(s_full.basis_size):
            if i==6:
                s_new[0] = s_full[6]
            elif i==7:
                s_new[1] = s_full[7]
            else:
                s_full[i] = 0.
    
    s_new = State(s_new)
    s_new = s_new/(s_new.norm())
    
    if do==0:
        s_new = g_lib.identity()(s_new)
    
    elif do==1:
        s_new = g_lib.pauli_x()(s_new)
        
    elif do==2:
        s_new = g_lib.pauli_z()(s_new)
    
    else:
        s_new = g_lib.pauli_x()(s_new)
        s_new = g_lib.pauli_z()(s_new)
        
    assert np.isclose(s_new[0],initial[0])
    assert np.isclose(s_new[1],initial[1])
    
    
    bloch.visualise_qubit(s_new)
    
main()
    