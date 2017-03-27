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

from tests.testcase import BaseTestCase


class TestShor(BaseTestCase):
    def test_shor(self):
        # run test and see if errors are thrown
        main()


def main():
    """
    This is a test to outline a simulation of the quantum teleportation concept
    
    We produce a random superposition on one qubit (a) and use the circuit 
    detailed below to transport this qubit's state to another (c).
    
    |abc>
                                                        
    |a> = |0> --<produce state to send>-- ... -----c-H-M----c------------------
    |b> = |0> ---H-c--------------------- ... -----X---M-c---------------------
    |c> = |0> -----X--------------------- ... -----------X--Z--<observe state>-
                                           ^        ^    alter the phase to get original qubit
                  ^                        |      Isolate and measure a & b
                  |                    Take them any distance away
                  |
            Creating a pair of Bell States
    """
    
    #   Create a state to teleport
    s = State.from_basis_state(1,0)
    s = g_lib.hadamard()(s)
    s = g_lib.phase()(s)
    s = g_lib.create_rotation_gate(random.uniform(0,np.pi))(s)
    
    bloch.visualise_qubit(s)
    
    #   Store the initial state for comparison
    initial = [s[0], s[1]]
    
    
    #   Now consider the entire system
    s_full = State([s[0],0,0,0,s[1],0,0,0])
    
    # Create the EPR pair
    g_list = [g_lib.identity(), g_lib.hadamard(), g_lib.identity()]
    bell1 = MatrixGate.join_gates(3, g_list)
    bell2 = MatrixGate.controlled_gate(3, g_lib.pauli_x(), [0], [1])
    bell = bell2 * bell1
    
    s_full = bell(s_full)


    #   entangle our desired state with one of the EPR qubits
    invBell1 = MatrixGate.controlled_gate(3, g_lib.pauli_x(), [1], [2])
    g_list = [g_lib.identity(), g_lib.identity(), g_lib.hadamard()]
    invBell2 = MatrixGate.join_gates(3, g_list)
    invBell = invBell2 * invBell1
    
    s_full = invBell(s_full)
    
    #   Measure the state (this in fact measures the qubit we are wanting to
    #   keep in a superposition, but the method does not collapse the state
    #   unlike with a real system. This means we can 'pretend' one of the
    #   qubits is still in a superposition as follows
    m = s_full.random_measure_bs()
    
    #   Create a list to hold the amplitudes of the superposed qubit  
    s_new = [0.,0.]
    
    #   A series of statements to extract the qubit still in its superposition
    #   depending on the measurement of the other two bits (as this would 
    #   normally have collapsed)
    
    #   if m == 0 or 1 then our combined state is |000> or |001>
    if m==0 or m==1:
        do = 0 #    This tells us the value we need to 'send' in order to
               #    receive the original superposition (00 in binary)
        
        for i in range(s_full.basis_size):
            if i==0:
                s_new[0] = s_full[0]    #   probability of focus qubit being |0>
            elif i==1:
                s_new[1] = s_full[1]    #   probability of focus qubit being |1>
            else:
                s_full[i] = 0.  #   This manually collapses the state of two of the qubits
    
    #   if m == 2 or 3 then our combined state is |010> or |011>
    elif m==2 or m==3:
        do = 1  #   binary value 01
        
        for i in range(s_full.basis_size):
            if i==2:
                s_new[0] = s_full[2]
            elif i==3:
                s_new[1] = s_full[3]
            else:
                s_full[i] = 0.
    
    elif m==4 or m==5:
        do = 2  #   binary 10
        
        for i in range(s_full.basis_size):
            if i==4:
                s_new[0] = s_full[4]
            elif i==5:
                s_new[1] = s_full[5]
            else:
                s_full[i] = 0.
        
    else:
        do = 3  #   binary 11
        
        for i in range(s_full.basis_size):
            if i==6:
                s_new[0] = s_full[6]
            elif i==7:
                s_new[1] = s_full[7]
            else:
                s_full[i] = 0.
    
    #   Create the new state object that is the focus qubit
    s_new = State(s_new)
    
    #   normalise this qubit
    s_new = s_new/(s_new.norm())
    
    #   This is the section where we 'send' the two classical bits to the
    #   holder of the focus bit so they can retreive the original state.
    #   They do this by acting different gates on their qubit depending
    #   on the classical bits the are given
    if do==0:
        s_new = g_lib.identity()(s_new)
    
    elif do==1:
        s_new = g_lib.pauli_x()(s_new)
        
    elif do==2:
        s_new = g_lib.pauli_z()(s_new)
    
    else:
        s_new = g_lib.pauli_x()(s_new)
        s_new = g_lib.pauli_z()(s_new)
        
    #   Now, s_new should be exactly equivalent to the initial state
    assert np.isclose(s_new[0],initial[0])
    assert np.isclose(s_new[1],initial[1])
    
    
    bloch.visualise_qubit(s_new)
    
if __name__ == '__main__':
    main()
    