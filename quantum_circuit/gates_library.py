from unittest import TestCase
#from .functional_gates import Gate as FunctionalGate
from .matrix_gates import Gate as MatrixGate
import numpy as np

def create_rotation_gate(theta):
	"""
	:param theta: the rotation angle
	:return: phase rotation gate for 1 qubit, rotating by theta
	"""
	return MatrixGate(1, [[1, 0], [0, np.exp( theta*1j )]])



"""
Pauli X gate for 1 qubit
Matrix representation is:
/0 1\
\1 0/
 ___
"""
pauli_x = MatrixGate(1, [[0, 1], [1, 0]])

"""
Pauli Y gate for 1 qubit
Matrix representation is:
/0 -i\
\i  0/
"""
pauli_y = MatrixGate(1, [[0, -1j],[1j, 0]] )


"""
Pauli Z gate for 1 qubit
Matrix representation is:
/1  0\
\0 -1/
"""
pauli_z = MatrixGate(1, [[1,0],[0,-1]] )


"""
Hadamard gate for 1 qubit
"""
s2 = 2**(-.5)
hadamard = MatrixGate(1, [[s2, s2], [s2, -s2]])