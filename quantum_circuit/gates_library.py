# from quantum_circuit.functional_gates import FunctionalGate
from quantum_circuit.gates import MatrixGate
from quantum_circuit.state import State
import numpy as np


def create_rotation_gate(theta):
    """
    :param theta: the rotation angle
    :return: phase rotation gate for 1 qubit, rotating by theta
    Matrix representation is:
    /1         0      \
    \0    e^(i*theta) /

    """
    return MatrixGate(1, [[1, 0], [0, np.exp(theta * 1j)]])


# Method to merge gates (using tensor product) into a larger one
def create_gate(qubit_count, gate_list):
    """
    :param qubit_count: number of qubits
    :param gate_list: list of applied gates
    :return: joint gate
    """
    gate = [[1]]
    for i in range(qubit_count):
        gate = np.kron(gate_list[i].matrix, gate)
    return MatrixGate(qubit_count, gate)


"""
Phase flip gate

Example for a key = 001
(the rightmost digit represents 0th qubit)
|1> -----Z-----
|0> --X--C--X--
|0> --X--C--X--

Need to add X gates around zero states
Middle line is controlled Z gate.
Does not mater to which qubit Z gate is applied.

Do all this or simply create an identity matrix and change [key][key] to -1?
"""


def phase_flip_gate(qubit_count, key):
    gate_list = []
    # add X gates around zero states
    for i in range(qubit_count):
        if (key & (0b1 << i)) == 0:
            gate_list.append(pauli_x)
        else:
            gate_list.append(identity)

    gate_X = create_gate(qubit_count, gate_list)
    gate_flip = MatrixGate.controlled_gate(qubit_count, pauli_z, [0],
                                           list(range(1, qubit_count)))
    return gate_X * gate_flip * gate_X


"""
Grover diffusion operator
Inverts around the mean
"""


def diffusion_gate(qubit_count):
    # create matrix to apply Hadamard to every qubit
    gate_list = [hadamard for i in range(qubit_count)]
    H_n = create_gate(qubit_count, gate_list)
    # create matrix to apply identity to every qubit
    gate_list = [identity for i in range(qubit_count)]
    I_n = create_gate(qubit_count, gate_list)
    # create the matrix Z0 = 2|0><0|-I
    zero_state = State.from_basis_state(qubit_count, 0)
    Z0 = MatrixGate(qubit_count,
                    2 * np.outer(zero_state, zero_state) - I_n.matrix)
    # diffusion gate is defined as H Z0 H
    return H_n * Z0 * H_n


"""
Empty gate - identity matrix
"""
identity = MatrixGate(1, [[1, 0], [0, 1]])

"""
Pauli X gate (aka NOT) for 1 qubit
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
pauli_y = MatrixGate(1, [[0, -1j], [1j, 0]])

"""
Pauli Z gate for 1 qubit
Matrix representation is:
/1  0\
\0 -1/
"""
pauli_z = MatrixGate(1, [[1, 0], [0, -1]])

"""
Hadamard gate for 1 qubit
"""
s2 = 2 ** (-.5)
hadamard = MatrixGate(1, [[s2, s2], [s2, -s2]])

"""
Phase gate for 1 qubit
Matrix repersentation is:
/1 0\
\0 i/
"""
phase = MatrixGate(1, [[1,0], [0,1j]])
