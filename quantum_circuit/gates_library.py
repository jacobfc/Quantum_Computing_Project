import numpy as np

from quantum_circuit.gates import MatrixGate
from quantum_circuit.state import State


def create_rotation_gate(theta):
    """
    :param theta: the rotation angle
    :return: phase rotation gate for 1 qubit, rotating by theta
    Matrix representation is:
    |1         0      |
    |0    e^(i*theta) |
    """
    return MatrixGate(1, [[1, 0], [0, np.exp(theta * 1j)]])


def qft(qubit_count):
    """
    Quantum Fourier Transform
    """
    N = 1 << qubit_count  # basis_size (number of rows and columns)
    omega = np.exp(2.0 * np.pi * 1j / N)  # Nth root of unity
    # Construct the operation matrix
    matrix = []
    for i in range(N):
        row = []
        for j in range(N):
            power = i * j
            row.append(omega ** power)
        matrix.append(row)
    matrix = np.array(matrix) / np.sqrt(N)  # Normalization factor
    return MatrixGate(qubit_count, matrix)


def phase_flip_gate(qubit_count, key):
    """ Phase flip gate

    Example for a key = 001
    (the rightmost digit represents 0th qubit)
    |1> -----Z-----
    |0> --X--C--X--
    |0> --X--C--X--

    Need to add X gates around zero states
    Middle line is controlled Z gate.
    Does not mater to which qubit Z gate is applied.
    """
    # more efficient (but less interesting) implementation:
    # create identity matrix and set [key][key] to -1
    gate_list = []
    # add X gates around zero states
    for i in range(qubit_count):
        if (key & (0b1 << i)) == 0:
            gate_list.append(pauli_x())
        else:
            gate_list.append(identity())

    gate_X = MatrixGate.join_gates(qubit_count, gate_list)
    gate_flip = MatrixGate.controlled_gate(qubit_count, pauli_z(), [0],
                                           list(range(1, qubit_count)))
    return gate_X * gate_flip * gate_X


def diffusion_gate(qubit_count):
    """
    Grover diffusion operator, inverts around the
    """
    # create matrix to apply Hadamard to every qubit
    gate_list = [hadamard() for i in range(qubit_count)]
    H_n = MatrixGate.join_gates(qubit_count, gate_list)
    # create matrix to apply identity to every qubit
    gate_list = [identity() for i in range(qubit_count)]
    I_n = MatrixGate.join_gates(qubit_count, gate_list)
    # create the matrix Z0 = 2|0><0|-I
    zero_state = State.from_basis_state(qubit_count, 0)
    Z0 = MatrixGate(qubit_count,
                    2 * np.outer(zero_state.amplitudes, \
                    zero_state.amplitudes) - I_n.matrix)
    # diffusion gate is defined as H Z0 H
    return H_n * Z0 * H_n


def identity():
    """
    Empty gate - identity matrix
    """
    return MatrixGate(1, [[1, 0], [0, 1]])


def pauli_x():
    """
    Pauli X gate (aka NOT) for 1 qubit
    Matrix representation is:
    |0 1|
    |1 0|
    """
    return MatrixGate(1, [[0, 1], [1, 0]])


def pauli_y():
    """
    Pauli Y gate for 1 qubit
    Matrix representation is:
    |0 -i|
    |i  0|
    """
    return MatrixGate(1, [[0, -1j], [1j, 0]])


def pauli_z():
    """
    Pauli Z gate for 1 qubit
    Matrix representation is:
    |1  0|
    |0 -1|
    """
    return MatrixGate(1, [[1, 0], [0, -1]])


def hadamard():
    """
    Hadamard gate for 1 qubit
    """
    s2 = np.power(np.complex128(2), (-.5))
    return MatrixGate(1, [[s2, s2], [s2, -s2]])


def phase():
    """
    Phase gate for 1 qubit
    Matrix representation is:
    |1 0|
    |0 i|
    """
    return MatrixGate(1, [[1, 0], [0, 1j]])
