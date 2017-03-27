import quantum_circuit.gates
from quantum_circuit.gates import State
import quantum_circuit.gates_library as g_lib
import numpy as np
import math

from tests.testcase import BaseTestCase


class TestShor(BaseTestCase):
    def test_shor(self):
        # run test and see if errors are thrown
        main()


def main():
    # arbitrary number of qubits
    qubit_count = 6
    basis_size = 1 << qubit_count
    # arbitrary key: 1000 - 1 gives the marked state
    key = 0b000111
    # optimal number of iterations of Grover step
    n = math.floor(math.pi / (4 * math.asin(basis_size ** (-1 / 2))))
    # initial state
    state = State.from_basis_state(qubit_count, 0)
    # matrix to apply the phase flip
    flip_gate = g_lib.phase_flip_gate(qubit_count, key)
    # matrix to apply diffusion
    diff_gate = g_lib.diffusion_gate(qubit_count)
    # matrix to apply Hadamard to every qubit
    gate_list = [g_lib.hadamard() for i in range(qubit_count)]
    H_n = quantum_circuit.MatrixGate.join_gates(qubit_count, gate_list)

    # create superposition by applying Hadamard to every qubit
    state = H_n(state)
    print(flip_gate)
    print(diff_gate)
    print(H_n)
    # perform Grover step required number of times
    full = flip_gate * diff_gate
    for i in range(n - 1):
        full *= flip_gate * diff_gate
    # apply the matrix to get the final state
    final = full(state)
    print(final)


if __name__ == '__main__':
    main()
