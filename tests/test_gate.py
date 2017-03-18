import quantum_circuit
from quantum_circuit.functional_gates import FunctionalGate
from quantum_circuit.matrix_gates import MatrixGate

import numpy as np
from unittest import TestCase


s2 = 2**(-.5)
H = MatrixGate(1, [[s2, s2], [s2, -s2]])


class TestGate(TestCase):
    def test_controlled_u(self):
        print(H * MatrixGate(1, np.transpose(H.matrix)))

        for Gate in (FunctionalGate, MatrixGate):
            print(Gate)
            gate = Gate.controlled_u(3, H, [1], [0])

            print(gate.eval_bs(0))
            print(gate.eval_bs(1))
            print(gate.eval_bs(2))
            print(gate.eval_bs(3))
            print(gate.eval_bs(4))
            print(gate.eval_bs(5))
            print(gate.eval_bs(6))
            print(gate.eval_bs(7))

            # TODO: actually write a test here (instead of printing)


def test_controlled_u():
    print(H * MatrixGate(1, np.transpose(H.matrix)))

    for Gate in (FunctionalGate, MatrixGate):
        print(Gate)
        gate = Gate.controlled_u(3, H, [1], [0])

        print(gate.eval_bs(0))
        print(gate.eval_bs(1))
        print(gate.eval_bs(2))
        print(gate.eval_bs(3))
        print(gate.eval_bs(4))
        print(gate.eval_bs(5))
        print(gate.eval_bs(6))
        print(gate.eval_bs(7))

if __name__ == '__main__':
    import doctest
    doctest.testmod(quantum_circuit)
