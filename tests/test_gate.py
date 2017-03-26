from quantum_circuit import MatrixGate
from quantum_circuit import FunctionalGate
from quantum_circuit import Gate
from quantum_circuit.gates_library import hadamard

from tests.testcase import BaseTestCase


class TestGate(BaseTestCase):
    def generic_controlled_u_example(self, gate_type):
        gate = gate_type.controlled_gate(3, hadamard, [1], [0])

        # check if type is correct if we explicitly created
        # a functional / matrix gate
        if gate_type != Gate:
            self.assertEqual(type(gate), gate_type)

        s2 = gate.dtype(2 ** -.5)
        # This example is the same as in the docstring of Gate.controlled_gate
        self.assertEqual(gate.eval_bs(0), [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(gate.eval_bs(1), [0, s2, 0, s2, 0, 0, 0, 0])
        self.assertEqual(gate.eval_bs(2), [0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(gate.eval_bs(3), [0, s2, 0, -s2, 0, 0, 0, 0])
        self.assertEqual(gate.eval_bs(4), [0, 0, 0, 0, 1, 0, 0, 0])
        self.assertEqual(gate.eval_bs(5), [0, 0, 0, 0, 0, s2, 0, s2])
        self.assertEqual(gate.eval_bs(6), [0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(gate.eval_bs(7), [0, 0, 0, 0, 0, s2, 0, -s2])

    def test_controlled_u_matrix(self):
        self.generic_controlled_u_example(MatrixGate)

    def test_controlled_u_functional(self):
        self.generic_controlled_u_example(FunctionalGate)

    def test_controlled_u_default(self):
        # controlled_gate can also be called on the (partially abstract)
        # Gate class, so the end user could decide to only use Gate
        # and not bother with Functional/Matrix implementation.
        self.generic_controlled_u_example(Gate)
