from unittest import TestCase
from .functional_gates import Gate as FunctionalGate
from .matrix_gates import Gate as MatrixGate


s2 = 2**(-.5)
H = MatrixGate(1, 2, [[s2, s2], [s2, -s2]])


class TestGate(TestCase):
    def test_controlled_u(self):
        for Gate in (FunctionalGate,):
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
