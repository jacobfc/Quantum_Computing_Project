import numpy as np

from quantum_circuit import State
from tests.testcase import BaseTestCase

""" Unittests for quantum_circuit.state

Run as 'python -m unittest tests.test_state' from outside the library folder.
"""


class TestState(BaseTestCase):
    def test_initialization(self):
        print("Running initialization test")
        state1 = State([1, 0, 0, 0])
        state2 = State.from_basis_state(2, 0)
        state3 = State.from_basis_state(3, 0)
        state4 = State.from_basis_state(2, 1)

        self.assertEqual(state1, state2)
        # should be commutative
        self.assertEqual(state2, state1)
        # both in state '0' but different basis
        self.assertNotEqual(state1, state3)
        self.assertNotEqual(state2, state3)
        # same basis, different state
        self.assertNotEqual(state1, state4)

    def test_norm(self):
        qsize = 4
        bsize = 1 << qsize

        # all basis states
        basis_states = [State.from_basis_state(qsize, bs) for bs in
                        range(bsize)]

        for bs in basis_states:
            self.assertEqual(bs.norm(), 1)

        for i in range(1, qsize):
            summed_state = (basis_states[0] + basis_states[i]) * (2 ** -.5)
            # need to use approximate equal due to float imprecision
            self.assertTrue(np.isclose(summed_state.norm(), 1))

    def test_add_sub_mul(self):
        state1 = State([1, 0, 0, 2])  # don't care about normalization here
        state2 = State([1, .2, 0, -2])
        state3 = State([0, -.2, 0, 0])
        state4 = State.from_basis_state(2, 0)

        self.assertEqual(state1 + state2 + state3, 2 * state4)
        self.assertEqual((state2 + state2) / 2, state2)

        self.assertEqual(state1 + state2 - state2, state1)
        self.assertEqual(2 * state2 - state2, state2)

        self.assertEqual(0 * state4 + 1 * state2, state2)
