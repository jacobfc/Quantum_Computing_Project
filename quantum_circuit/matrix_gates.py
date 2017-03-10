import numpy as np

from .mainframe import State
from .mainframe import Gate as AbstractGate
from .functional_gates import Gate as FunctionalGate

"""
Convention for the basis:
in a list of qubits [x_0, x_1, x_2] the leftmost is of least value,
the representation in the computational basis would be
|x_0 + 2 x_1 + 4 x_2>

Therefore |6> = |110> = [0, 1, 1].
"""


class Gate(AbstractGate):
    def __init__(self, qubit_count, basis_size, matrix):
        self._qubit_count = qubit_count
        self._basis_size = basis_size
        self.matrix = np.array(matrix, np.complex64)

    def eval_bs(self, basis_state, need_copy=True):
        if need_copy:
            return State(np.copy(self.matrix[:, basis_state]))
        else:
            return State(self.matrix[:, basis_state])

    @property
    def qubit_count(self):
        return self._qubit_count

    @property
    def basis_size(self):
        return self._basis_size

    @classmethod
    def controlled_u(cls, qubit_count, u, apply_qubits, control_qubits):
        """ Create a controlled-U gate, given the matrix and the used qubits.

        Example:
            qubit_count = 3
            u = H = [[1,1],[1,-1]] / sqrt(2)
            apply_qubits = [1]   # note for n apply_qubits H is 2^n x 2^n
            control_qubits = [0] # counting starts with 0

            what it does to the basis states (omitting normalization factors):
            |0> = |000> -> |000> = |0> (=[1,0,0,0,0,0,0,0])
            |1> = |001> -> |0>(|0> + |1>)|1> = |1> + |3> (=[0,.7,0,.7,0,0,0,0])
            |2> = |010> -> |010> = |2>
            |3> = |011> -> |0>(|0> - |1>)|1> = |1> - |3>
            |4> = |100> -> |100> = |4>
            |5> = |101> -> |1>(|0> + |1>)|1> = |5> + |7>
            |6> = |110> -> |110> = |6>
            |7> = |111> -> |1>(|0> - |1>)|1> = |5> - |7>

            so we can see the matrix representation of the whole gate would be
             /1   0   0   0   0   0   0   0  \
            | 0   s2  0   s2  0   0   0   0   |
            | 0   0   1   0   0   0   0   0   |
            | 0   s2  0   -s2 0   0   0   0   |
            | 0   0   0   0   1   0   0   0   |
            | 0   0   0   0   0   s2  0   s2  |
            | 0   0   0   0   0   0   1   0   |
             \0   0   0   0   0   s2  0   -s2/
            where s2 = 1/sqrt(2).

        Specification of the U matrix in relation with apply_qubits:
            u = [[0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1},
                 [0, 0, 1, 0]]
            apply_qubits = [3, 1]

            This represents applying a nor gate to the third qubit of the total
            gate and an identity gate (no gate; "wire") to the first
            (again counting from 0: 0th gate, 1st gate, 2nd gate, ...).
            In order to reproduce u from this statement, note that it is
            written in the computational basis. Qubit 3 of the gate is treated
            as 2^0 - valued, qubit 1 is 2^1 - valued.


        :param qubit_count: Dimensionality of the gate ("number of wires").
        :param u: Unitary matrix. Assumed to be given in computational basis,
            using the order as in apply_qubits.
        :param apply_qubits: List of integers, length must fit dimensionality
            of u. If all control gates are true, u is applied to these qubits.
        :param control_qubits: List of integers, specifying control qubits.
        :return: Gate representing the full operation.
        """
        fn_gate = FunctionalGate.controlled_u(qubit_count, u,
                                              apply_qubits, control_qubits)
        basis_size = 2 ** qubit_count

        mat = np.zeros((basis_size, basis_size), np.complex64)
        for bs in range(basis_size):
            mat[:, bs] = fn_gate.eval_bs(bs).amplitudes

        return Gate(qubit_count, basis_size, mat)

    def __call__(self, state):
        return np.dot(self.matrix, state.amplitudes)

    def __repr__(self):
        return self.matrix.__repr__()

    def __mul__(self, gate2):
        """ g1 * g2 is equivalent of saying first apply g2 then g1

        :param gate2: A gate.
        :return: A gate equivalent to the operation g1(g2(state)).
            The gate is a matrix gate if gate2 is a matrix gate,
            otherwise a functional gate is returned
        """
        if isinstance(gate2, Gate):
            return Gate(self.qubit_count, self.basis_size,
                        np.dot(self.matrix, gate2.matrix))
        else:
            return gate2 * self
