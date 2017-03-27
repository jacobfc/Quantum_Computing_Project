from quantum_circuit.state import State

import numpy as np
import abc


class Gate(metaclass=abc.ABCMeta):
    def __init__(self):
        """ Initialize a gate matrix.

        Extending classes may freely use a different signature for this method.

        self._dtype must be set to a complex number type, that is used in
        creating and computing output states (for the amplitudes).
        """
        self._dtype = None  # must be a complex number type
        raise NotImplementedError()

    @property
    @abc.abstractproperty
    def qubit_count(self):
        """ Number of qubits (in classical analogy: wires). """
        raise NotImplementedError()

    @property
    @abc.abstractproperty
    def basis_size(self):
        """ Number of basis states, same as 2 ^ qubit_count. """
        raise NotImplementedError()

    @property
    def dtype(self):
        """ Specifies the data type used in constructing states and matrices.

        The property (attribute) *Gate._dtype must be set by child classes.
        """
        return self._dtype

    @abc.abstractmethod
    def eval_bs(self, basis_state):
        """ Apply gate to basis state as input.

        :type basis_state: int
        :param basis_state: Integer in [0, 2**qubit_count) representing
            the basis state in the computational basis.
        :return: State
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, state):
        """ Call a gate to act on a state.

        Example using gates library:
            >>> from quantum_circuit.gates_library import identity
            >>> gate_id = identity()
            >>> print(gate_id(State([0, 1])))
            1.000 |1>

        :type state: State
        :param state: The State for the gate to act on.
        :return: The State obtained by applying the gate's operation
            on the input state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __mul__(self, gate2):
        """ g1 * g2 returns a gate that first applies g2, then g1.

        :type gate2: Gate
        :param gate2: A gate of same size as self.
        :return: A gate equivalent to the operation g1(g2(state)).
            The gate is a matrix gate if gate2 is a matrix gate,
            otherwise a functional gate is returned
        """
        raise NotImplementedError()

    @classmethod
    def from_eval_bs(cls, qubit_count, _eval_bs, dtype=np.complex128):
        """ Construct a gate given eval_bs (tells how gate acts on basis states)

        Must be overridden in classes that inherit from Gate

        :type qubit_count: int
        :type _eval_bs: function
        :type dtype: type
        :param qubit_count: Number of qubits.
        :param _eval_bs: Method specifying the gate's output state, given
            a basis state (int) as input.
        :param dtype: Data type used in states and matrices (complex number).
        :return: Gate
        """
        # by default use functional gate here (least overhead)
        return FunctionalGate.from_eval_bs(qubit_count, _eval_bs, dtype=dtype)

    @classmethod
    def multi_gate(cls, qubit_count, gate, apply_qubits):
        """ Apply gate to multiple qubits at once.

        Example:
            Apply hadamard gate to the first two qubits
            >>> import quantum_circuit.gates_library as g_lib
            >>> g = Gate.multi_gate(3, g_lib.hadamard(), [0, 1])
            >>> out1 = g(State.from_basis_state(3, 4))
            >>> out2 = g(State.from_basis_state(3, 0))
            >>> print(out1)
            0.500 |4> + 0.500 |5> + 0.500 |6> + 0.500 |7>
            >>> print(out2)
            0.500 |0> + 0.500 |1> + 0.500 |2> + 0.500 |3>

        :type qubit_count: int
        :type gate: Gate
        :type apply_qubits: [int]
        :param qubit_count: Number of qubits of the gate to be created.
        :param gate: 1 - qubit gate, to be applied to multiple qubits at once.
        :param apply_qubits: Labels of qubits to apply the gate to.
        :return: Gate
        """
        def _eval_bs(basis_state):
            out_states = [gate.eval_bs(_extract_sub_bs(basis_state, [qi]))
                          for qi in apply_qubits]

            # compute amplitudes in computational basis according to
            # qubit order in apply_qubits
            # note here that boolean is implicitly cast to int
            out_state = [mul(out_states[i][int(_is_set(i, k))]
                             for i in range(len(out_states)))
                         for k in range(1 << len(apply_qubits))]

            return _insert_sub_bit_superpos(
                1 << qubit_count, basis_state, State(out_state), apply_qubits)

        return cls.from_eval_bs(qubit_count, _eval_bs)

    @classmethod
    def controlled_gate(cls, qubit_count, gate, apply_qubits, control_qubits,
                        dtype=np.complex128):
        """ Create a controlled gate, given the gate and the used qubits.

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
            |1   0   0   0   0   0   0   0  |
            |0   s2  0   s2  0   0   0   0  |
            |0   0   1   0   0   0   0   0  |
            |0   s2  0   -s2 0   0   0   0  |
            |0   0   0   0   1   0   0   0  |
            |0   0   0   0   0   s2  0   s2 |
            |0   0   0   0   0   0   1   0  |
            |0   0   0   0   0   s2  0   -s2|
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


        :type qubit_count: int
        :type gate: Gate
        :type apply_qubits: [int]
        :type control_qubits: [int]
        :type dtype: type
        :param qubit_count: Dimensionality of the gate ("number of wires").
        :param gate: Unitary matrix. Assumed to be given in computational basis,
            using the order as in apply_qubits.
        :param apply_qubits: List of integers, length must fit dimensionality
            of u. If all control gates are true, u is applied to these qubits.
        :param control_qubits: List of integers, specifying control qubits.
        :param dtype: Data type to use in states and matrices.
        :return: MatrixGate representing the full operation.
        """
        # make sure each qubit is mentioned at most once
        control_qubits_s = set(control_qubits)
        apply_qubits_s = set(apply_qubits)
        assert len(control_qubits_s) == len(control_qubits)
        assert len(apply_qubits_s) == len(apply_qubits)
        assert control_qubits_s.isdisjoint(apply_qubits_s)

        # gate parameter
        basis_size = 1 << qubit_count

        # mask for control gates
        control_mask = sum(1 << i for i in control_qubits)

        def _eval_bs(basis_state):
            # if not all control qubits 1 => identity
            if basis_state & control_mask != control_mask:
                return State.from_basis_state(qubit_count, basis_state)
            else:
                # Represent apply gates as a state in gate's computational basis
                # Since gate's basis is a subset of the full basis,
                # and we handle a basis_state, this is also a basis state
                u_input_bs = _extract_sub_bs(basis_state, apply_qubits)
                # as opposed to u_input_bs (int) this is a full state
                u_out_state = gate.eval_bs(u_input_bs)

                # now the result in u_out_state has to be incorporated with
                # the rest of the qubits (which remain unchanged)
                return _insert_sub_bit_superpos(
                    basis_size, basis_state, u_out_state, apply_qubits, dtype)

        return cls.from_eval_bs(qubit_count, _eval_bs, dtype=dtype)


class FunctionalGate(Gate):
    """ Concrete implementation of a gate.

    The implementation is based on storing function to map basis states to
    full states these are transformed into by applying the gate to it.

    FunctionalGates are most easily constructed using the static methods
    in the Gates class (can be called on FunctionalGate to explicitly specify
    the gate class).
    """
    @classmethod
    def from_eval_bs(cls, qubit_count, _eval_bs, dtype=None):
        """ See Gate.from_eval_bs.

        :type qubit_count: int
        :type _eval_bs: function
        :type dtype: type
        :param qubit_count: Number of qubits.
        :param _eval_bs: Method specifying the gate's output state, given
            a basis state (int) as input.
        :param dtype: Data type used in states and matrices (complex number).
        :return: Gate
        """
        return cls(qubit_count, _eval_bs, dtype=dtype)

    def __init__(self, qubit_count, _eval_bs, dtype=np.complex128):
        """ Create a Functional gate from eval_bs function (see Gate.eval_bs).

        :type qubit_count: int
        :type _eval_bs: function
        :type dtype: type
        :param qubit_count: Number of qubits.
        :param _eval_bs: Function mapping basis state to output State
        :param dtype: Data type for constructing states (amplitudes)
        """
        self._dtype = dtype
        self._basis_size = 1 << qubit_count
        self._qubit_count = qubit_count
        self._eval_bs = _eval_bs

    def eval_bs(self, basis_state):
        return self._eval_bs(basis_state)

    @property
    def qubit_count(self):
        return self._qubit_count

    @property
    def basis_size(self):
        return self._basis_size

    def __call__(self, state):
        """ Calls a gate on state.
        
        :param state: State which the gate is acting on
        :return: The result of the Gate acting on the State
        """
        return sum(state[k] * self.eval_bs(k) for k in range(self.basis_size))

    def __mul__(self, gate2):
        return FunctionalGate(self.qubit_count,
                              lambda bs: self(gate2.eval_bs(bs)))


class MatrixGate(Gate):
    @classmethod
    def from_eval_bs(cls, qubit_count, _eval_bs, dtype=np.complex128):
        basis_size = 1 << qubit_count
        # Start with an empty (basis_size x basis_size) matrix
        mat = np.zeros((basis_size, basis_size), dtype)
        for bs in range(basis_size):
            mat[:, bs] = _eval_bs(bs).amplitudes

        return cls(qubit_count, mat, dtype=dtype)

    @classmethod
    def join_gates(cls, qubit_count, gate_list):
        """ Method to merge matrix gates (using tensor product) into larger one

        :param qubit_count: number of qubits
        :param gate_list: list of applied gates
        :return: joint gate
        """
        gate = [[1]]
        for i in range(qubit_count):
            gate = np.kron(gate_list[i].matrix, gate)
        return cls(qubit_count, gate)

    def __init__(self, qubit_count, matrix, dtype=np.complex128):
        self._dtype = dtype
        self._qubit_count = qubit_count
        self._basis_size = 1 << qubit_count
        self.matrix = np.array(matrix, dtype)

        # If there are N basis states the matrix for the gate be an N x N matrix
        assert self._basis_size == self.matrix.shape[0]
        assert self._basis_size == self.matrix.shape[1]

    def eval_bs(self, basis_state):
        return State(self.matrix[:, basis_state])

    @property
    def qubit_count(self):
        return self._qubit_count

    @property
    def basis_size(self):
        return self._basis_size

    def __call__(self, state):
        """ Allows a Gate to be called on a State object
        :param state: the State object the gate is being called on
        :return: the State object resulting from the operation of the gate
        """
        return State(np.dot(self.matrix, state.amplitudes))

    def __repr__(self):
        return self.matrix.__repr__()

    def __mul__(self, gate2):
        if not isinstance(gate2, MatrixGate):
            return gate2 * self

        return MatrixGate(self.qubit_count, np.dot(self.matrix, gate2.matrix))

    def __sub__(self, gate2):
        return self.matrix - gate2.matrix


def mul(iterator):
    """ Equivalent to built-in sum for multiplication

    Example: mul([2, 1,3]) == 2 * 1 * 3 == 6

    :param iterator:
    :return:
    """
    res = 1
    for i in iterator:
        res *= i
    return res


def _is_set(i, k):
    """

    :param i: the index of the bit
    :param k:
    :return:
    """
    # not sure this is actually the most efficient implementation
    return (1 << i) & k != 0


def _clear_bits(basis_state, apply_qubits):
    return basis_state - (sum(1 << i for i in apply_qubits) & basis_state)


def _extract_sub_bs(basis_state, qubits):
    """ Extract state of qubits in specified order, given in computational basis

    Since the input is in basis state, and the basis states of system only
    containing the sublist of qubits are a subset of the full basis,
    the state we look for is a basis state as well. This means we can
    return an integer here, instead of a full state.

    :param basis_state:
    :param qubits:
    :return: Integer, representing state of
    """
    return sum(1 << i
               for i in range(len(qubits))
               # if i-th apply qubit is set
               if basis_state & (1 << qubits[i]) != 0)


def _insert_sub_bit_superpos(basis_size, state, insert_state, apply_qubits,
                             dtype=np.complex128):
    """

    :param basis_size:
    :param state:
    :param insert_state: int, in basis of size 2 ** len(apply_qubits)
    :param apply_qubits:
    :return:
    """
    out_state_raw = np.zeros(basis_size, dtype)

    # set apply qubits to zero
    empty_apply = _clear_bits(state, apply_qubits)

    # iterate over all output states
    for k in range(insert_state.basis_size):
        # transfer bit occupation of basis state from u's basis
        # back to the full basis
        set_apply = sum(1 << apply_qubits[i]  # value of ith qubit
                        # iterate over apply qubits
                        for i in range(len(apply_qubits))
                        # for first bit, add 2**0 every second entry
                        # for nth bit, add 2**n every 2**n-th entry
                        if (i + 1) & k)  # set 1 every 2nd, 4th, ...
        out_state_raw[set_apply + empty_apply] = insert_state[k]

    return State(out_state_raw)
