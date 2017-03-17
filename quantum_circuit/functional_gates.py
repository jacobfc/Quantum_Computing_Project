import numpy as np
from .mainframe import Gate as AbstractGate
from .mainframe import State


def mul(iterator):
    res = 1
    for i in iterator:
        res *= i
    return res


def _clear_bits(basis_state, apply_qubits):
    return basis_state - sum(1 << i for i in apply_qubits) & basis_state


def _extract_sub_basis_state(basis_state, qubits):
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


def _insert_sub_bit_superpos(basis_size, state, insert_state, apply_qubits):
    """

    :param basis_size:
    :param state:
    :param insert_state: int, in basis of size 2 ** len(apply_qubits)
    :param apply_qubits:
    :return:
    """

    out_state_raw = np.zeros(basis_size, np.complex64)

    # set apply qubits to zero
    empty_apply = _clear_bits(state, apply_qubits)

    # iterate over all output states
    for k in range(len(insert_state)):
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


def _is_set(i, k):
    # not sure this is actually the most efficient implementation
    return (1 << i) & k != 0


class Gate(AbstractGate):
    def __init__(self, qubit_count, _eval_bs):
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

    @classmethod
    def multi_gate(cls, qubit_count, gate, apply_qubits):
        """ Apply gate to multiple qubits at once.

        :param qubit_count:
        :param apply_qubits:
        :param gate:
        :return:
        """

        def _eval_bs(basis_state):
            out_states = [gate(_extract_sub_basis_state(basis_state, [qi]))
                          for qi in apply_qubits]

            # compute amplitudes in computational basis according to
            # qubit order in apply_qubits
            # note here that boolean is implicitly cast to int
            out_state = [mul(out_states[i][_is_set(i, k)]
                             for i in range(len(out_states)))
                         for k in range(1 << len(apply_qubits))]

            return _insert_sub_bit_superpos(
                1 << qubit_count, basis_state, out_state, apply_qubits)

    @classmethod
    def controlled_u(cls, qubit_count, u, apply_qubits, control_qubits):
        """

        :param qubit_count:
        :param u: A gate.
        :param apply_qubits:
        :param control_qubits:
        :return:
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
            """ Apply the gate to one basis state.

            :param basis_state: Integer in [0, 2**qubit_count) representing
                the basis state in the computational basis.
            :return: An array representing the (possibly superposed)
                state obtained by applying the gate.
            """
            # if not all control qubits 1 => identity
            if basis_state & control_mask != control_mask:
                return State.from_basis_state(basis_size, basis_state)
            else:
                # Represent apply gates as a state in u's computational basis.
                # Since u's basis is a subset of the full basis,
                # and we handle a basis_state, this is also a basis state
                u_input_bs = _extract_sub_basis_state(basis_state, apply_qubits)
                # as opposed to u_input_bs (int) this is a full state
                u_out_state = u.eval_bs(u_input_bs)

                # now the result in u_out_state has to be incorporated with
                # the rest of the qubits (which remain unchanged)
                return _insert_sub_bit_superpos(
                    basis_size, basis_state, u_out_state, apply_qubits)

        return cls(qubit_count, _eval_bs)

    def __call__(self, state):
        # simple implementation, may be overridden
        return sum(state[k] * self.eval_bs(k) for k in range(self.basis_size))

    def __mul__(self, gate2):
        """ g1 * g2 is equivalent of saying first apply g2 then g1

        :param gate2: A gate.
        :return: A functional gate equivalent to the operation g1(g2(state)).
        """
        return Gate(self.qubit_count, lambda bs: self(gate2.eval_bs(bs)))
