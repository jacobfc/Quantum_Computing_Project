import numpy as np


class State(object):
    def __init__(self, initial_state, dtype=np.complex128):
        """ Initialize state given a set of amplitudes.

        Example:
            >>> s = State([1, 0, 0, 0])
            >>> s
            array([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j])
            >>> print(s)
            1.000 |0>

        :param initial_state: Iterable, specifying amplitude for
            each basis state.
        :param dtype: Numeric type to be used for amplitudes.
        """
        # The state is represented as amplitudes of constituent basis states.
        self.amplitudes = np.array(initial_state, dtype)
        self.basis_size = len(self.amplitudes)
        # For n qubits, there are 2**n basis states
        self.qubit_count = int(np.log2(self.basis_size))

        # assert initial_state has a power of two number of entries
        # note: 1 << n == 2 ** n
        assert 1 << self.qubit_count == self.basis_size

    def norm(self):
        """ Returns the sum of the squared amplitudes of the state.

        :return: float, norm of the state vector
        """
        # .real is necessary because norm should be a float
        return np.sqrt(np.conj(self).dot(self).real)

    def is_normalized(self):
        return np.isclose(self.norm(), 1)

    def prob_of_bs(self, bs, normalize=True):
        """

        :param bs: The basis state of interest.
        :param normalize: Normalize state before calculating amplitudes.
        :return: The probability of the system being in the basis state
            of interest.
        """
        if normalize:
            return np.square(abs(self[bs])) / np.conj(self).dot(self).real
        else:
            return np.square(abs(self[bs]))

    def prob_of_state(self, state, normalize=True):
        """

        :param state: The state of interest (does NOT have to be a basis state,
            can be any State object)
        :param normalize: Normalize state before calculating amplitudes.
        :return: The probability of the system being in the state of interest
        """
        if normalize:
            # doing the calculation like this makes it more accurate
            norm = np.conj(self).dot(self).real * np.conj(state).dot(state).real
            return np.square(abs(np.conj(self).dot(state))) / norm
        else:
            return np.square(np.conj(self).dot(state).real)

    def random_measure_bs(self):
        """ Select a basis state with probabilities according to amplitudes.

        Make a random choice on the basis states each weighted with
        the probability of finding the system in that state
        (i.e. amplitude squared).

        To illustrate the implementation, consider the (2-qubit) state
        with amplitudes [0.25, 0.25, 0, 0.5] = 0.25|0> + 0.25|1> + 0.5|3>.
        The first step, conceptually, is to split the real numbers
        between 0 and 1 into chunks with sizes according to the probabilities
        of the basis states:
        0.0   0.25   0.5   0.75 1.0
        |     |      |     |    |
        00000|11111||33333 33333
        A random number (in [0, 1)) is generated and the basis state chosen
        is the one to which the block belongs, the random number is found in.
        By making blocks belonging to one basis state larger,
        it gets more probable this basis state will be measured.

        :return: The basis state which the state was measured to be in
        """

        sample = np.random.random_sample()  # random number in [0, 1)
        bs = 0  # start "checking" if it's in the 0th basis state
        acc = self.prob_of_bs(bs)
        # if we haven't measured that state, keep checking the next states
        while acc < sample:
            bs += 1
            acc += self.prob_of_bs(bs)
        return bs

    @classmethod
    def from_basis_state(cls, qubit_count: int, basis_state: int,
                         dtype: type = np.complex128):
        """ Creates a State object that is a basis state, given its label.

        :param qubit_count: Number of qubits in the state
        :param basis_state: The basis state to create
        :param dtype: Numeric type to use for amplitudes in state.
        :return: State object in the basis state specified
        """
        state = np.zeros(1 << qubit_count, dtype)
        assert basis_state < 1 << qubit_count, 'Basis state is not valid.'
        state[basis_state] = 1.0
        return State(state, dtype=dtype)

    def __len__(self):
        return self.basis_size

    def __getitem__(self, item):
        return self.amplitudes.__getitem__(item)

    def __setitem__(self, key, value):
        return self.amplitudes.__setitem__(key, value)

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __repr__(self):
        return self.amplitudes.__repr__()

    def __add__(self, other):
        return State(self.amplitudes + other.amplitudes)

    def __radd__(self, other):
        # supporting sum
        if other == 0:
            return State(self.amplitudes)
        return State(self.amplitudes + other.amplitudes)

    def __sub__(self, other):
        return State(self.amplitudes - other.amplitudes)

    def __mul__(self, other):
        return State(self.amplitudes.__mul__(other))

    def __rmul__(self, other):
        return State(self.amplitudes.__rmul__(other))

    def __truediv__(self, number):
        return State(self.amplitudes / number)

    def __iter__(self):
        return self.amplitudes.__iter__()

    def __str__(self):
        parts = []

        def sign(num):
            return '+' if num > 0 else '-'

        for amp, label in zip(self.amplitudes, range(self.basis_size)):
            if amp == 0:
                continue

            imag, real = amp.imag, amp.real

            if imag == 0:
                rep = [sign(real), '%.3f' % abs(real)]  # [sign, number]
            elif real == 0:
                rep = [sign(imag), '%.3f' % abs(imag)]
            elif imag < 0:
                if real < 0:
                    rep = ['-', '(%.3f + j %.3f)' % (-real, -imag)]
                else:
                    rep = ['+', '(%.3f - j %.3f)' % (real, -imag)]
            else:
                rep = ['+', '(%.3f + j %.3f' % (real, imag)]

            rep.append('|%d>' % label)

            if len(parts) == 0:
                if rep[0] == '-':
                    parts.append('-' + rep[1])
                    parts.append(rep[2])
                else:
                    parts.extend(rep[1:])
            else:
                parts.extend(rep)

        return " ".join(parts)
