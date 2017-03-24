import numpy as np


class State(object):
    def __init__(self, initial_state):
        # The state, represented as amplitudes of its constituent basis states.
        # Typecast because the amplitudes can be complex
        self.amplitudes = np.array(initial_state, np.complex64)
        self.basis_size = len(self.amplitudes)
        # For n qubits, there are 2**n basis states
        self.qubit_count = int(np.log2(self.basis_size))

        # assert initial_state has a power of two number of entries
        # note: 1 << n == 2 ** n
        assert 1 << self.qubit_count == self.basis_size

    def norm(self):
        """

        :return: Returns the square amplitude of the state
        """
        # .real is necessary because norm should be a float
        return np.conj(self).dot(self).real

    def prob_of_bs(self, bs):
        """

        :param bs: The basis state of interest
        :return: The probability of the system being in the basis state
            of interest
        """
        return np.square(abs(self[bs])) / self.norm()

    def prob_of_state(self, state):
        """

        :param state: The state of interest (does NOT have to be a basis state,
            can be any State object)
        :return: The probability of the system being in the state of interest
        """
        return np.conj(self).dot(state).real / (self.norm() * state.norm())

    def random_measure_bs(self,):
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
    def from_basis_state(cls, qubit_count: int, basis_state: int):
        state = np.zeros(1 << qubit_count, np.complex64)
        """
        Creates a State object that is a basis state
        :param qubit_count: Number of qubits in the state
        :param basis_state: The basis state to create
        :return: State object in the basis state specified
        """
        assert basis_state < 1 << qubit_count, 'Basis state is not valid.'
        state[basis_state] = 1.0
        return State(state)

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
        format_c = lambda c: "(%.3f + j %.3f)" % (c.real, c.imag)
        return " + ".join("%s |%d>" % (format_c(self.amplitudes[i]), i)
                          for i in range(self.basis_size)
                          if self.amplitudes[i] != 0)
