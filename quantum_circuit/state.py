import numpy as np


class State(object): ## Do people normally put a docstring at the beginning of the Class declaratio
    def __init__(self, initial_state):
        # The state, represented as amplitudes of its constituent basis states.
        # Typecast because the amplitudes can be complex
        self.amplitudes = np.array(initial_state, np.complex64)
        self.basis_size = len(self.amplitudes)
        self.qubit_count = int(np.log2(self.basis_size)) # For n qubits, there are 2**n basis states

        # assert initial_state has a power of two number of entries
        assert 2 ** self.qubit_count == self.basis_size

    def norm(self):
        """

        :return: Returns the square amplitude of the state
        """
        # .real is necessary because norm should be a float
        return np.conj(self).dot(self).real

    def prob_of_bs(self, bs):
        """

        :param bs: The basis state of interest
        :return: The probability of the system being in the basis state of interest
        """
        return np.square(abs(self[bs])) / self.norm()

    def prob_of_state(self, state):
        """

        :param state: The state of interest (does NOT have to be a basis state, can be any State object)
        :return: The probability of the system being in the state of interest
        """
        return np.conj(self).dot(state).real / (self.norm() * state.norm())

    def random_measure_bs(self,):
        """ Select a basis state with probabilities according to amplitudes.

        Make a random choice on the basis states each weighted with
        the probability of finding the system in that state
        (i.e. amplitude squared).

        :return: The basis state which the state was measured to be in
        """

        sample = np.random.random_sample() #get a random number between 0 and 1
        bs = 0 #start "checking" if it's in the 0th basis state
        acc = self.prob_of_bs(bs)
        # if we haven't measured that state, keep checking the next states until we measure one of them
        while acc < sample:
            bs += 1
            acc += self.prob_of_bs(bs)
        return bs

    @classmethod
    def from_basis_state(cls, qubit_count, basis_state):
        """
        Creates a State object that is a basis state
        :param qubit_count: Number of qubits in the state
        :param basis_state: The basis state to create
        :return: State object in the basis state specified
        """
        state = np.zeros(1 << qubit_count, np.complex64) #note that 1 << qubit_count == 2**n
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
