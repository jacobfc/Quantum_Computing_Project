import numpy as np


class State(object):
    def __init__(self, initial_state):
        self.amplitudes = np.array(initial_state, np.complex64)
        self.basis_size = len(self.amplitudes)
        self.qubit_count = int(np.log2(self.basis_size))

        # assert initial_state has a power of two number of entries
        assert 2 ** self.qubit_count == self.basis_size

    def norm(self):
        # .real is necessary because norm should be a float
        return np.conj(self).dot(self).real

    def prob_of_bs(self, bs):
        return np.square(abs(self[bs])) / self.norm()

    def prob_of_state(self, state):
        return np.conj(self).dot(state).real / (self.norm() * state.norm())

    @classmethod
    def from_basis_state(cls, qubit_count, basis_state):
        state = np.zeros(1 << qubit_count, np.complex64)
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

