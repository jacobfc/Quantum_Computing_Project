import numpy as np


class State(object):
    def __init__(self, initial_state):
        self.amplitudes = np.array(initial_state, np.float64)
        self.basis_size = len(self.amplitudes)
        self.qubit_count = int(np.log2(self.basis_size))

        # assert initial_state has a power of two number of entries
        assert 2**self.qubit_count == self.basis_size

    def norm(self):
        self.amplitudes.dot(self.amplitudes)

    @classmethod
    def from_basis_state(cls, basis_size, basis_state):
        state = np.zeros(basis_size, np.float64)
        state[basis_state] = 1.0
        return State(state)

    def __getitem__(self, item):
        return self.amplitudes.__getitem__(item)

    def __setitem__(self, key, value):
        return self.amplitudes.__setitem__(key, value)

    def __repr__(self):
        return self.amplitudes.__repr__()

    def __add__(self, other):
        return State(self.amplitudes + other.amplitudes)

    def __radd__(self, other):
        # supporting sum
        if other == 0:
            return State(self.amplitudes)
        return State(self.amplitudes + other.amplitudes)

    def __mul__(self, other):
        return State(self.amplitudes.__mul__(other))

    def __rmul__(self, other):
        return State(self.amplitudes.__rmul__(other))

    def __iter__(self):
        return self.amplitudes.__iter__()

    def __str__(self):
        return " + ".join("%.3f |%d>" % (self.amplitudes[i], i)
                          for i in range(self.basis_size)
                          if self.amplitudes[i] != 0)
