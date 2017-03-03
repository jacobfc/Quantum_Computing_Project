import abc


class Gate(metaclass=abc.ABCMeta):
    @property
    @abc.abstractproperty
    def qubit_count(self):
        return

    @property
    @abc.abstractproperty
    def basis_size(self):
        return -1

    @abc.abstractmethod
    def eval_bs(self, basis_state):
        """ Apply gate to basis state as input.

        :param basis_state: Integer representing a basis state in computational
            basis.
        :return: State
        """
        return

    def __call__(self, state):
        # simple implementation, may be overridden
        return sum(state[k] * self.eval_bs(k) for k in range(self.basis_size))
