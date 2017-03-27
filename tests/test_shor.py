from quantum_circuit.gates import State
from tests.testcase import BaseTestCase
import quantum_circuit.gates_library as g_lib
import numpy as np
import math


# global variables
QFT_n = None


class TestShor(BaseTestCase):
    def test_shor(self):
        # run test and see if errors are thrown
        main()


# Quantum part of the Shor's algorithm
def shor_quantum(N, m, n, i):
    # Create input i-qubit register
    input_register = State.from_basis_state(i, 1)
    # Apply qft to put the states into uniform superposition
    input_register = State(np.dot(input_register, QFT_n.matrix))

    found = np.full(n, False)
    number = 0
    for j in range(n):
        value = m ** j % N
        if not found[value]:
            if number == 0:
                output_register = State.from_basis_state(i, value)
            else:
                output_register += State.from_basis_state(i, value)
            number += 1
            found[value] = True
    output_register /= math.sqrt(number)  # normalize the state

    # Measure the state of the output register
    result = output_register.random_measure_bs()
    # Partially collapse the input register states
    # If for xth basis state f(x) != result, its amplitude is set to zero
    number = 0  # number of remaining states (needed to renormalize the array)
    for j in range(n):
        value = m ** j % N
        if value == result:
            input_register.amplitudes[j] = 1
            number += 1
        else:
            input_register.amplitudes[j] = 0

    input_register /= math.sqrt(number)  # normalize the state
    input_register = State(np.dot(input_register, QFT_n.matrix))  # apply qft
    # Should return an integer number of the period of the function
    return input_register.random_measure_bs()


def contd_fraction(x):
    """
    Method to determine the continued fraction expansion of an arbitrary real
    number implemented by using the continued fractions algorithm.
    It describes a real number x in terms of integers [a_0, ..., a_M] so that:
    x = a_0 + 1 / (a_1 + 1 / (a_2 + 1 / (... + 1 / a_M )))
    """
    # Threshold value needed because of the imprecision of floats so that
    # dividing would stop at most likely after the last integer a_M
    threshold = 10. ** (-5)
    # Split the number into its integer and fractional parts
    if x < 1:
        a = [0]
    else:
        a = [int(x)]
        x -= a[0]
    # Find all a values
    while x > threshold:
        # Invert the fractional part
        x = 1 / x
        # Split the inverted number
        a.append(int(x))
        x -= a[-1]
    return a


def find_period(N, m, s):
    """
    Method to obtain the period from the result of the phase estimation algorithm,
    which is given by s = y/n, where y is the measured outcome of the input
    register and n is the size of the basis (input register).

    The nth convergent of a continued fraction is given by:
    x_n = [a_0, a_1, ..., a_n]
    Each convergent x_n can be written as x_n = p_n / q_n where p_n and q_n are
    coprime. The integers p_n and q_n are determined by the following relation:
    p_0 = a_0, p_1 = a_1 * a_0 + 1, p_n = a_n * p_(n-1) + p_(n-2)
    q_0 = 1,   q_1 = a_1,           q_n = a_n * q_(n-1) + q_(n-2)

    We know that s is a rational number (ratio of two integers), thus by computing
    its convergents we will obtain the nearest fractions to s, and so the possible
    candidates for the period will be given by q_0, q_1, ..., q_n.
    We know that for the period r: m^r mod N = 1, so by iterating through q[]
    the first value of q satisfying the equation will be the period.
    """
    # Find the continued fraction expansion of s
    a = contd_fraction(float(s))
    # Find the denominators of all convergents
    q = [1, a[1]]
    for i in range(2, len(a)):
        q.append(a[i] * q[i - 1] + q[i - 2])

    # Find the period r from the obtained candidates q[]
    r = 0  # In case the period is not found q[] we initially set it to zero
    for i in range(1, len(q)):
        if 1 == m ** (q[i]) % N:
            return q[i]
    return r


def main():
    """ The main program implementing the Shor's algorithm """
    # Factoring the arbitrary integer N
    # BE CAREFUL: setting N=50 already results in 2^12 x 2^12 matrices
    N = 21
    print("Factoring the integer", N)
    found_factors = False  # In case the step has to be repeated
    # If the given number is prime
    if N > 1 and all(N % i for i in range(2, N)):
        print("Found the factors (N is prime):", 1, N)
        found_factors = True
    # If the given number is odd
    elif N % 2 == 0:
        print("Found them (N is even):", 2, int(N / 2))
        found_factors = True

    j = 1  # Iteration number
    while not found_factors:
        print("----- ITERATION", j, " -----")
        j += 1

        # Pick a random integer m where 1 < m < N
        m = np.random.randint(2, N)
        print("The chosen m is", m)
        # If the gcd(m, N) != 1, then we have found a non-trivial factor of N
        # already. If, on the other hand, gcd(m, N) = 1, we proceed further.
        if math.gcd(N, m) > 1:
            # Using the Euclidean algorithm to find the greatest common
            # denominator for N and m
            p = math.gcd(N, m)
            q = int(N / p)
            print("Found them (lucky guess):", p, q)
            found_factors = True
        else:
            # Quantum part of the algorithm
            i = 0  # Number of qubits i must be such that 2^i >= N^2
            n = 0  # Size of the quantum register
            while n < N ** 2:  # Adjust i and n accordingly for the algorithm
                i += 1
                n = 2 ** i
            global QFT_n
            QFT_n = g_lib.qft(i)  # Create a n x n qft gate
            found_period = False  # In case the step has to be repeated
            while not found_period:
                y = shor_quantum(N, m, n, i)  # Find one of the peaking states
                print("Measured state: ", y)
                if y == 0:
                    print("Measured state is |0>: the step will be repeated")
                else:
                    r = find_period(N, m, y / n)  # Find the period
                    if r == 0:
                        # Most likely found a factor of the period
                        print("Period r = 0: the step will be repeated")
                    else:
                        found_period = True
                        print("Found the period:", r)

            if (r + 1) % 2 == 0:  # If the obtained period is odd
                print("Period problem: r is odd")
                print("Return to step one")
            else:
                if (m ** (int(r / 2)) + 1) % N == 0:
                    print("Found the trivial factors: return to step one")
                else:
                    # Determine the factors by using the Euclidean algorithm
                    p = math.gcd(m ** (int(r / 2)) + 1, N)
                    q = math.gcd(m ** (int(r / 2)) - 1, N)
                    print("Found the factors:", p, q)
                    found_factors = True


if __name__ == '__main__':
    main()
