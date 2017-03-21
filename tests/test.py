from quantum_circuit import MatrixGate
from quantum_circuit import FunctionalGate
from quantum_circuit import Gate
import quantum_circuit.gates_library as g_lib
from quantum_circuit.gates import State

state = State.from_basis_state(2, 0)
H = g_lib.hadamard
gate = MatrixGate.multi_gate(2, H, [0, 1])
print(gate.matrix)
print(gate(state))
