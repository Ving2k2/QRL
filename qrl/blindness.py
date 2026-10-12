"""
Which inputs can each readout see at initialization without re-uploading?

At zero variational angles the no-reup circuit is RY(pi*x_i)|0> on every qubit
followed by L rings of CNOT(q, q+1 mod n), a Clifford map. A Z-string observable
then depends only on the inputs whose qubits appear in its Heisenberg image.
It propagates each two-body Z_iZ_j (and each single Z_i) back through
L CNOT rings by exact Pauli-string algebra and lists the qubits it ends up on.
No training, no sampling.

"""
import itertools

N = 4
NAMES = ["cart pos", "cart vel", "pole angle", "pole ang vel"]


def cnot_heisenberg(zmask, c, t):
    # Z-type strings only: conjugating by CNOT(c,t) maps Z_t -> Z_c Z_t and leaves Z_c fixed.
    if zmask >> t & 1:
        zmask ^= 1 << c
    return zmask


def image(zmask, layers):
    # Heisenberg picture: apply the gates in reverse order (last ring first, last gate first).
    for _ in range(layers):
        for q in reversed(range(N)):
            zmask = cnot_heisenberg(zmask, q, (q + 1) % N)
    return zmask


def qubits(mask):
    return [q for q in range(N) if mask >> q & 1]
