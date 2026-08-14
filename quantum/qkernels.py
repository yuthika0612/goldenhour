"""
Quantum feature maps, simulated exactly with numpy.

Each encoding maps a classical vector x to a pure state |psi(x)>. The kernel
is the state fidelity K_ij = |<psi(x_i)|psi(x_j)>|^2, which is what a
quantum kernel method would estimate on hardware.

Three encodings, chosen because they differ in a way that matters:

  angle       product state, one rotation per qubit. No entanglement.
              The kernel factorises into a product of cosines, so it is
              classically computable in O(n). Included as the control.

  zz          the standard entangling map (Havlicek et al. 2019): Hadamards,
              then diagonal phases on single qubits and on qubit pairs,
              repeated. Believed hard to simulate classically in general.

  reupload    data fed in twice with a rotation layer between, giving a
              deeper and more nonlinear map (Perez-Salinas et al. 2020).

Everything here is exact simulation, so results reflect the representation
itself and not sampling noise. Shot noise is modelled separately.
"""
import numpy as np


# --------------------------------------------------------------- utilities
def _walsh_hadamard(v: np.ndarray) -> np.ndarray:
    """H^{tensor n} applied to a batch of statevectors, in place-ish."""
    a = v.copy()
    n_states = a.shape[-1]
    h = 1
    while h < n_states:
        for i in range(0, n_states, h * 2):
            x = a[..., i:i + h].copy()
            y = a[..., i + h:i + 2 * h].copy()
            a[..., i:i + h] = x + y
            a[..., i + h:i + 2 * h] = x - y
        h *= 2
    return a / np.sqrt(n_states)


def _pm_bits(n_qubits: int) -> np.ndarray:
    """Matrix Z[b, q] = +1/-1, the eigenvalue of Z_q on basis state b."""
    idx = np.arange(2 ** n_qubits)
    bits = ((idx[:, None] >> np.arange(n_qubits)[None, :]) & 1)
    return 1.0 - 2.0 * bits          # 0 -> +1, 1 -> -1


# -------------------------------------------------------------- encodings
def _take_features(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    One feature per qubit. If more qubits than features are requested, cycle
    through the features again rather than failing: re-encoding a feature on
    another qubit is a legitimate (if redundant) map, and it lets the
    concentration diagnostic run past the feature count.
    """
    d = X.shape[1]
    cols = [q % d for q in range(n_qubits)]
    return X[:, cols]


def angle_states(X: np.ndarray, n_qubits: int, scale: float = 1.0) -> np.ndarray:
    """
    Product encoding: qubit q carries RY(scale * x_q) applied to |0>.
    |psi> = tensor_q [cos(theta_q/2), sin(theta_q/2)]
    """
    Xq = _take_features(X, n_qubits) * scale
    c = np.cos(Xq / 2.0)
    s = np.sin(Xq / 2.0)
    N = X.shape[0]
    psi = np.ones((N, 1), dtype=complex)
    for q in range(n_qubits):
        blk = np.stack([c[:, q], s[:, q]], axis=1).astype(complex)
        psi = (psi[:, :, None] * blk[:, None, :]).reshape(N, -1)
    return psi


def zz_states(X: np.ndarray, n_qubits: int, reps: int = 2,
              scale: float = 1.0) -> np.ndarray:
    """
    ZZ feature map. Per repetition: H^n, then diag phases
        phi_q(x) = scale * x_q
        phi_qp(x) = scale * (pi - x_q)(pi - x_p)   for each pair q<p
    """
    Xq = _take_features(X, n_qubits) * scale
    Z = _pm_bits(n_qubits)                          # (2^n, n)
    N = X.shape[0]

    single = Xq @ Z.T                               # (N, 2^n)
    pair = np.zeros((N, 2 ** n_qubits))
    for q in range(n_qubits):
        for p in range(q + 1, n_qubits):
            coef = (np.pi - Xq[:, q]) * (np.pi - Xq[:, p])
            pair += coef[:, None] * (Z[:, q] * Z[:, p])[None, :]
    phase = np.exp(1j * (single + pair))

    psi = np.zeros((N, 2 ** n_qubits), dtype=complex)
    psi[:, 0] = 1.0
    for _ in range(reps):
        psi = _walsh_hadamard(psi)
        psi = psi * phase
    return psi


def reupload_states(X: np.ndarray, n_qubits: int, layers: int = 3,
                    scale: float = 1.0, seed: int = 7) -> np.ndarray:
    """
    Data re-uploading: alternate data-dependent rotations with a fixed
    entangling layer. Uses more of the input than one feature per qubit.
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    dim = 2 ** n_qubits
    Z = _pm_bits(n_qubits)

    psi = np.zeros((N, dim), dtype=complex)
    psi[:, 0] = 1.0
    psi = _walsh_hadamard(psi)

    for L in range(layers):
        # each layer sees a different slice of the feature vector
        cols = [(L * n_qubits + q) % d for q in range(n_qubits)]
        w = rng.normal(1.0, 0.15, size=n_qubits)
        ang = X[:, cols] * scale * w
        psi = psi * np.exp(1j * (ang @ Z.T))
        psi = _walsh_hadamard(psi)                 # mixing between layers
        ring = np.zeros(dim)
        for q in range(n_qubits):
            p = (q + 1) % n_qubits
            ring += Z[:, q] * Z[:, p]
        psi = psi * np.exp(1j * 0.5 * ring)[None, :]
    return psi


ENCODINGS = {
    "angle": angle_states,
    "zz": zz_states,
    "reupload": reupload_states,
}


# ----------------------------------------------------------------- kernels
def fidelity_kernel(psi_a: np.ndarray, psi_b: np.ndarray = None) -> np.ndarray:
    """K_ij = |<psi_i|psi_j>|^2."""
    if psi_b is None:
        psi_b = psi_a
    G = psi_a.conj() @ psi_b.T
    return np.abs(G) ** 2


def add_shot_noise(K: np.ndarray, shots: int, seed: int = 0) -> np.ndarray:
    """
    Hardware estimates each kernel entry from a finite number of shots.
    Model it as a binomial estimate, then re-symmetrise and clip.
    """
    if not shots:
        return K
    rng = np.random.default_rng(seed)
    noisy = rng.binomial(shots, np.clip(K, 0, 1)) / shots
    noisy = (noisy + noisy.T) / 2.0
    np.fill_diagonal(noisy, 1.0)
    return noisy


def classical_kernels(X: np.ndarray, Y: np.ndarray = None) -> dict:
    """Baselines: linear, and RBF at several bandwidths."""
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
    Y = X if Y is None else Y
    out = {"linear": linear_kernel(X, Y)}
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    for mult, name in [(0.25, "rbf_narrow"), (1.0, "rbf_median"), (4.0, "rbf_wide")]:
        gamma = 1.0 / (mult * med)
        out[name] = rbf_kernel(X, Y, gamma=gamma)
    return out
