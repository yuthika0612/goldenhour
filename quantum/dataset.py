"""
Case generator for the experiment.

HONEST FRAMING, and this must appear in any write-up: these are SYNTHETIC
cases. No labelled corpus of Indian fraud evidence bundles exists publicly,
which is precisely the gap the project proposes to fill. Synthetic data can
answer a REPRESENTATIONAL question — is the quantum feature space
geometrically different from the classical one, and does it align better
with labels — because that question is about the encoding, not about
whether the data is real.

It cannot answer whether the tool works on real fraud. Nothing here should
be reported as detection performance.

Two classes:
  y = 1  fraudulent episode  (escalating demands, payee rotation, pressure
                              stages, sometimes a small trust-building
                              refund early on)
  y = 0  legitimate episode  (ordinary repeated payments to a merchant,
                              landlord, family member, EMI)

The generator deliberately produces OVERLAPPING classes. A separable toy
problem would make every kernel score ~1.0 and the comparison meaningless.
"""
import numpy as np
from .features import N_FEATURES, FEATURE_NAMES


def _fraud_case(rng) -> np.ndarray:
    n_debits = rng.integers(2, 7)
    first = rng.lognormal(9.5, 0.8)                      # ~Rs 10k-40k
    escalation = rng.lognormal(1.1, 0.7)                 # demands grow
    total = first * (1 + escalation) * n_debits * rng.uniform(0.4, 0.9)
    n_payees = min(n_debits, 1 + rng.binomial(n_debits - 1, 0.45))
    switch = n_payees / n_debits
    span = rng.gamma(2.0, 1.6)                           # hours, compressed
    mean_gap = span * 60 / max(n_debits - 1, 1) if n_debits > 1 else 0
    burst = abs(rng.normal(0.85, 0.45))
    n_stages = rng.integers(5, 11)
    frac_pressure = np.clip(rng.beta(3.2, 3.0), 0, 1)
    trust = rng.random() < 0.35                          # some scams refund
    frac_trust = np.clip(rng.beta(1.4, 6.0), 0, 1) if trust else rng.beta(1, 20)
    n_tactics = rng.integers(3, 8)
    n_high = rng.poisson(2.1)
    inflow = rng.uniform(0.01, 0.12) if trust else 0.0
    return np.array([
        n_debits, np.log1p(total), escalation, n_payees, switch, span,
        mean_gap, burst, n_stages, frac_pressure, frac_trust, n_tactics,
        n_high, inflow,
    ])


def _legit_case(rng) -> np.ndarray:
    n_debits = rng.integers(1, 6)
    amt = rng.lognormal(8.6, 1.0)
    total = amt * n_debits
    escalation = abs(rng.normal(1.0, 0.35))              # stable amounts
    n_payees = min(n_debits, 1 + rng.binomial(max(n_debits - 1, 0), 0.15))
    switch = n_payees / n_debits
    span = rng.gamma(2.5, 3.5)                           # spread out
    mean_gap = span * 60 / max(n_debits - 1, 1) if n_debits > 1 else 0
    burst = abs(rng.normal(0.45, 0.35))
    n_stages = rng.integers(1, 5)
    frac_pressure = np.clip(rng.beta(1.0, 12.0), 0, 1)
    frac_trust = np.clip(rng.beta(1.2, 9.0), 0, 1)
    n_tactics = rng.integers(0, 3)
    n_high = rng.poisson(0.25)
    inflow = rng.uniform(0, 0.35) if rng.random() < 0.3 else 0.0   # refunds
    return np.array([
        n_debits, np.log1p(total), escalation, n_payees, switch, span,
        mean_gap, burst, n_stages, frac_pressure, frac_trust, n_tactics,
        n_high, inflow,
    ])


def make_dataset(n: int = 300, seed: int = 0, label_noise: float = 0.06,
                 blur: float = 0.35):
    """
    blur        mixes each case toward the other class's mean, creating the
                overlap real evidence has (a legitimate but chaotic set of
                payments; a careful scam that looks routine).
    label_noise fraction of labels flipped: mislabelled complaints exist.
    """
    rng = np.random.default_rng(seed)
    Xf = np.array([_fraud_case(rng) for _ in range(n // 2)])
    Xl = np.array([_legit_case(rng) for _ in range(n - n // 2)])

    mu_f, mu_l = Xf.mean(0), Xl.mean(0)
    Xf = (1 - blur) * Xf + blur * (mu_l + rng.normal(0, 0.25, Xf.shape) * np.abs(mu_l))
    Xl = (1 - blur) * Xl + blur * (mu_f + rng.normal(0, 0.25, Xl.shape) * np.abs(mu_f))

    X = np.vstack([Xf, Xl])
    y = np.concatenate([np.ones(len(Xf)), np.zeros(len(Xl))]).astype(int)

    flip = rng.random(len(y)) < label_noise
    y[flip] = 1 - y[flip]

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def scale_for_encoding(X: np.ndarray, method: str = "minmax_pi"):
    """
    Quantum encodings need bounded angles. Map each feature to [0, pi].
    Scaling is part of the encoding and is applied identically to the
    classical baselines so the comparison stays fair.
    """
    lo, hi = X.min(0), X.max(0)
    rng_ = np.where(hi - lo == 0, 1.0, hi - lo)
    Z = (X - lo) / rng_
    return Z * np.pi if method == "minmax_pi" else Z
