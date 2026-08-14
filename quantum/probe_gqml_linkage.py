#!/usr/bin/env python3
"""
PROBE 7 — Geometric QML for case linkage (similarity testing).

WHY THIS IS A DIFFERENT QUESTION FROM PROBE 1

Probes 1-6 tested single-case classification on aggregate features and found
nothing. That task has no symmetry to exploit and its signal is local.

Case linkage is structurally different, and matches the one shape where GQML
has shown large empirical gains (Umeano, Scali & Kyriienko, PRA 113, 052425,
2026 — barcode similarity):

    pairwise task            two inputs, one label
    Z2 exchange symmetry     link(A,B) = link(B,A)
    relabelling symmetry     the pattern matters, not the surface values
    few-shot regime          almost no labelled "same operation" pairs exist
    GLOBAL correlations      two complaints belong to one operation because
                             of overall script structure, not word overlap

Their result: the symmetry-aware MEASUREMENT model (LASSO over equivariant
observables) generalised from 3 samples per class on 20-qubit inputs, while
Siamese DNNs and CNNs sat near chance. The variational unitary model failed.
Their own caveat is equally important: for classical data in memory the
advantage dequantises, because 2-fold forrelation is classically simulable.

WHAT THIS PROBE DOES

Three data regimes, run through the same models:

  R1 LOCAL      linked cases share surface features (same handle pattern,
                similar amounts). What ordinary fraud data looks like.
  R2 ADVERSARIAL  the operation deliberately randomises surface features
                while preserving the script skeleton. Signal moves to global
                structure. This is a real adversarial behaviour, not a
                contrivance.
  R3 FORRELATED  the Raz-Tal construction from the paper. Included as a
                POSITIVE CONTROL: if the implementation cannot reproduce a
                known advantage here, any negative elsewhere is meaningless.

Models:
  GQML-M   equivariant observables + LASSO   (the architecture that won)
  GQML-U   variational equivariant circuit   (the one that failed; sanity)
  Siamese  classical MLP on paired features  (classical baseline)
  Feature  classical hand-crafted pair features + logistic regression
"""
import sys
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RNG_GLOBAL = np.random.default_rng(7)


# ═══════════════════════════════════════════════════ quantum machinery
def phase_states(bits: np.ndarray) -> np.ndarray:
    """
    Real equally weighted (phase) state: amplitude (-1)^{x_j} / sqrt(N).
    bits: (M, N) binary. Returns (M, N) real amplitudes, N = 2^n.
    """
    amps = (1.0 - 2.0 * bits) / np.sqrt(bits.shape[1])
    return amps


def wht(v: np.ndarray) -> np.ndarray:
    """Normalised Walsh-Hadamard transform along the last axis."""
    a = v.astype(float).copy()
    n = a.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[..., i:i + h].copy()
            y = a[..., i + h:i + 2 * h].copy()
            a[..., i:i + h] = x + y
            a[..., i + h:i + 2 * h] = x - y
        h *= 2
    return a / np.sqrt(n)


def equivariant_observables(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Expectation values of observables that respect BOTH symmetries:
      Z2 exchange   : swapping the pair leaves the label unchanged
      global sign   : flipping all bits leaves the label unchanged

    Each column is <O_k> for the pair state |phi_1> tensor |phi_2>. These are
    computed on the product state directly, which is what makes the
    architecture cheap: no 2n-qubit statevector is ever materialised.

    Observable 1 is the forrelation observable H^{2n} . prod SWAP, the one
    the paper identifies as the optimal decision feature.
    """
    h1, h2 = wht(p1), wht(p2)
    feats = [
        np.einsum("ij,ij->i", h1, p2),            # <phi1|H|phi2>  (forrelation)
        np.einsum("ij,ij->i", p1, h2),            # symmetric partner
        np.einsum("ij,ij->i", p1, p2),            # plain overlap (SWAP)
        np.einsum("ij,ij->i", h1, h2),            # overlap in Hadamard basis
        np.einsum("ij,ij->i", p1, p1) * 0 + 1.0,  # identity / bias
    ]
    f = np.stack(feats, axis=1)
    # symmetrise under exchange, then take invariant combinations
    sym = np.stack([
        0.5 * (f[:, 0] + f[:, 1]),      # exchange-symmetric forrelation
        f[:, 2], f[:, 3],
        f[:, 0] ** 2, f[:, 1] ** 2,     # squares: sign-flip invariant
        (0.5 * (f[:, 0] + f[:, 1])) ** 2,
        f[:, 2] ** 2, f[:, 3] ** 2,
        np.abs(f[:, 0]), np.abs(f[:, 2]),
    ], axis=1)
    return sym


def gqml_measurement_model(tr_p1, tr_p2, tr_y, te_p1, te_p2):
    """GQML-M: equivariant observables + LASSO (the winning architecture)."""
    Xtr = equivariant_observables(tr_p1, tr_p2)
    Xte = equivariant_observables(te_p1, te_p2)
    sc = StandardScaler().fit(Xtr)
    model = LassoCV(cv=min(3, max(2, len(tr_y) // 3)), max_iter=20000)
    model.fit(sc.transform(Xtr), tr_y.astype(float))
    return model.predict(sc.transform(Xte))


def gqml_unitary_model(tr_p1, tr_p2, tr_y, te_p1, te_p2, seed=0):
    """
    GQML-U: a variational equivariant model. Included because the paper
    found it trains but does not generalise; reproducing that is a check
    that the comparison is faithful.
    """
    rng = np.random.default_rng(seed)
    Xtr = equivariant_observables(tr_p1, tr_p2)[:, :3]
    Xte = equivariant_observables(te_p1, te_p2)[:, :3]
    theta = rng.normal(0, 0.5, size=Xtr.shape[1] + 1)
    lr = 0.15
    for _ in range(400):
        pred = np.tanh(Xtr @ theta[:-1] + theta[-1])
        err = pred - (2 * tr_y - 1)
        grad = np.concatenate([Xtr.T @ (err * (1 - pred ** 2)),
                               [np.sum(err * (1 - pred ** 2))]]) / len(tr_y)
        theta -= lr * grad
    return np.tanh(Xte @ theta[:-1] + theta[-1])


def siamese_model(tr_p1, tr_p2, tr_y, te_p1, te_p2, seed=0):
    """Classical Siamese-style baseline: shared embedding, distance metric."""
    n_feat = min(64, tr_p1.shape[1])
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(tr_p1.shape[1], n_feat)) / np.sqrt(tr_p1.shape[1])

    def emb(p):
        return np.tanh(p @ proj)

    def pairfeat(a, b):
        ea, eb = emb(a), emb(b)
        return np.hstack([np.abs(ea - eb), ea * eb])

    Xtr, Xte = pairfeat(tr_p1, tr_p2), pairfeat(te_p1, te_p2)
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500,
                        random_state=seed, early_stopping=False)
    clf.fit(sc.transform(Xtr), tr_y)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def dequantized_model(tr_p1, tr_p2, tr_y, te_p1, te_p2):
    """
    THE DECISIVE CONTROL.

    Compute the forrelation feature |<phi_1|H|phi_2>| directly with a
    classical fast Walsh-Hadamard transform — O(N log N) on a laptop — and
    feed it to logistic regression. No quantum computer, no Hilbert space.

    The barcode paper says this plainly: for classical data held in memory,
    the advantage dequantises, because the Hadamard transform is easy
    classically and 2-fold forrelation is classically simulable. This model
    tests that claim directly. If it matches GQML-M, the advantage is
    quantum-INSPIRED, not quantum.
    """
    def feats(a, b):
        ha, hb = wht(a), wht(b)
        f1 = np.einsum("ij,ij->i", ha, b)
        f2 = np.einsum("ij,ij->i", a, hb)
        ov = np.einsum("ij,ij->i", a, b)
        return np.stack([np.abs(f1), np.abs(f2), f1 ** 2, f2 ** 2,
                         np.abs(ov), ov ** 2], axis=1)

    Xtr, Xte = feats(tr_p1, tr_p2), feats(te_p1, te_p2)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(sc.transform(Xtr), tr_y)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def classical_features_model(tr_p1, tr_p2, tr_y, te_p1, te_p2):
    """
    Classical baseline WITH THE SAME SYMMETRIES.

    This is the fairness control the barcode paper itself flags: symmetries
    help classical ML too, so a symmetric quantum model must be compared
    against a symmetric classical model, not a naive one. Features here are
    exchange-symmetric and global-sign-invariant, exactly like the quantum
    observable pool.
    """
    def feats(a, b):
        ov = np.einsum("ij,ij->i", a, b)
        cos = ov / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        agree = (np.sign(a) == np.sign(b)).mean(1)
        return np.stack([
            np.abs(ov), ov ** 2,                 # sign-flip invariant
            np.abs(cos), cos ** 2,
            np.abs(agree - 0.5),                 # invariant agreement
            np.abs(a - b).sum(1),
            np.abs(np.abs(a).sum(1) - np.abs(b).sum(1)),
        ], axis=1)

    Xtr, Xte = feats(tr_p1, tr_p2), feats(te_p1, te_p2)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(Xtr), tr_y)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


# ═══════════════════════════════════════════════════════ data regimes
def regime_local(M, N, rng):
    """
    R1 — linked complaints share SURFACE features. Ordinary fraud data:
    the same operation reuses handles, amounts and phrasing.
    """
    y = rng.integers(0, 2, M)
    b1 = rng.integers(0, 2, (M, N))
    b2 = b1.copy()
    for m in range(M):
        if y[m] == 1:                       # linked: small perturbation
            flip = rng.random(N) < 0.15
            b2[m] = np.where(flip, 1 - b1[m], b1[m])
        else:                               # unlinked: independent
            b2[m] = rng.integers(0, 2, N)
    return b1, b2, y


def regime_adversarial(M, N, rng, n_motifs=4):
    """
    R2 — the operation randomises surface features but keeps the SCRIPT
    SKELETON. Linked pairs share a global structural motif (which basis
    pattern they are built from) while their raw bits look unrelated.

    This is realistic adversarial behaviour: change the amounts, the names
    and the handles, keep the playbook.
    """
    y = rng.integers(0, 2, M)
    basis = rng.integers(0, 2, (n_motifs, N))
    b1 = np.empty((M, N), int)
    b2 = np.empty((M, N), int)
    for m in range(M):
        k1 = rng.integers(n_motifs)
        k2 = k1 if y[m] == 1 else (k1 + 1 + rng.integers(n_motifs - 1)) % n_motifs
        # heavy surface randomisation: 22% of bits flipped independently
        b1[m] = np.where(rng.random(N) < 0.22, 1 - basis[k1], basis[k1])
        b2[m] = np.where(rng.random(N) < 0.22, 1 - basis[k2], basis[k2])
        # and an independent global sign flip, which the label must ignore
        if rng.random() < 0.5:
            b1[m] = 1 - b1[m]
        if rng.random() < 0.5:
            b2[m] = 1 - b2[m]
    return b1, b2, y


def regime_forrelated(M, N, rng):
    """
    R3 — the Raz-Tal / Aaronson construction used in the barcode paper.
    Linked pairs are related by a Hadamard transform. POSITIVE CONTROL:
    a correct GQML implementation must win here.
    """
    y = rng.integers(0, 2, M)
    b1 = np.empty((M, N), int)
    b2 = np.empty((M, N), int)
    for m in range(M):
        v1 = rng.normal(size=N)
        if y[m] == 1:
            v2 = wht(v1[None, :])[0] + rng.normal(0, 0.35, N)
        else:
            v2 = rng.normal(size=N)
        b1[m] = (v1 < 0).astype(int)
        b2[m] = (v2 < 0).astype(int)
    return b1, b2, y


REGIMES = {
    "R1 local (ordinary fraud data)": regime_local,
    "R2 adversarial (surface randomised)": regime_adversarial,
    "R3 forrelated (positive control)": regime_forrelated,
}


# ═══════════════════════════════════════════════════════════ evaluation
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return 0.5


def run_regime(gen, n_train, N=64, n_test=400, n_trials=12, seed0=0):
    out = {k: [] for k in ("GQML-M", "GQML-U", "Siamese", "Classical", "Dequantized")}
    for t in range(n_trials):
        rng = np.random.default_rng(seed0 + t)
        b1, b2, y = gen(n_train + n_test, N, rng)
        p1, p2 = phase_states(b1), phase_states(b2)
        tr = slice(0, n_train)
        te = slice(n_train, n_train + n_test)
        if len(np.unique(y[tr])) < 2:
            continue
        out["GQML-M"].append(auc(y[te], gqml_measurement_model(
            p1[tr], p2[tr], y[tr], p1[te], p2[te])))
        out["GQML-U"].append(auc(y[te], gqml_unitary_model(
            p1[tr], p2[tr], y[tr], p1[te], p2[te], seed=t)))
        out["Siamese"].append(auc(y[te], siamese_model(
            p1[tr], p2[tr], y[tr], p1[te], p2[te], seed=t)))
        out["Classical"].append(auc(y[te], classical_features_model(
            p1[tr], p2[tr], y[tr], p1[te], p2[te])))
        out["Dequantized"].append(auc(y[te], dequantized_model(
            p1[tr], p2[tr], y[tr], p1[te], p2[te])))
    return {k: (np.mean(v), np.std(v)) for k, v in out.items() if v}


def main():
    print("=" * 78)
    print("PROBE 7  GEOMETRIC QML FOR CASE LINKAGE (similarity testing)")
    print("=" * 78)
    print("""  A different task from probes 1-6: pairwise, symmetric, few-shot, and
  potentially driven by global rather than local correlations. This is the
  one shape where GQML has shown large empirical gains.

  R3 is a POSITIVE CONTROL. If the implementation cannot reproduce the known
  advantage there, every negative result below is uninterpretable.
""")

    for label, gen in REGIMES.items():
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"  {'train':>6} {'GQML-M':>15} {'GQML-U':>15} "
              f"{'Siamese':>15} {'Classical+sym':>15} {'Dequantized':>15}")
        for n_train in (6, 10, 20, 50, 150):
            r = run_regime(gen, n_train)
            row = f"  {n_train:>6}"
            for k in ("GQML-M", "GQML-U", "Siamese", "Classical",
                      "Dequantized"):
                m, s = r.get(k, (np.nan, np.nan))
                row += f"  {m:>6.3f}+-{s:<5.3f}"
            print(row)
        print()

    print("=" * 78)
    print("READING THE TABLE")
    print("=" * 78)
    print("""
  R3 (control) tells you whether the machinery works at all.
  R1 tells you what happens on data that looks like real fraud complaints.
  R2 tells you what happens when the operation actively hides surface
     similarity, which is the case worth caring about, because a linkage
     tool that only catches lazy scammers catches nothing that matters.
""")


if __name__ == "__main__":
    main()
