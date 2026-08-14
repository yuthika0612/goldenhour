#!/usr/bin/env python3
"""
PROBE 9B — MONOLITHIC QED. No chunking.

Probe 9 used 32-byte blocks and the fixed alignment destroyed the signal
under obfuscation: insert one space and every subsequent block shifts. That
was an artefact of the implementation, not of the QED idea.

A monolithic encoding removes the problem entirely. The whole document
accumulates into ONE statevector of 2^n amplitudes, 8 to 10 qubits. There are
no block boundaries to misalign, because nothing is partitioned.

Four monolithic encodings, all alignment-free by construction:

  M1  byte-histogram amplitude, no phase        8 qubits, 256 amps
      amplitude[b] = sqrt(count of byte b).  Order-blind: the control that
      shows what phase is worth.

  M2  byte-histogram amplitude + POSITION PHASE  8 qubits
      amplitude[b] = sqrt(count of b)
      phase[b]     = mean position of b across the document
      Same magnitudes as M1, but order now lives in the phase. This is the
      HDQS "phase as a free dimension" claim in its purest testable form:
      identical amplitudes, information added at zero extra dimension.

  M3  byte-histogram amplitude + FOURIER position phase   8 qubits
      phase[b] = arg( sum_p exp(2*pi*i*p/L) ) over positions p of byte b.
      A circular-statistics summary of where each byte lives. More robust to
      insertions than the mean, because it is a phase of a global sum.

  M4  bigram amplitude + phase                  10 qubits, 1024 amps
      Byte pairs folded to 1024 bins. Captures local order in the magnitude
      as well as the phase.

Classical controls on the SAME information:
  C1  char n-gram TF-IDF                        the practical baseline
  C2  the same 256 real magnitudes M1 uses      isolates what phase adds
  C3  the same 256 magnitudes PLUS the phases as extra REAL features
      This is the decisive one. If a classical model handed the identical
      numbers (magnitude and phase as 512 real values) matches the fidelity
      kernel, the quantum structure is a packaging choice, not a mechanism.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from quantum.probe_qed_workflow import make_corpus


# ═══════════════════════════════════════════ monolithic encodings
def _bytes(doc):
    return np.frombuffer(doc.encode("utf-8", errors="ignore"), dtype=np.uint8)


def m1_hist(doc):
    """8 qubits. Amplitude only. Order-blind by construction."""
    b = _bytes(doc)
    cnt = np.bincount(b, minlength=256).astype(float)
    amp = np.sqrt(cnt)
    n = np.linalg.norm(amp)
    return (amp / n if n else amp).astype(complex)


def m2_hist_meanpos(doc):
    """8 qubits. Same amplitudes as M1; order encoded in phase."""
    b = _bytes(doc)
    L = max(len(b), 1)
    cnt = np.bincount(b, minlength=256).astype(float)
    pos_sum = np.bincount(b, weights=np.arange(len(b)), minlength=256)
    mean_pos = np.divide(pos_sum, cnt, out=np.zeros(256), where=cnt > 0)
    amp = np.sqrt(cnt)
    psi = amp * np.exp(2j * np.pi * mean_pos / L)
    n = np.linalg.norm(psi)
    return psi / n if n else psi


def m3_hist_fourierpos(doc):
    """8 qubits. Phase from a circular sum over positions: insertion-robust."""
    b = _bytes(doc)
    L = max(len(b), 1)
    cnt = np.bincount(b, minlength=256).astype(float)
    ph = np.exp(2j * np.pi * np.arange(len(b)) / L)
    acc = np.zeros(256, dtype=complex)
    np.add.at(acc, b, ph)
    ang = np.angle(acc)
    psi = np.sqrt(cnt) * np.exp(1j * ang)
    n = np.linalg.norm(psi)
    return psi / n if n else psi


def m4_bigram(doc):
    """10 qubits. Byte pairs folded to 1024 bins, with positional phase."""
    b = _bytes(doc)
    if len(b) < 2:
        return np.zeros(1024, dtype=complex)
    L = len(b) - 1
    idx = ((b[:-1].astype(int) * 31 + b[1:].astype(int)) % 1024)
    cnt = np.bincount(idx, minlength=1024).astype(float)
    ph = np.exp(2j * np.pi * np.arange(L) / L)
    acc = np.zeros(1024, dtype=complex)
    np.add.at(acc, idx, ph)
    psi = np.sqrt(cnt) * np.exp(1j * np.angle(acc))
    n = np.linalg.norm(psi)
    return psi / n if n else psi


ENCODERS = {"M1 hist (no phase)": m1_hist,
            "M2 hist + mean-pos phase": m2_hist_meanpos,
            "M3 hist + fourier phase": m3_hist_fourierpos,
            "M4 bigram + phase (10q)": m4_bigram}


def fidelity_kernel(P, Q=None):
    Q = P if Q is None else Q
    return np.abs(P.conj() @ Q.T) ** 2


# ═══════════════════════════════════════════════════ evaluation
def run_split(docs, y, n_train, seed):
    tr = np.arange(n_train)
    te = np.arange(n_train, len(y))
    res = {}

    # C1 TF-IDF
    da = np.array(docs, dtype=object)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=1, max_features=5000)
    Xtr = vec.fit_transform(da[tr]); Xte = vec.transform(da[te])
    clf = LogisticRegression(max_iter=3000).fit(Xtr, y[tr])
    res["C1 TF-IDF"] = roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])

    states = {k: np.stack([f(d) for d in docs]) for k, f in ENCODERS.items()}

    for name, P in states.items():
        K = fidelity_kernel(P)
        c = SVC(kernel="precomputed", C=1.0).fit(K[np.ix_(tr, tr)], y[tr])
        res[name] = roc_auc_score(y[te], c.decision_function(K[np.ix_(te, tr)]))

    # C2: the same magnitudes M3 uses, as classical real features
    mag = np.abs(states["M3 hist + fourier phase"])
    s = StandardScaler().fit(mag[tr])
    c = LogisticRegression(max_iter=3000).fit(s.transform(mag[tr]), y[tr])
    res["C2 magnitudes only"] = roc_auc_score(
        y[te], c.predict_proba(s.transform(mag[te]))[:, 1])

    # C3 THE DECISIVE CONTROL: identical numbers, magnitude AND phase, but
    # handed to a classical model as 512 plain real features.
    P3 = states["M3 hist + fourier phase"]
    both = np.hstack([np.abs(P3), np.angle(P3)])
    s = StandardScaler().fit(both[tr])
    c = LogisticRegression(max_iter=3000).fit(s.transform(both[tr]), y[tr])
    res["C3 mag + phase (classical)"] = roc_auc_score(
        y[te], c.predict_proba(s.transform(both[te]))[:, 1])

    return res


def main():
    print("=" * 78)
    print("PROBE 9B  MONOLITHIC QED — no chunking, 8 to 10 qubits")
    print("=" * 78)
    print("""  The whole document accumulates into ONE statevector. No block
  boundaries, so no alignment to break. Hard regime throughout:
  filter-evading obfuscation plus short excerpts.
""")

    for excerpt in (120, 256):
        print("=" * 78)
        print(f"EXCERPT {excerpt} BYTES")
        print("=" * 78)
        keys = ["C1 TF-IDF", "M1 hist (no phase)", "M2 hist + mean-pos phase",
                "M3 hist + fourier phase", "M4 bigram + phase (10q)",
                "C2 magnitudes only", "C3 mag + phase (classical)"]
        print(f"  {'train':>6}" + "".join(f"{k.split()[0]:>9}" for k in keys))
        for n_train in (20, 60, 120):
            acc = {k: [] for k in keys}
            for t in range(10):
                docs, y = make_corpus(n_train + 200, seed=700 + t,
                                      hard=True, excerpt=excerpt)
                if len(np.unique(y[:n_train])) < 2:
                    continue
                r = run_split(docs, y, n_train, seed=t)
                for k in keys:
                    acc[k].append(r[k])
            print(f"  {n_train:>6}" +
                  "".join(f"{np.mean(acc[k]):>9.3f}" for k in keys))
        print()
        print("  legend: C1 TF-IDF | M1-M4 quantum fidelity kernels | "
              "C2, C3 classical controls")
        print()

    print("=" * 78)
    print("WHAT TO LOOK FOR")
    print("=" * 78)
    print("""
  M2/M3 vs M1   what the phase layer is worth. M1 has identical amplitudes
                and no phase, so any gap is the phase doing work.

  M3 vs C3      THE DECISIVE COMPARISON. C3 receives the exact same numbers
                — magnitudes and phases — as 512 plain real features for a
                classical model. If C3 matches M3, the fidelity kernel is a
                packaging of information a classical model can use directly,
                and no quantum mechanism is required.

  anything vs C1  whether the whole approach is competitive with a char
                n-gram baseline that takes one line to write.
""")


if __name__ == "__main__":
    main()
