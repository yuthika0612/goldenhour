#!/usr/bin/env python3
"""
PROBE 8 — ENCODING, not learning.

Probes 1-7 all asked the same question in different costumes: given a feature
vector, does a quantum model learn better? Answer: no. But every one of them
used AMPLITUDE-ONLY encoding of AGGREGATE features, and aggregate features are
histogram-like — they destroy order by construction.

Two representational ideas from prior HDQS work were never tested here:

  A. PHASE AS A FREE DIMENSION
     Amplitude carries one layer, phase carries another, at zero storage cost.
     Two records with identical value histograms are identical classically;
     different phases make them distinguishable. Fraud scripts are SEQUENCES,
     so this is directly relevant: two operations can move the same amounts
     to the same handles in a different ORDER and look identical to any
     aggregate feature vector.

  B. SUPERPOSITION OF AMBIGUOUS READINGS
     Golden Hour's actual bottleneck is uncertain evidence: OCR gives
     'T25O114O9OO123456' and the true reading could be several strings; a
     timestamp is ambiguous between two parses. Classically you either commit
     early (lossy) or enumerate branches (exponential: k readings for each of
     n items is k^n combinations). Amplitude encoding holds all k^n branches
     in n*log2(k) qubits and computes expectations over ALL of them without
     enumerating.

These are REPRESENTATION questions, not learning questions, so they need
different experiments and different controls. In particular the honest
control for A is a classical model given ORDER-AWARE features, and for B it
is Monte Carlo sampling over branches, which is what a competent engineer
would actually write.
"""
import sys, time, itertools, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ══════════════════════════════════════════ PART A: PHASE ENCODING
STAGES = ["S01", "S03", "S04", "S05", "S06", "S08", "S09", "S10"]


def make_scripts(M, L=8, seed=0, n_ops=2):
    """
    Two fraud operations that are IDENTICAL in aggregate and differ only in
    ORDER. Both use the same stage multiset and the same amount multiset.
    Operation 0 threatens before demanding; operation 1 demands before
    threatening. Any histogram feature sees one distribution.
    """
    rng = np.random.default_rng(seed)
    base_stages = np.array([0, 2, 4, 1, 3, 5, 6, 7])       # index into STAGES
    amounts = np.array([5000., 15000., 50000., 25000.,
                        75000., 10000., 30000., 20000.])

    seqs, amts, y = [], [], []
    for m in range(M):
        op = rng.integers(n_ops)
        s = base_stages.copy()
        a = amounts.copy()
        if op == 1:
            # same elements, different order: swap two blocks
            s = np.concatenate([s[4:], s[:4]])
            a = np.concatenate([a[4:], a[:4]])
        # realistic jitter that does NOT change the multiset
        perm = rng.permutation(L)[:2]
        s[perm] = s[perm[::-1]]
        a = a * rng.uniform(0.97, 1.03, L)
        seqs.append(s); amts.append(a); y.append(op)
    return np.array(seqs), np.array(amts), np.array(y)


def encode_amplitude_only(seqs, amts):
    """Aggregate features: what probes 1-7 used. Order-blind by construction."""
    feats = []
    for s, a in zip(seqs, amts):
        hist = np.bincount(s, minlength=len(STAGES)).astype(float)
        feats.append(np.concatenate([
            hist, [a.sum(), a.mean(), a.std(), a.max(), a.min(), len(a)]]))
    return np.array(feats)


def encode_amplitude_phase(seqs, amts):
    """
    HDQS-style: amplitude carries the magnitude layer, phase carries the
    order layer, in ONE complex vector of the same length.

        amplitude_j = normalised amount at position j
        phase_j     = 2*pi * (stage index) / n_stages  +  position term

    Features are then extracted the way a quantum device would: expectation
    values of observables on the state, including phase-sensitive ones.
    """
    out = []
    for s, a in zip(seqs, amts):
        L = len(a)
        amp = a / (np.linalg.norm(a) + 1e-12)
        phase = 2 * np.pi * s / len(STAGES) + np.pi * np.arange(L) / L
        psi = amp * np.exp(1j * phase)

        # observables: real/imag structure, autocorrelation at several lags,
        # and the Fourier magnitude spectrum (a global, order-sensitive read)
        spec = np.abs(np.fft.fft(psi))
        lags = [np.abs(np.vdot(psi, np.roll(psi, k))) for k in (1, 2, 3, 4)]
        out.append(np.concatenate([
            [psi.real.sum(), psi.imag.sum(),
             np.abs(psi.sum()), np.angle(psi.sum())],
            lags, spec[:6],
        ]))
    return np.array(out)


def encode_classical_order_aware(seqs, amts):
    """
    THE HONEST CONTROL. A classical model given order-aware features:
    bigram counts and lagged differences. This is what a competent engineer
    writes when told 'order matters'. If it matches the phase encoding, the
    phase representation is a nice packaging, not an advantage.
    """
    n = len(STAGES)
    out = []
    for s, a in zip(seqs, amts):
        big = np.zeros(n * n)
        for i in range(len(s) - 1):
            big[s[i] * n + s[i + 1]] += 1
        diffs = np.diff(a)
        out.append(np.concatenate([
            big, [np.sign(diffs).sum(), diffs.mean(), diffs.std()],
            a[:3] / (a.sum() + 1e-9),          # positional shares
        ]))
    return np.array(out)


def cv_auc(X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc = []
    for tr, te in skf.split(X, y):
        s = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=4000)
        clf.fit(s.transform(X[tr]), y[tr])
        sc.append(roc_auc_score(y[te], clf.predict_proba(s.transform(X[te]))[:, 1]))
    return float(np.mean(sc)), float(np.std(sc))


def part_a():
    print("=" * 78)
    print("PART A  PHASE AS A FREE DIMENSION")
    print("=" * 78)
    print("""  Task: two fraud operations with the SAME stage multiset and the SAME
  amount multiset, differing only in ORDER. This is the case where
  aggregate features are provably blind.
""")
    print(f"  {'encoding':<38} {'AUC':>16} {'dims':>7}")
    for seed in (0,):
        seqs, amts, y = make_scripts(400, seed=seed)
        for name, fn in (
            ("amplitude only (aggregate) [probes 1-7]", encode_amplitude_only),
            ("amplitude + PHASE (HDQS style)", encode_amplitude_phase),
            ("classical order-aware [CONTROL]", encode_classical_order_aware),
        ):
            X = fn(seqs, amts)
            m, s = cv_auc(X, y, seed)
            print(f"  {name:<38} {m:>8.4f}+-{s:<6.4f} {X.shape[1]:>7}")

    print("""
  Amplitude-only sits at chance: probes 1-7 were throwing order away.
  Phase recovers it in the SAME number of dimensions. The classical control
  recovers it too, but needs 5x more dimensions to do so.

  That is a COMPRESSION difference, not an accuracy difference, and
  compression matters most when training data is scarce — which is exactly
  the forensic situation. Tested below.
""")

    print("=" * 78)
    print("PART A2  FEW-SHOT: does the compact encoding win when data is scarce?")
    print("=" * 78)
    print("""  14 phase dimensions vs 70 classical dimensions. With few samples the
  wider representation should overfit. This is the one regime where a
  compact order-aware encoding could genuinely pay.
""")
    print(f"  {'train':>7} {'amplitude only':>17} {'amplitude+PHASE':>18} "
          f"{'classical order':>18}")
    for n_train in (12, 20, 40, 80, 200):
        rows = {k: [] for k in ("amp", "phase", "cls")}
        for trial in range(15):
            seqs, amts, y = make_scripts(n_train + 300, seed=100 + trial)
            enc = {"amp": encode_amplitude_only(seqs, amts),
                   "phase": encode_amplitude_phase(seqs, amts),
                   "cls": encode_classical_order_aware(seqs, amts)}
            tr = slice(0, n_train); te = slice(n_train, n_train + 300)
            if len(np.unique(y[tr])) < 2:
                continue
            for k, X in enc.items():
                sc_ = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=4000)
                clf.fit(sc_.transform(X[tr]), y[tr])
                rows[k].append(roc_auc_score(
                    y[te], clf.predict_proba(sc_.transform(X[te]))[:, 1]))
        print(f"  {n_train:>7}"
              f"   {np.mean(rows['amp']):>7.4f}+-{np.std(rows['amp']):<6.4f}"
              f"  {np.mean(rows['phase']):>7.4f}+-{np.std(rows['phase']):<6.4f}"
              f"  {np.mean(rows['cls']):>7.4f}+-{np.std(rows['cls']):<6.4f}")

    print("""
  READING: if the phase encoding leads at small n and the classical
  representation only catches up as data grows, the compact encoding is
  doing real work — and it is work that matters in a domain where labelled
  cases are scarce. If classical leads everywhere, phase encoding is a
  tidier way to write down something already available.
""")


# ═══════════════════════════════ PART B: SUPERPOSITION OF AMBIGUITY
def ambiguity_scaling():
    """
    Golden Hour's real situation: n evidence items, each with k plausible
    readings. The question 'what is the total amount, over all consistent
    readings?' requires an expectation over k^n branches.

      classical exact      enumerate k^n combinations
      classical sampling   Monte Carlo over branches (what engineers write)
      amplitude encoding   one state of n*log2(k) qubits; the expectation is
                           a single inner product, no enumeration
    """
    print("=" * 78)
    print("PART B  SUPERPOSITION OF AMBIGUOUS READINGS")
    print("=" * 78)
    print("""  Setting: n evidence items, each with k plausible OCR readings weighted
  by confidence. Question: the expected total, and the variance, over all
  jointly-consistent readings.
""")
    rng = np.random.default_rng(0)
    k = 3
    print(f"  {'items':>6} {'branches':>14} {'exact enum':>12} "
          f"{'amplitude':>11} {'MC n=10k':>11} {'MC error':>10}")

    for n in (4, 8, 12, 16, 20):
        vals = rng.uniform(1000, 90000, (n, k))
        conf = rng.dirichlet(np.ones(k) * 2.0, n)          # confidences per item
        branches = k ** n

        # amplitude-encoded expectation: separable, one pass
        t0 = time.perf_counter()
        amp_mean = float((vals * conf).sum())
        amp_var = float((conf * vals ** 2).sum() - (conf * vals).sum(1) @ (conf * vals).sum(1).T
                        if False else (conf * vals ** 2).sum() -
                        ((conf * vals).sum(1) ** 2).sum())
        t_amp = time.perf_counter() - t0

        # exact enumeration, only where feasible
        if branches <= 3 ** 12:
            t0 = time.perf_counter()
            tot = 0.0
            for combo in itertools.product(range(k), repeat=n):
                w = np.prod([conf[i, c] for i, c in enumerate(combo)])
                tot += w * sum(vals[i, c] for i, c in enumerate(combo))
            t_enum = time.perf_counter() - t0
            enum_s = f"{t_enum:>11.3f}s"
        else:
            enum_s = f"{'infeasible':>12}"

        # Monte Carlo, the honest engineering baseline
        t0 = time.perf_counter()
        S = 10000
        idx = np.array([rng.choice(k, size=S, p=conf[i]) for i in range(n)])
        mc = vals[np.arange(n)[:, None], idx].sum(0).mean()
        t_mc = time.perf_counter() - t0
        err = abs(mc - amp_mean) / amp_mean

        print(f"  {n:>6} {branches:>14,} {enum_s} {t_amp*1000:>10.3f}ms "
              f"{t_mc*1000:>10.1f}ms {err:>9.4%}")

    print("""
  READING: enumeration dies at ~10^6 branches. The amplitude formulation is
  exact and instant. But Monte Carlo gets within a fraction of a percent in
  milliseconds, and it is twelve lines of numpy.

  The honest distinction: for SEPARABLE questions (expected total, variance)
  the amplitude form is a closed-form identity, not a quantum advantage —
  it is the same factorisation a statistician would write, and no qubits are
  involved. Where it stops being separable is the interesting case, tested
  next.
""")


def constrained_ambiguity():
    """
    The non-separable case: readings are CONSTRAINED against each other.

    A transaction reference read from a screenshot must match the one in the
    bank SMS. A timestamp must be consistent with the message ordering. Once
    branches are coupled by constraints, the expectation no longer factorises
    and the closed form above does not apply.

    This is exactly a constraint-satisfaction weighting problem, which is
    #P-hard in general — and the honest classical baseline is belief
    propagation or MCMC, not enumeration.
    """
    print("=" * 78)
    print("PART B2  CONSTRAINED (NON-SEPARABLE) AMBIGUITY")
    print("=" * 78)
    print("""  Readings are coupled: the reference on the screenshot must match the
  bank SMS, timestamps must respect message order. The expectation no
  longer factorises.
""")
    rng = np.random.default_rng(1)
    k = 3
    print(f"  {'items':>6} {'branches':>12} {'exact':>10} {'belief prop':>13} "
          f"{'BP error':>10}")

    for n in (4, 6, 8, 10, 12):
        vals = rng.uniform(1000, 90000, (n, k))
        conf = rng.dirichlet(np.ones(k) * 2.0, n)
        # pairwise compatibility between consecutive items
        compat = rng.random((n - 1, k, k)) * 0.9 + 0.1

        # exact: enumerate with constraint weights
        t0 = time.perf_counter()
        num = den = 0.0
        for combo in itertools.product(range(k), repeat=n):
            w = np.prod([conf[i, c] for i, c in enumerate(combo)])
            for i in range(n - 1):
                w *= compat[i, combo[i], combo[i + 1]]
            num += w * sum(vals[i, c] for i, c in enumerate(combo))
            den += w
        exact = num / den
        t_ex = time.perf_counter() - t0

        # belief propagation on a chain: EXACT for this structure, linear time
        t0 = time.perf_counter()
        fwd = np.zeros((n, k)); fwd[0] = conf[0]
        for i in range(1, n):
            fwd[i] = conf[i] * (fwd[i - 1] @ compat[i - 1])
        bwd = np.zeros((n, k)); bwd[-1] = 1.0
        for i in range(n - 2, -1, -1):
            bwd[i] = compat[i] @ (conf[i + 1] * bwd[i + 1])
        marg = fwd * bwd
        marg = marg / marg.sum(1, keepdims=True)
        bp = float((marg * vals).sum())
        t_bp = time.perf_counter() - t0

        print(f"  {n:>6} {k**n:>12,} {t_ex:>9.3f}s {t_bp*1000:>12.3f}ms "
              f"{abs(bp-exact)/exact:>9.5%}")

    print("""
  READING: the evidence-constraint graph in a fraud case is a CHAIN or a
  TREE — a screenshot constrains the SMS beside it, a timestamp constrains
  its neighbours. On chains and trees, belief propagation is EXACT and
  LINEAR. The exponential branch count is an illusion created by writing
  the problem down badly.

  A quantum representation would only earn its place if the constraint graph
  were densely cyclic AND high-treewidth. Evidence bundles are not: each
  artifact touches a handful of others, and the graph an investigator draws
  is close to a tree by construction.
""")


def main():
    part_a()
    ambiguity_scaling()
    constrained_ambiguity()

    print("=" * 78)
    print("VERDICT ON PROBE 8")
    print("=" * 78)
    print("""
  The encoding ideas from the HDQS/VFX line are real and they were being
  wasted by the earlier probes, which used order-destroying aggregate
  features. Phase encoding recovers information those features threw away.

  But recovering it is not the same as needing quantum mechanics to recover
  it. Order is cheap classically; ambiguity over a tree-structured
  constraint graph is cheap classically. Both were tested against the
  baseline a competent engineer would actually write, not against a
  strawman.

  What survives as a genuine contribution is a REPRESENTATIONAL one: a
  single complex-valued object that carries magnitude and order together,
  and holds unresolved ambiguity without committing. That is a good design
  for a forensic record — it keeps provenance and uncertainty in one place
  instead of scattering them across fields. It is not a speedup, and calling
  it one would be the same mistake the field keeps making.
""")


if __name__ == "__main__":
    main()
