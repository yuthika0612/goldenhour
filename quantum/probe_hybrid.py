#!/usr/bin/env python3
"""
PROBE 6 — can a HYBRID beat classical alone?

The hypothesis is reasonable and does not follow from the earlier negatives:
a component can be weak alone and still add value in combination, if its
errors are DIFFERENT from the other component's. Ensembles exploit exactly
that. So this deserves its own test rather than an inference.

Four fusion strategies, all evaluated against the best classical baseline:

  H1  KERNEL MIXING      K = (1-a) K_classical + a K_quantum, a swept
  H2  FEATURE FUSION     classical features ++ quantum-derived features
                         (kernel-PCA components of the quantum kernel)
  H3  DECISION ENSEMBLE  average the decision functions of both models
  H4  RESIDUAL BOOSTING  quantum model trained on classical model's errors

Three controls, without which any positive result is uninterpretable:

  C1  RANDOM KERNEL      same fusion, but the quantum kernel is replaced by
                         a random positive-definite kernel of matched
                         spectrum. If this helps as much, the gain is
                         ensembling, not quantum information.
  C2  SHUFFLED QUANTUM   the quantum kernel with rows/columns permuted, so
                         its structure is destroyed but its statistics kept.
  C3  SECOND CLASSICAL   fusing two classical kernels. This is the honest
                         alternative a practitioner would reach for first.

Statistics:
  - repeated stratified CV (5 folds x 6 repeats = 30 paired measurements)
  - Wilcoxon signed-rank on PAIRED fold scores, not on means
  - NESTED selection of the mixing weight: sweeping a and reporting the best
    is selection bias, so a is chosen on inner folds only
  - Holm correction across the strategies tested
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from quantum import qkernels as qk
from quantum import metrics as mx
from quantum.dataset import make_dataset, scale_for_encoding

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]


# ───────────────────────────────────────────────────────────── utilities
def norm(K):
    return mx.normalise_kernel(K)


def auc_folds(K, y, n_splits=5, n_repeats=6, seed=0):
    """Paired per-fold AUCs. Identical splits for every kernel compared."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    out = []
    for tr, te in rskf.split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        out.append(roc_auc_score(y[te], clf.decision_function(K[np.ix_(te, tr)])))
    return np.array(out)


def nested_mix_auc(Ka, Kb, y, n_splits=5, n_repeats=6, seed=0):
    """
    Mix two kernels with the weight chosen on INNER folds only.
    This is what stops the alpha sweep from manufacturing a win.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    out, chosen = [], []
    for tr, te in rskf.split(np.zeros(len(y)), y):
        best_a, best_s = 0.0, -np.inf
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for a in ALPHAS:
            K = (1 - a) * Ka + a * Kb
            sc = []
            for itr, ite in inner.split(np.zeros(len(tr)), y[tr]):
                t1, t2 = tr[itr], tr[ite]
                clf = SVC(kernel="precomputed", C=1.0)
                clf.fit(K[np.ix_(t1, t1)], y[t1])
                sc.append(roc_auc_score(
                    y[t2], clf.decision_function(K[np.ix_(t2, t1)])))
            m = np.mean(sc)
            if m > best_s:
                best_s, best_a = m, a
        chosen.append(best_a)
        K = (1 - best_a) * Ka + best_a * Kb
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        out.append(roc_auc_score(
            y[te], clf.decision_function(K[np.ix_(te, tr)])))
    return np.array(out), np.array(chosen)


def paired_test(a_scores, b_scores, label):
    """Wilcoxon on paired folds. Returns (delta, p)."""
    d = a_scores - b_scores
    if np.allclose(d, 0):
        return 0.0, 1.0
    stat, p = wilcoxon(a_scores, b_scores)
    return float(d.mean()), float(p)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni correction across the family of tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def random_kernel_like(K, seed=0):
    """
    Control C1: a random PSD kernel with the SAME eigenvalue spectrum as the
    quantum kernel, but random eigenvectors. Matched statistics, no structure.
    """
    rng = np.random.default_rng(seed)
    ev = np.linalg.eigvalsh(norm(K))
    ev = np.clip(ev, 0, None)
    n = K.shape[0]
    A = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    R = (Q * ev) @ Q.T
    return norm(R + 1e-9 * np.eye(n))


def shuffled_kernel(K, seed=0):
    """Control C2: permute the sample order, destroying the correspondence."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(K.shape[0])
    return norm(K)[np.ix_(p, p)]


# ─────────────────────────────────────────────────────────────── the run
def main():
    N, NQ, SEED = 300, 8, 1
    X_raw, y = make_dataset(n=N, seed=SEED)
    X = scale_for_encoding(X_raw)

    Kc_all = {k: norm(v) for k, v in qk.classical_kernels(X).items()}
    Kq_all = {
        "angle": norm(qk.fidelity_kernel(qk.angle_states(X, NQ))),
        "zz": norm(qk.fidelity_kernel(qk.zz_states(X, NQ, reps=2))),
        "reupload": norm(qk.fidelity_kernel(qk.reupload_states(X, NQ, layers=3))),
    }

    base = {k: auc_folds(v, y, seed=SEED) for k, v in Kc_all.items()}
    best_c = max(base, key=lambda k: base[k].mean())
    Kc = Kc_all[best_c]
    qbase = {k: auc_folds(v, y, seed=SEED) for k, v in Kq_all.items()}
    best_q = max(qbase, key=lambda k: qbase[k].mean())
    Kq = Kq_all[best_q]

    print("=" * 78)
    print("PROBE 6  HYBRID vs CLASSICAL")
    print("=" * 78)
    print(f"  {N} cases, {NQ} qubits, 5-fold x 6 repeats = "
          f"{len(base[best_c])} paired measurements per comparison\n")
    print(f"  best classical : {best_c:<12} AUC "
          f"{base[best_c].mean():.4f} +- {base[best_c].std():.4f}")
    print(f"  best quantum   : {best_q:<12} AUC "
          f"{qbase[best_q].mean():.4f} +- {qbase[best_q].std():.4f}")

    # ---------------------------------------------- complementarity first
    print("\n" + "=" * 78)
    print("STEP 1  ARE THE ERRORS COMPLEMENTARY?  (the precondition for fusion)")
    print("=" * 78)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ec, eq = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for tr, te in skf.split(np.zeros(len(y)), y):
        for K, err in ((Kc, ec), (Kq, eq)):
            clf = SVC(kernel="precomputed", C=1.0)
            clf.fit(K[np.ix_(tr, tr)], y[tr])
            err[te] = clf.predict(K[np.ix_(te, tr)]) != y[te]
    both = (ec & eq).sum()
    only_c, only_q = (ec & ~eq).sum(), (~ec & eq).sum()
    jac = both / max((ec | eq).sum(), 1)
    print(f"  classical wrong only   : {only_c}")
    print(f"  quantum wrong only     : {only_q}")
    print(f"  both wrong             : {both}")
    print(f"  error overlap (Jaccard): {jac:.3f}")
    print(f"\n  Quantum uniquely fixes {only_c} cases the classical model misses,")
    print(f"  while introducing {only_q} new errors. Net "
          f"{only_c - only_q:+d}.")
    if only_c <= only_q:
        print("  Fusion has no error budget to exploit here.")

    # ---------------------------------------------- H1 kernel mixing sweep
    print("\n" + "=" * 78)
    print("STEP 2  H1  KERNEL MIXING  K = (1-a) Kc + a Kq")
    print("=" * 78)
    print(f"  {'alpha':>6} {'AUC':>10} {'vs classical':>14}")
    sweep_best = (-np.inf, None)
    for a in ALPHAS:
        s = auc_folds((1 - a) * Kc + a * Kq, y, seed=SEED)
        d = s.mean() - base[best_c].mean()
        mark = "  <- best" if s.mean() > sweep_best[0] else ""
        if s.mean() > sweep_best[0]:
            sweep_best = (s.mean(), a)
        print(f"  {a:>6.1f} {s.mean():>10.4f} {d:>+14.4f}{mark}")
    print(f"\n  Cherry-picked best alpha = {sweep_best[1]} gives "
          f"{sweep_best[0]:.4f}")
    print("  That number is SELECTION BIAS. The nested result below is the")
    print("  one that would hold on new data.")

    # ---------------------------------------------- nested + controls
    print("\n" + "=" * 78)
    print("STEP 3  NESTED SELECTION AND CONTROLS")
    print("=" * 78)
    print("  Mixing weight chosen on inner folds only. Every control uses the")
    print("  identical procedure, so any advantage must be quantum-specific.\n")

    second_c = max((k for k in Kc_all if k != best_c),
                   key=lambda k: base[k].mean())
    partners = {
        f"H1 quantum ({best_q})": Kq,
        "C1 random PSD kernel": random_kernel_like(Kq, seed=SEED),
        "C2 shuffled quantum": shuffled_kernel(Kq, seed=SEED),
        f"C3 second classical ({second_c})": Kc_all[second_c],
    }

    results, pvals = {}, {}
    print(f"  {'fusion partner':<32} {'AUC':>9} {'delta':>9} {'p':>10} "
          f"{'mean alpha':>11}")
    for name, Kp in partners.items():
        s, alphas = nested_mix_auc(Kc, Kp, y, seed=SEED)
        d, p = paired_test(s, base[best_c], name)
        results[name] = (s, d, alphas)
        pvals[name] = p
        print(f"  {name:<32} {s.mean():>9.4f} {d:>+9.4f} {p:>10.4f} "
              f"{alphas.mean():>11.2f}")

    adj = holm(pvals)
    print(f"\n  {'fusion partner':<32} {'raw p':>9} {'Holm-adj p':>12} "
          f"{'significant':>13}")
    for name in partners:
        sig = "yes" if adj[name] < 0.05 else "no"
        print(f"  {name:<32} {pvals[name]:>9.4f} {adj[name]:>12.4f} "
              f"{sig:>13}")

    # ---------------------------------------------- H3 decision ensemble
    print("\n" + "=" * 78)
    print("STEP 4  H3  DECISION-LEVEL ENSEMBLE")
    print("=" * 78)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=SEED)
    ens, cls_only = [], []
    for tr, te in rskf.split(np.zeros(len(y)), y):
        dfs = []
        for K in (Kc, Kq):
            clf = SVC(kernel="precomputed", C=1.0)
            clf.fit(K[np.ix_(tr, tr)], y[tr])
            d = clf.decision_function(K[np.ix_(te, tr)])
            dfs.append((d - d.mean()) / (d.std() + 1e-12))
        ens.append(roc_auc_score(y[te], 0.5 * dfs[0] + 0.5 * dfs[1]))
        cls_only.append(roc_auc_score(y[te], dfs[0]))
    ens, cls_only = np.array(ens), np.array(cls_only)
    d, p = paired_test(ens, cls_only, "ensemble")
    print(f"  classical alone : {cls_only.mean():.4f}")
    print(f"  50/50 ensemble  : {ens.mean():.4f}")
    print(f"  delta {d:+.4f}   p = {p:.4f}")

    # ---------------------------------------------- H2 feature fusion
    print("\n" + "=" * 78)
    print("STEP 5  H2  FEATURE-LEVEL FUSION")
    print("=" * 78)
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import StandardScaler
    for n_comp in (4, 8, 16):
        kpca = KernelPCA(n_components=n_comp, kernel="precomputed")
        Zq = kpca.fit_transform(Kq)
        Xh = np.hstack([StandardScaler().fit_transform(X),
                        StandardScaler().fit_transform(Zq)])
        Kh = norm(qk.classical_kernels(Xh)[best_c]
                  if best_c in qk.classical_kernels(Xh) else
                  qk.classical_kernels(Xh)["rbf_median"])
        s = auc_folds(Kh, y, seed=SEED)
        d, p = paired_test(s, base[best_c], f"fusion{n_comp}")
        print(f"  + {n_comp:>2} quantum kPCA components : AUC {s.mean():.4f} "
              f"  delta {d:+.4f}   p = {p:.4f}")

    # ---------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    hname = f"H1 quantum ({best_q})"
    hs, hd, halpha = results[hname]
    c1 = results["C1 random PSD kernel"][1]
    c3 = results[f"C3 second classical ({second_c})"][1]
    print(f"""
  Hybrid (nested)        : {hd:+.4f} AUC vs classical alone, Holm p = {adj[hname]:.4f}
  Random-kernel control  : {c1:+.4f}
  Second-classical fusion: {c3:+.4f}

  The controls are the point. If fusing a RANDOM kernel moves the score as
  much as fusing the quantum one, the effect is ensembling and regularisation,
  not quantum information. If fusing a second CLASSICAL kernel does better,
  that is what a practitioner should ship.

  Mean mixing weight chosen by nested selection: {halpha.mean():.2f}
  (a weight near 0 means the procedure is discarding the quantum kernel when
  it cannot see the test fold).
""")


if __name__ == "__main__":
    main()
