#!/usr/bin/env python3
"""
FOLLOW-UP — interrogating the one apparent positive.

Probe 6 produced a single result above the classical baseline:

    classical features + 8 quantum kPCA components
    AUC +0.0055,  p = 0.0316

That is exactly the shape of result that gets reported as a hybrid quantum
advantage. Before believing it, three things have to be checked, and each
one can kill it independently:

  1. MULTIPLE COMPARISONS. Three component counts were tried (4, 8, 16) and
     one was significant. Correcting across the three is mandatory.

  2. RANDOM-FEATURE CONTROL. Does adding 8 RANDOM components of matched
     scale help just as much? Extra features can improve an SVM through
     regularisation and dimensionality alone. If random works too, the
     quantum content is irrelevant.

  3. SEED STABILITY. One dataset draw is one sample. If the effect does not
     survive independent draws, it is noise that happened to land favourably.

A result that survives all three is real. A result that fails any one is not.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wilcoxon

from quantum import qkernels as qk
from quantum import metrics as mx
from quantum.dataset import make_dataset, scale_for_encoding

NQ = 8
COMPONENTS = (4, 8, 16)


def med_rbf(X):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    return mx.normalise_kernel(rbf_kernel(X, X, gamma=1.0 / (4.0 * med)))


def auc_folds(K, y, seed=0, n_repeats=6):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats,
                                   random_state=seed)
    out = []
    for tr, te in rskf.split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        out.append(roc_auc_score(y[te], clf.decision_function(K[np.ix_(te, tr)])))
    return np.array(out)


def augment(X, Z):
    return np.hstack([StandardScaler().fit_transform(X),
                      StandardScaler().fit_transform(Z)])


def run_seed(seed, n_repeats=6):
    X_raw, y = make_dataset(n=300, seed=seed)
    X = scale_for_encoding(X_raw)
    base = auc_folds(med_rbf(X), y, seed=seed, n_repeats=n_repeats)

    Kq = mx.normalise_kernel(
        qk.fidelity_kernel(qk.reupload_states(X, NQ, layers=3)))
    rng = np.random.default_rng(seed + 999)

    row = {"base": base}
    for n_comp in COMPONENTS:
        # quantum-derived components
        Zq = KernelPCA(n_components=n_comp, kernel="precomputed").fit_transform(Kq)
        row[f"q{n_comp}"] = auc_folds(med_rbf(augment(X, Zq)), y,
                                      seed=seed, n_repeats=n_repeats)

        # CONTROL A: gaussian random projections of the SAME data
        W = rng.normal(size=(X.shape[1], n_comp))
        row[f"rand{n_comp}"] = auc_folds(med_rbf(augment(X, X @ W)), y,
                                         seed=seed, n_repeats=n_repeats)

        # CONTROL B: classical kPCA components (an RBF kernel's own kPCA)
        Zc = KernelPCA(n_components=n_comp,
                       kernel="precomputed").fit_transform(med_rbf(X))
        row[f"ckpca{n_comp}"] = auc_folds(med_rbf(augment(X, Zc)), y,
                                          seed=seed, n_repeats=n_repeats)

        # CONTROL C: pure noise columns, no information at all
        Zn = rng.normal(size=(X.shape[0], n_comp))
        row[f"noise{n_comp}"] = auc_folds(med_rbf(augment(X, Zn)), y,
                                          seed=seed, n_repeats=n_repeats)
    return row


def wtest(a, b):
    if np.allclose(a, b):
        return 0.0, 1.0
    _, p = wilcoxon(a, b)
    return float((a - b).mean()), float(p)


def main():
    print("=" * 78)
    print("FOLLOW-UP  interrogating the one apparent positive")
    print("=" * 78)
    print("  Claim under test: classical features + 8 quantum kPCA components")
    print("  beat classical features alone (+0.0055 AUC, p = 0.0316).\n")

    # ─────────────────────────────────── check 1 + 2 on the original seed
    print("=" * 78)
    print("CHECK 1 & 2  multiple comparisons and the random-feature control")
    print("=" * 78)
    r = run_seed(1)
    print(f"  {'added components':<34} {'AUC':>9} {'delta':>9} {'p':>9}")
    raw_p = {}
    for n in COMPONENTS:
        for tag, label in (("q", "quantum kPCA"),
                           ("rand", "random projection"),
                           ("ckpca", "classical kPCA"),
                           ("noise", "pure noise")):
            d, p = wtest(r[f"{tag}{n}"], r["base"])
            if tag == "q":
                raw_p[n] = p
            print(f"  {label + f' x{n}':<34} {r[f'{tag}{n}'].mean():>9.4f} "
                  f"{d:>+9.4f} {p:>9.4f}")
        print()

    m = len(COMPONENTS)
    print(f"  Holm correction across the {m} component counts tried:")
    for n, p in sorted(raw_p.items(), key=lambda kv: kv[1]):
        print(f"    quantum kPCA x{n:<3} raw p = {p:.4f}   "
              f"adjusted = {min(1.0, m * p):.4f}   "
              f"{'significant' if min(1.0, m*p) < 0.05 else 'NOT significant'}")

    # ─────────────────────────────────────────────── check 3 seed stability
    print("\n" + "=" * 78)
    print("CHECK 3  does it survive independent dataset draws")
    print("=" * 78)
    print(f"  {'seed':>5} {'base':>9} {'quantum x8':>12} {'delta':>9} "
          f"{'random x8':>11} {'delta':>9}")
    dq, dr = [], []
    for seed in range(8):
        rr = run_seed(seed, n_repeats=4)
        d1, _ = wtest(rr["q8"], rr["base"])
        d2, _ = wtest(rr["rand8"], rr["base"])
        dq.append(d1); dr.append(d2)
        print(f"  {seed:>5} {rr['base'].mean():>9.4f} {rr['q8'].mean():>12.4f} "
              f"{d1:>+9.4f} {rr['rand8'].mean():>11.4f} {d2:>+9.4f}")

    dq, dr = np.array(dq), np.array(dr)
    print(f"\n  quantum x8 delta across seeds : {dq.mean():+.4f} "
          f"+- {dq.std():.4f}   (wins {int((dq>0).sum())}/8)")
    print(f"  random  x8 delta across seeds : {dr.mean():+.4f} "
          f"+- {dr.std():.4f}   (wins {int((dr>0).sum())}/8)")
    _, p_seed = wilcoxon(dq, np.zeros_like(dq)) if not np.allclose(dq, 0) else (0, 1.0)
    _, p_vs_rand = wilcoxon(dq, dr) if not np.allclose(dq, dr) else (0, 1.0)
    print(f"\n  quantum delta differs from zero      : p = {p_seed:.4f}")
    print(f"  quantum delta differs from random    : p = {p_vs_rand:.4f}")

    # ───────────────────────────────────────────────────────────── verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    survives = (min(1.0, m * raw_p[8]) < 0.05 and dq.mean() > 0
                and p_seed < 0.05 and p_vs_rand < 0.05)
    if survives:
        print("\n  The effect SURVIVES all three checks. That is a real,")
        print("  quantum-specific hybrid gain and should be reported as one.")
    else:
        print("\n  The effect does NOT survive.")
        reasons = []
        if min(1.0, m * raw_p[8]) >= 0.05:
            reasons.append("fails multiple-comparison correction across the "
                           "three component counts tried")
        if dq.mean() <= 0 or p_seed >= 0.05:
            reasons.append("does not reproduce across independent dataset "
                           "draws")
        if p_vs_rand >= 0.05:
            reasons.append("is indistinguishable from adding the same number "
                           "of RANDOM features, so the gain is dimensionality "
                           "and regularisation, not quantum information")
        for i, r_ in enumerate(reasons, 1):
            print(f"    {i}. It {r_}.")
        print("\n  This is the mechanism by which hybrid quantum advantage gets")
        print("  reported in the literature: one favourable configuration, one")
        print("  seed, no random-feature control, no correction for the")
        print("  configurations that were tried and discarded.")


if __name__ == "__main__":
    main()
