#!/usr/bin/env python3
"""
Quantum vs classical representations on Golden Hour case features.

    python -m quantum.experiment
    python -m quantum.experiment --n 400 --qubits 8 --seed 1

Protocol, fixed before looking at any result:

  Step 1  Build case feature vectors, scale to [0, pi].
  Step 2  Build classical kernels (linear, RBF at three bandwidths).
  Step 3  Build quantum kernels (angle, zz, reupload) by exact simulation.
  Step 4  GEOMETRIC DIFFERENCE first. If g <= sqrt(N), a classical kernel
          can match the quantum one for ANY labelling, and no accuracy
          result can be evidence of advantage. Record this before training.
  Step 5  Kernel-target alignment: does the geometry line up with the labels.
  Step 6  Same-splits cross-validated accuracy and AUC.
  Step 7  Learning curves: does any gap survive as data grows.
  Step 8  Shot-noise realism: recompute with finite measurement shots.

Whatever comes out is the result. The design does not privilege either side.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum import qkernels as qk          # noqa: E402
from quantum import metrics as mx           # noqa: E402
from quantum.dataset import make_dataset, scale_for_encoding   # noqa: E402
from quantum.features import FEATURE_NAMES  # noqa: E402


def hr(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t)
        print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--json", help="write results to this path")
    args = ap.parse_args()

    N, NQ = args.n, args.qubits
    results = {"config": vars(args)}

    # ---------------------------------------------------------- step 1
    X_raw, y = make_dataset(n=N, seed=args.seed)
    X = scale_for_encoding(X_raw)
    hr("DATA")
    print(f"  {N} synthetic cases, {X.shape[1]} features, "
          f"{int(y.sum())} fraud / {int((1-y).sum())} legitimate")
    print(f"  features: {', '.join(FEATURE_NAMES[:6])} ...")
    print(f"  encoding uses the first {NQ} features (one per qubit)")
    print("  SYNTHETIC DATA: this measures the representation, not detection "
          "performance.")

    # ---------------------------------------------------------- step 2
    Kc = qk.classical_kernels(X)
    # single best classical reference for the geometry test: highest KTA
    kta_c = {k: mx.kernel_target_alignment(v, 2 * y - 1) for k, v in Kc.items()}
    best_c = max(kta_c, key=kta_c.get)

    # ---------------------------------------------------------- step 3
    states = {
        "angle":    qk.angle_states(X, NQ),
        "zz":       qk.zz_states(X, NQ, reps=2),
        "reupload": qk.reupload_states(X, NQ, layers=3),
    }
    Kq = {k: qk.fidelity_kernel(v) for k, v in states.items()}

    # ---------------------------------------------------------- step 4
    hr("STEP 1  GEOMETRIC DIFFERENCE  (asked before any training)")
    print(f"  Threshold: advantage requires g >> sqrt(N) = {np.sqrt(N):.1f}")
    print(f"  Classical reference kernel: {best_c}\n")
    print("  g is taken as the MINIMUM over the classical family: advantage")
    print("  must beat every classical kernel, not one weak one.\n")
    print(f"  {'quantum kernel':<12} {'min g':>8} {'closest classical':>19} "
          f"{'verdict':>36}")
    geo = {}
    for name, K in Kq.items():
        gd = mx.geometric_difference_vs_family(Kc, K)
        geo[name] = {"g_vs_best": gd["min_g"],
                     "closest_classical": gd["closest_classical"],
                     "per_kernel": gd["per_kernel"]}
        verdict = ("advantage possible" if gd["min_g"] > np.sqrt(N)
                   else "classical can match it for ANY labels")
        print(f"  {name:<12} {gd['min_g']:>8.2f} {gd['closest_classical']:>19} "
              f"{verdict:>36}")

    # internal control: the product-state encoding is classically simulable,
    # so its g must NOT come out large. If it does, the metric is broken.
    if geo["angle"]["g_vs_best"] > np.sqrt(N):
        print("\n  WARNING: the product-state control (angle) reports a large g.")
        print("  That encoding is classically simulable in O(n), so this")
        print("  indicates a conditioning problem in the metric, not physics.")
    else:
        print("\n  Control check passed: the product-state encoding (angle), "
              "which is")
        print("  classically simulable, correctly reports g below threshold.")
    results["geometric_difference"] = geo
    results["sqrt_N"] = float(np.sqrt(N))

    # ---------------------------------------------------------- step 5
    hr("STEP 2  KERNEL-TARGET ALIGNMENT  (does the geometry track the labels)")
    ys = 2 * y - 1
    align = {}
    print(f"  {'kernel':<14} {'KTA':>8} {'eff.dim':>9} {'mean off-diag':>15}")
    for name, K in list(Kc.items()) + list(Kq.items()):
        a = mx.kernel_target_alignment(K, ys)
        ed = mx.effective_dimension(K)
        sp = mx.expressivity_spread(K)
        align[name] = {"kta": a, "eff_dim": ed, "spread": sp}
        tag = "Q" if name in Kq else "C"
        print(f"  [{tag}] {name:<10} {a:>8.4f} {ed:>9d} {sp:>15.4f}")
    results["alignment"] = align

    # ---------------------------------------------------------- step 6
    hr("STEP 3  TASK PERFORMANCE  (identical 5-fold splits for every kernel)")
    print(f"  {'kernel':<14} {'bal.acc':>16} {'ROC-AUC':>16}")
    perf = {}
    for name, K in list(Kc.items()) + list(Kq.items()):
        s = mx.cv_scores(K, y, n_splits=5, seed=args.seed)
        perf[name] = s
        tag = "Q" if name in Kq else "C"
        print(f"  [{tag}] {name:<10} {s['acc_mean']:>10.3f} +-{s['acc_std']:<4.3f}"
              f" {s['auc_mean']:>10.3f} +-{s['auc_std']:<4.3f}")
    results["performance"] = perf

    best_cls = max(Kc, key=lambda k: perf[k]["auc_mean"])
    best_qnt = max(Kq, key=lambda k: perf[k]["auc_mean"])
    gap = perf[best_qnt]["auc_mean"] - perf[best_cls]["auc_mean"]
    pooled = np.hypot(perf[best_qnt]["auc_std"], perf[best_cls]["auc_std"])
    print(f"\n  best classical: {best_cls} (AUC {perf[best_cls]['auc_mean']:.3f})")
    print(f"  best quantum  : {best_qnt} (AUC {perf[best_qnt]['auc_mean']:.3f})")
    print(f"  difference    : {gap:+.3f}  "
          f"({'within' if abs(gap) < pooled else 'beyond'} fold-to-fold noise "
          f"of +-{pooled:.3f})")
    results["headline"] = {"best_classical": best_cls, "best_quantum": best_qnt,
                           "auc_gap": float(gap), "pooled_std": float(pooled)}

    # ---------------------------------------------------------- step 7
    hr("STEP 4  LEARNING CURVES  (does any gap survive more data)")
    sizes = [20, 40, 80, 140, 200]
    sizes = [s for s in sizes if s < N - 60]
    curves = {}
    print(f"  {'n_train':>8}  " + "  ".join(f"{k:>12}" for k in
                                            [best_cls, *Kq.keys()]))
    for m in sizes:
        row = []
        for name in [best_cls, *Kq.keys()]:
            K = Kc[name] if name in Kc else Kq[name]
            lc = mx.learning_curve(K, y, [m], n_reps=8, seed=args.seed)
            v = lc.get(m, (np.nan, np.nan))[0]
            curves.setdefault(name, {})[m] = v
            row.append(f"{v:>12.3f}")
        print(f"  {m:>8}  " + "  ".join(row))
    results["learning_curves"] = curves

    # ---------------------------------------------------------- step 8
    hr("STEP 5  SHOT NOISE  (what a real device would return)")
    print(f"  Kernel entries estimated from {args.shots} shots\n")
    print(f"  {'kernel':<12} {'exact AUC':>11} {'noisy AUC':>11} {'change':>9}")
    noisy = {}
    for name, K in Kq.items():
        Kn = qk.add_shot_noise(K, args.shots, seed=args.seed)
        s = mx.cv_scores(Kn, y, n_splits=5, seed=args.seed)
        noisy[name] = s
        d = s["auc_mean"] - perf[name]["auc_mean"]
        print(f"  {name:<12} {perf[name]['auc_mean']:>11.3f} "
              f"{s['auc_mean']:>11.3f} {d:>+9.3f}")
    results["shot_noise"] = noisy

    # ---------------------------------------------------------- verdict
    hr("VERDICT")
    g_max = max(v["g_vs_best"] for v in geo.values())
    if g_max <= np.sqrt(N):
        print(f"  Geometric difference is at most {g_max:.2f}, below "
              f"sqrt(N) = {np.sqrt(N):.1f}.")
        print("  A classical kernel can reproduce these quantum kernels for any")
        print("  labelling of this data. No accuracy figure can demonstrate")
        print("  advantage here; the representation is not doing anything a")
        print("  classical kernel cannot.")
    else:
        print(f"  Geometric difference reaches {g_max:.2f} > sqrt(N) = "
              f"{np.sqrt(N):.1f}: the quantum geometry is genuinely distinct.")
        print("  Whether that helps depends on alignment and task scores above.")

    print(f"\n  On the task, the best quantum kernel is {gap:+.3f} AUC against")
    print(f"  the best classical kernel, {'inside' if abs(gap) < pooled else 'outside'}"
          f" fold-to-fold variation.")
    print("\n  Scope: synthetic cases, 14 engineered features, exact simulation.")
    print("  This measures the representation, not fraud-detection performance.")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\n  results written to {args.json}")
    return results


if __name__ == "__main__":
    main()
