#!/usr/bin/env python3
"""
Robustness sweep.

One run is an anecdote. This repeats the comparison across seeds, dataset
sizes, qubit counts and encoding scales, and deliberately searches for the
regime most favourable to the quantum side rather than the most convenient
one. If a quantum advantage exists anywhere in this problem, this is where
it would show.

Regimes tested:
  - small training sets (the usual claim: quantum kernels help when data is
    scarce)
  - varying qubit count (8, 10, 12)
  - varying encoding scale (bandwidth of the quantum feature map: the
    equivalent of tuning gamma for an RBF, and often omitted in papers,
    which biases comparisons against the quantum side)
  - all three encodings
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum import qkernels as qk
from quantum import metrics as mx
from quantum.dataset import make_dataset, scale_for_encoding


def one_run(n, nq, seed, scale, shots=None):
    X_raw, y = make_dataset(n=n, seed=seed)
    X = scale_for_encoding(X_raw)
    Kc = qk.classical_kernels(X)

    Kq = {
        "angle": qk.fidelity_kernel(qk.angle_states(X, nq, scale=scale)),
        "zz": qk.fidelity_kernel(qk.zz_states(X, nq, reps=2, scale=scale)),
        "reupload": qk.fidelity_kernel(
            qk.reupload_states(X, nq, layers=3, scale=scale)),
    }
    if shots:
        Kq = {k: qk.add_shot_noise(v, shots, seed=seed) for k, v in Kq.items()}

    out = {}
    for name, K in Kc.items():
        out[("C", name)] = mx.cv_scores(K, y, seed=seed)["auc_mean"]
    for name, K in Kq.items():
        g = mx.geometric_difference_vs_family(Kc, K)["min_g"]
        out[("Q", name)] = mx.cv_scores(K, y, seed=seed)["auc_mean"]
        out[("G", name)] = g
    return out


def main():
    print("=" * 78)
    print("SWEEP A  scale of the quantum feature map (its bandwidth)")
    print("=" * 78)
    print("  Tuning this is the quantum equivalent of tuning RBF gamma.")
    print("  Omitting it biases comparisons against the quantum side.\n")
    print(f"  {'scale':>6} {'angle AUC':>11} {'zz AUC':>9} {'reup AUC':>10} "
          f"{'best C AUC':>12} {'max g':>8}")
    best_overall = {"gap": -9, "detail": None}
    for scale in [0.25, 0.5, 1.0, 1.5, 2.0]:
        r = one_run(240, 8, seed=1, scale=scale)
        bc = max(v for (t, _), v in r.items() if t == "C")
        qs = {k[1]: v for k, v in r.items() if k[0] == "Q"}
        gs = {k[1]: v for k, v in r.items() if k[0] == "G"}
        print(f"  {scale:>6.2f} {qs['angle']:>11.3f} {qs['zz']:>9.3f} "
              f"{qs['reupload']:>10.3f} {bc:>12.3f} {max(gs.values()):>8.2f}")
        gap = max(qs.values()) - bc
        if gap > best_overall["gap"]:
            best_overall = {"gap": gap, "detail": f"scale={scale}"}

    print("\n" + "=" * 78)
    print("SWEEP B  qubit count at the best scale")
    print("=" * 78)
    print(f"  {'qubits':>7} {'angle':>9} {'zz':>9} {'reupload':>10} "
          f"{'best C':>9} {'max g':>8} {'sqrt(N)':>9}")
    for nq in [6, 8, 10, 12]:
        r = one_run(240, nq, seed=1, scale=1.0)
        bc = max(v for (t, _), v in r.items() if t == "C")
        qs = {k[1]: v for k, v in r.items() if k[0] == "Q"}
        gs = {k[1]: v for k, v in r.items() if k[0] == "G"}
        print(f"  {nq:>7} {qs['angle']:>9.3f} {qs['zz']:>9.3f} "
              f"{qs['reupload']:>10.3f} {bc:>9.3f} {max(gs.values()):>8.2f} "
              f"{np.sqrt(240):>9.1f}")

    print("\n" + "=" * 78)
    print("SWEEP C  scarce data, the regime where quantum kernels are claimed")
    print("         to help most")
    print("=" * 78)
    print(f"  {'N':>5} {'best quantum':>14} {'best classical':>16} {'gap':>8}")
    for n in [60, 100, 160, 240, 320]:
        r = one_run(n, 8, seed=2, scale=1.0)
        bc = max(v for (t, _), v in r.items() if t == "C")
        bq = max(v for (t, _), v in r.items() if t == "Q")
        print(f"  {n:>5} {bq:>14.3f} {bc:>16.3f} {bq - bc:>+8.3f}")

    print("\n" + "=" * 78)
    print("SWEEP D  seed stability at the standard setting")
    print("=" * 78)
    gaps, gmaxes = [], []
    print(f"  {'seed':>5} {'best quantum':>14} {'best classical':>16} "
          f"{'gap':>8} {'max g':>8}")
    for seed in range(6):
        r = one_run(240, 8, seed=seed, scale=1.0)
        bc = max(v for (t, _), v in r.items() if t == "C")
        bq = max(v for (t, _), v in r.items() if t == "Q")
        gm = max(v for (t, _), v in r.items() if t == "G")
        gaps.append(bq - bc)
        gmaxes.append(gm)
        print(f"  {seed:>5} {bq:>14.3f} {bc:>16.3f} {bq - bc:>+8.3f} {gm:>8.2f}")
    print(f"\n  gap across seeds: {np.mean(gaps):+.3f} +- {np.std(gaps):.3f}")
    print(f"  max g across seeds: {np.mean(gmaxes):.2f} +- {np.std(gmaxes):.2f}"
          f"   (threshold sqrt(N) = {np.sqrt(240):.1f})")
    print(f"\n  Most favourable configuration found for quantum: "
          f"{best_overall['detail']}, gap {best_overall['gap']:+.3f}")


if __name__ == "__main__":
    main()
