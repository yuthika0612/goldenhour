#!/usr/bin/env python3
"""
Why the quantum kernels lose. Two mechanisms, both measurable.

MECHANISM 1 — the product encoding is a classical kernel in disguise.
  For angle encoding, |<psi(x)|psi(z)>|^2 = prod_q cos^2((x_q - z_q)/2).
  That is a shift-invariant product kernel, computable classically in O(n)
  and closely related to an RBF. So when angle encoding performs well, it is
  not doing anything quantum. Measured here by correlating the angle kernel
  with the best RBF kernel.

MECHANISM 2 — exponential concentration in the entangling maps.
  As qubit count grows, fidelities between distinct inputs collapse toward
  zero, the kernel matrix approaches the identity, and the model can only
  memorise. This is a known failure mode (Thanasilp et al. 2024). Measured
  here as the mean off-diagonal kernel value versus qubit count.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum import qkernels as qk
from quantum import metrics as mx
from quantum.dataset import make_dataset, scale_for_encoding


def main():
    X_raw, y = make_dataset(n=200, seed=3)
    X = scale_for_encoding(X_raw)
    Kc = qk.classical_kernels(X)

    print("=" * 74)
    print("MECHANISM 1  is the quantum kernel secretly a classical one")
    print("=" * 74)
    print("  Correlation between each quantum kernel's off-diagonal entries")
    print("  and the best classical kernel's.\n")
    print(f"  {'encoding':<12} {'corr with RBF':>15} {'reading':>40}")
    for nq in (8, 12):
        Kq = {
            "angle": qk.fidelity_kernel(qk.angle_states(X, nq)),
            "zz": qk.fidelity_kernel(qk.zz_states(X, nq, reps=2)),
            "reupload": qk.fidelity_kernel(qk.reupload_states(X, nq, layers=3)),
        }
        print(f"\n  --- {nq} qubits ---")
        mask = ~np.eye(len(y), dtype=bool)
        ref = mx.normalise_kernel(Kc["rbf_median"])[mask]
        for name, K in Kq.items():
            c = np.corrcoef(mx.normalise_kernel(K)[mask], ref)[0, 1]
            reading = ("essentially the classical kernel" if abs(c) > 0.9
                       else "partly classical" if abs(c) > 0.5
                       else "genuinely different geometry")
            print(f"  {name:<12} {c:>15.3f} {reading:>40}")

    print("\n" + "=" * 74)
    print("MECHANISM 2  exponential concentration")
    print("=" * 74)
    print("  Mean off-diagonal fidelity as qubits grow. Approaching zero means")
    print("  the kernel matrix approaches the identity: every case looks")
    print("  equally dissimilar to every other, so the model can only memorise.\n")
    print(f"  {'qubits':>7} {'angle':>10} {'zz':>10} {'reupload':>11} "
          f"{'rbf_median':>12}")
    ref_spread = mx.expressivity_spread(mx.normalise_kernel(Kc["rbf_median"]))
    for nq in (4, 6, 8, 10, 12, 14):
        row = []
        for name, fn in (("angle", lambda: qk.angle_states(X, nq)),
                         ("zz", lambda: qk.zz_states(X, nq, reps=2)),
                         ("reupload", lambda: qk.reupload_states(X, nq, layers=3))):
            K = qk.fidelity_kernel(fn())
            row.append(mx.expressivity_spread(K))
        print(f"  {nq:>7} {row[0]:>10.4f} {row[1]:>10.4f} {row[2]:>11.4f} "
              f"{ref_spread:>12.4f}")

    print("\n" + "=" * 74)
    print("CONSEQUENCE FOR SHOT COUNT")
    print("=" * 74)
    print("  A kernel entry of size s needs roughly 1/s^2 shots to resolve.")
    print("  Concentration therefore sets the measurement cost on hardware.\n")
    print(f"  {'qubits':>7} {'zz mean off-diag':>18} {'shots to resolve':>19}")
    for nq in (8, 12, 16):
        K = qk.fidelity_kernel(qk.zz_states(X[:40], nq, reps=2))
        s = mx.expressivity_spread(K)
        shots = int(1 / max(s, 1e-12) ** 2)
        print(f"  {nq:>7} {s:>18.5f} {shots:>19,d}")
    print("\n  At the point where the map is most expressive it is also most")
    print("  expensive to measure, and the two effects cancel.")


if __name__ == "__main__":
    main()
