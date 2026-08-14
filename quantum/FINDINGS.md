# Quantum vs classical representations on Golden Hour case features

**Result: quantum encodings did not help. The best quantum kernel scored
0.089 ± 0.029 AUC below the best classical kernel across six seeds, and never
won in any configuration tested. The geometric difference test explains why,
and it ruled the outcome out before any model was trained.**

This is a negative result, reported as such.

---

## What was actually tested

Not "quantum computing for fraud detection" in the abstract. A specific,
falsifiable question:

> Take the 14 case features the Golden Hour pipeline already computes.
> Encode them into quantum states. Does the resulting geometry separate
> fraudulent from legitimate episodes better than a classical kernel on the
> same features?

Everything is exact statevector simulation, so results reflect the
representation itself rather than hardware noise. Shot noise is modelled
separately.

| | |
|---|---|
| Features | 14, from `quantum/features.py` — transaction structure, timing, stage/tactic profile, findings |
| Encodings | angle (product state), ZZ feature map (entangling), data re-uploading |
| Classical baselines | linear, RBF at three bandwidths |
| Data | synthetic cases with deliberate class overlap and 6% label noise |
| Protocol | geometry first, then alignment, then accuracy — fixed in advance |

---

## The four measurements

### 1. Geometric difference — asked before training

Huang et al. (Nature Communications 12, 2631, 2021) give a test that settles
whether a quantum kernel *could* beat classical ones, for any labelling,
before any training happens. Advantage requires g ≫ √N.

| encoding | min g | closest classical kernel | verdict |
|---|---|---|---|
| angle | 3.39 | rbf_median | classical can match it for any labels |
| zz | 3.84 | rbf_narrow | classical can match it for any labels |
| reupload | 2.92 | rbf_narrow | classical can match it for any labels |

With N = 300, √N = 17.3. Every encoding lands around g ≈ 3, far below.
**A classical kernel can reproduce these quantum kernels on this data for any
labelling.** No accuracy number could have demonstrated advantage here.

*g is taken as the minimum over the classical family, not against one
conveniently weak baseline — advantage has to beat every classical option.*

### 2. Kernel–target alignment

| kernel | KTA | effective dim | mean off-diagonal |
|---|---|---|---|
| linear (C) | **0.335** | 12 | 19.89 |
| rbf_wide (C) | 0.334 | 39 | 0.77 |
| rbf_median (C) | 0.315 | 166 | 0.39 |
| angle (Q) | 0.096 | 83 | 0.33 |
| reupload (Q) | 0.090 | 293 | 0.009 |
| zz (Q) | 0.059 | 297 | **0.005** |

Classical kernels align with the labels three to five times better. The
quantum kernels have high effective dimension and near-zero off-diagonal
mass — expressive, but expressive in directions that have nothing to do with
whether the case is fraud.

### 3. Task performance (identical 5-fold splits)

| kernel | balanced acc | ROC-AUC |
|---|---|---|
| rbf_wide (C) | 0.875 ± 0.025 | **0.939 ± 0.040** |
| rbf_median (C) | 0.871 ± 0.027 | 0.937 ± 0.039 |
| linear (C) | 0.858 ± 0.029 | 0.932 ± 0.035 |
| reupload (Q) | 0.765 ± 0.034 | 0.858 ± 0.042 |
| angle (Q) | 0.770 ± 0.090 | 0.798 ± 0.076 |
| zz (Q) | 0.564 ± 0.036 | 0.612 ± 0.043 |

The entangling ZZ map — the one that is supposed to be classically hard —
performs worst, barely above chance.

### 4. Robustness sweep

The sweep deliberately searched for the regime most favourable to quantum.

- **Feature-map scale** (the quantum equivalent of tuning RBF γ, and often
  omitted in papers, which biases against quantum): tested 0.25–2.0. Best
  quantum result 0.900 vs classical 0.910.
- **Qubit count** 6→12: at 12 qubits, angle encoding reaches 0.910 — exactly
  matching classical, never exceeding it.
- **Scarce data**, where quantum kernels are most often claimed to help:
  the gap is *widest* there (−0.184 at N = 60) and narrows as data grows.
  The opposite of the usual claim.
- **Six seeds**: gap −0.089 ± 0.029. Quantum never won once.
- **Shot noise** at 1024 shots changed AUC by less than 0.02 — the
  representation is the problem, not measurement.

---

## Why it fails — two measured mechanisms

**1. When the quantum kernel works, it is not being quantum.**
Angle encoding is a product state, so its kernel is exactly
∏ᵩ cos²((xᵩ − zᵩ)/2) — a shift-invariant product kernel computable
classically in O(n). Its correlation with the classical RBF kernel rises from
0.67 at 8 qubits to **0.84 at 12 qubits**, precisely where it starts matching
classical performance. It performs well by *becoming* a classical kernel.

**2. When it is genuinely quantum, it concentrates.**
The ZZ map's geometry is genuinely different (correlation with RBF: 0.02).
But mean off-diagonal fidelity collapses geometrically with qubit count:

| qubits | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| zz | 0.076 | 0.019 | 0.005 | 0.0015 | 0.0006 | 0.0003 |
| angle | 0.524 | 0.366 | 0.288 | 0.184 | 0.125 | 0.086 |
| classical RBF | 0.384 | — | — | — | — | — |

The kernel matrix approaches the identity: every case looks equally
dissimilar to every other, so the model can only memorise. This is the known
exponential-concentration failure mode (Thanasilp et al., 2024).

**And the two mechanisms trap each other.** A kernel entry of size *s* needs
roughly 1/s² shots to resolve on hardware:

| qubits | zz mean off-diagonal | shots per kernel entry |
|---|---|---|
| 8 | 0.0049 | 42,000 |
| 12 | 0.00062 | 2,600,000 |
| 16 | 0.00023 | 18,700,000 |

At the point where the map is most expressive it is also most expensive to
measure. Expressivity and measurability cancel.

---

## Honest scope

**What this shows:** for 14 engineered case features with this structure,
quantum kernel encodings offer no representational benefit, and the
geometric-difference test predicts that *before* training.

**What it does not show:** that quantum methods are useless for financial
crime generally. Specifically not tested — raw high-dimensional transaction
graphs, sequence structure rather than aggregate features, quantum
optimisation formulations (a different question entirely from kernels), or
datasets with genuinely quantum-structured correlations.

**The data is synthetic.** No labelled corpus of Indian fraud evidence
bundles exists publicly — that gap is what the project proposes to fill. A
representational question can be asked of synthetic data because it concerns
the encoding, not the realism. Detection performance cannot, and none of
these numbers should be reported as such.

---

## What to do with this

The correct project decision is to **leave quantum out of the pipeline** and
say so with evidence. That is a stronger position than a hand-waving "future
scope: quantum" line, because it is a measured claim with a named test, a
control that passed, and a mechanism.

It is also publishable in its own right: applying the geometric-difference
test *before* training, with a classically-simulable control encoding to
validate the metric, is a methodology most applied quantum-ML papers skip.
The finding that the best-performing quantum kernel earns its performance by
converging to a classical kernel is worth stating plainly.

### One methodological note worth keeping

The first version of this experiment reported g ≈ 130 for angle encoding —
apparent strong advantage. It was wrong. The linear kernel used as reference
was rank-deficient (effective dimension 12 of 300), so inverting it inflated
g for every quantum kernel. The fix was to normalise kernels to unit diagonal
and take the minimum over the classical family.

The tell was the control: angle encoding is a product state, classically
simulable in O(n), so it *cannot* show advantage. When a metric says
otherwise, the metric is broken. That control is now built into the
experiment and prints a warning if it ever fires again.

---

## Reproduce

```bash
python -m quantum.experiment      # main run, all five steps
python -m quantum.sweep           # scales, qubits, data sizes, seeds
python -m quantum.diagnostics     # the two failure mechanisms
```
