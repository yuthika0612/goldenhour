# Where can quantum actually help this project?

Nine entry points tested, the last three chosen from the QML literature's own
account of where advantage should live and from the prior HDQS/QED encoding
work. **Six analytics entry points fail.
One survives — and it is not an analytics one.**

| # | Entry point | Classical falls short when… | Fraud cases are… | Verdict |
|---|---|---|---|---|
| 1 | Kernel learning | never, for this feature geometry | not in that regime | ✗ |
| 2 | QUBO optimisation | 10⁴+ variables, rugged landscape | tens of accounts | ✗ |
| 3 | Subgraph matching | fuzzy structural matching at scale | exact identifier strings | ✗ |
| 4 | Generative sampling | high-entropy, non-factorisable target | sparse, low-entropy | ✗ |
| 6 | **Hybrid** quantum+classical | components make complementary errors | quantum errs where classical already errs | ✗ |
| 7 | **Geometric QML**, pairwise linkage | correlations need a classically hard transform | correlations are local or symmetry-capturable | ✗ (replicates, then dequantises) |
| 8 | **Phase / superposition encoding** | order and ambiguity are classically expensive | order is cheap; ambiguity graph is a tree | ✗ (but fixed a real defect) |
| 9 | **QED workflow**, raw bytes → statevectors | — | chunking was my implementation flaw, not QED's | ✗ (superseded by 9B) |
| 9B | **Monolithic QED**, 8–10 qubits, no chunking | — | phase *hurts* once alignment is fixed | ✗ (amplitude-only ties classical) |
| 5 | **Evidence integrity** | **the adversary has a quantum computer and the archive must survive decades** | **exactly that situation** | **✓** |

---

## The four negatives, briefly

### 1. Kernel learning — measured in `FINDINGS.md`
Best quantum kernel scored 0.089 ± 0.029 AUC *below* classical across six
seeds. Geometric difference g ≈ 3 against a threshold of √N = 17.3, so a
classical kernel can reproduce these quantum kernels for *any* labelling.
Two mechanisms: the product encoding is a classical kernel in disguise
(correlation with RBF rises to 0.84 at 12 qubits, exactly where it starts
performing), and the entangling map concentrates exponentially (mean
off-diagonal fidelity 0.076 → 0.0003 over 4→14 qubits).

### 2. QUBO optimisation — freeze-target selection
Choosing which mule accounts to freeze is genuinely NP-hard and maps
naturally to QUBO, the native input of an annealer.

| accounts | choose | exact | simulated annealing | SA / exact |
|---|---|---|---|---|
| 12 | 4 | 0.005 s | 0.32 s | 1.0000 |
| 16 | 5 | 0.036 s | 0.32 s | 1.0000 |
| 20 | 6 | 0.30 s | 0.33 s | 1.0000 |
| 24 | 7 | 2.0 s | 0.34 s | 1.0000 |

Classical simulated annealing hits the exact optimum on every instance small
enough to verify, and scales to 1,000 accounts in under a second. A real mule
network in one case has tens of accounts. **NP-hard in theory, trivial at the
size fraud actually produces.**

### 3. Case linkage — subgraph matching
Formally subgraph isomorphism. In practice investigators link on *exact*
identifiers: a UPI handle shared verbatim across complaints.

| cases | links found | time |
|---|---|---|
| 1,000 | 55 | 0.002 s |
| 10,000 | 6,161 | 0.030 s |
| 100,000 | 627,192 | 0.224 s |

An inverted index — a hash join, linear in cases. The NP-hard version only
appears if you insist on fuzzy structural matching, which is not the task.

### 4. Generative sampling — the data-scarcity bottleneck
The project's real constraint is that no labelled corpus of Indian fraud
evidence exists, so quantum Born machines looked promising.

| distribution | entropy (bits) | support | max prob |
|---|---|---|---|
| Born machine output | 8.79 | 1020 / 1024 | 0.030 |
| real case features | **5.59** | **135 / 1024** | 0.093 |
| uniform | 10.00 | 1024 | 0.001 |

Real cases occupy 13% of the state space: a handful of recurring scam shapes.
Born machines pay off on high-entropy, hard-to-factorise distributions. This
is the opposite kind of target.

### 6. Hybrid quantum–classical — the strongest remaining case
A component can be weak alone and still add value in combination, if its
errors differ. So this was tested separately rather than inferred, with four
fusion strategies (kernel mixing, feature fusion, decision ensemble, residual
boosting) and — critically — four controls.

**The precondition fails first.** Error-complementarity analysis: the quantum
model uniquely fixes 11 cases the classical model misses, while introducing
44 new errors. Net −33. There is no error budget for fusion to exploit.

| fusion partner | AUC | Δ vs classical | Holm-adj p |
|---|---|---|---|
| quantum kernel (nested α) | 0.9177 | −0.0006 | 0.47 |
| C1 random PSD kernel | 0.9168 | −0.0015 | 0.11 |
| C2 shuffled quantum | 0.9149 | −0.0034 | 0.010 |
| C3 second classical kernel | 0.9162 | −0.0021 | 0.014 |

Decision-level ensemble: −0.0160 (p = 0.0002, significantly *worse*). Nested
selection chose a mean mixing weight of 0.23 — the procedure discards the
quantum kernel when it cannot see the test fold.

**One apparent positive appeared and was interrogated.** Classical features
plus 8 quantum kPCA components scored +0.0055, p = 0.0316 — exactly the shape
of result that gets published as hybrid quantum advantage. It failed all three
follow-up checks:

1. **Multiple comparisons.** Three component counts were tried; Holm
   correction across them → adjusted p = 1.00.
2. **Seed stability.** Across 8 independent dataset draws: −0.0009 ± 0.0038,
   winning 2 of 8. Differs from zero at p = 0.38.
3. **Random-feature control.** Adding 8 *random* projections gave −0.0009 ±
   0.0018 — statistically indistinguishable (p = 0.95). Any movement is
   dimensionality and regularisation, not quantum information.

The most instructive line in the whole study: **adding pure noise columns
scored +0.0039 with p = 0.0145** — a "more significant" improvement than the
quantum features achieved. That is what an uncontrolled hybrid experiment
would have reported as a finding.

### 7. Geometric QML for case linkage — the literature's own best shot
The QML field's honest position (Xanadu 2026; Huang et al. Science 2022) is
that advantage lives in **structure classical methods cannot cheaply fake**:
quantum data, deep algebraic structure, or an inductive bias built from
symmetry. Probes 1–6 used aggregate features with no symmetry — arguably the
worst possible setting. So probe 7 tested the setting the literature actually
points to.

The template is Umeano, Scali & Kyriienko (*Phys. Rev. A* 113, 052425, 2026):
geometric QML for **similarity testing**, where symmetry-aware measurement
adaptation generalised from 3 samples per class on 20-qubit inputs while
Siamese DNNs and CNNs sat near chance. Our project has an exact structural
match — **case linkage**: *did these two complaints come from the same
operation?* Pairwise, Z₂-symmetric, few-shot, plausibly driven by global
rather than local correlation.

Three data regimes, five models. **R3 is a positive control** — the Raz–Tal
forrelated construction from the paper. Without it, no negative result would
be interpretable.

| regime | train pairs | GQML-M | Siamese | Classical **+ symmetry** | **Dequantised** |
|---|---|---|---|---|---|
| R3 forrelated *(control)* | 10 | **0.999** | 0.521 | 0.520 | **1.000** |
| R2 adversarial | 6 | 0.705 | 0.500 | **0.915** | 0.845 |
| R2 adversarial | 50 | 0.932 | 0.557 | 0.931 | 0.932 |
| R1 ordinary fraud | 6 | 0.950 | 0.879 | **1.000** | 0.955 |

**The control replicates the published advantage exactly.** GQML reaches
0.999 AUC on 10 training pairs where a Siamese network manages 0.52. That is
a real, large, reproducible effect — the literature result is not wrong.

**And it dequantises completely.** The "Dequantised" column is a twenty-line
classical function: compute the same forrelation feature |⟨φ₁|H|φ₂⟩| with a
fast Walsh–Hadamard transform, feed it to logistic regression. It scores
**1.000** — matching or beating the quantum model on its own home turf, on a
laptop, with no Hilbert space anywhere. The barcode paper says this itself:
for classical data held in memory the advantage dequantises, because the
Hadamard transform is cheap classically and 2-fold forrelation is classically
simulable.

**On fraud-like data the quantum model loses outright.** In the adversarial
regime — where the operation randomises handles, amounts and phrasing while
keeping the script skeleton, which is the case actually worth catching — the
symmetry-aware classical model reaches 0.915 from **6 training pairs** while
GQML-M manages 0.705.

**The methodological finding, which generalises beyond this project:** the
quantum advantage in similarity testing is an advantage over *the wrong
baseline*. Siamese networks cannot express a global transform, so they fail;
that failure is architectural, not classical-in-general. Give a classical
model the same inductive bias — symmetry invariance plus the transform — and
it matches or beats the quantum model everywhere, including on the
construction designed to favour quantum.

The one condition under which this would flip: correlations requiring a
transform that is genuinely classically hard, such as 4-fold forrelation.
Fraud evidence has no such structure, and nothing in criminology suggests it
should.

### 8. Encoding rather than learning — and the defect it exposed
Probes 1–7 all asked the same question in different costumes: given a feature
vector, does a quantum *model* learn better? Probe 8 asked instead whether the
*representation* was wrong. Two ideas from prior HDQS/VFX work were never
tested: **phase as a free dimension** (amplitude carries one layer, phase
another, at zero storage cost — two records with identical value histograms
are identical classically but distinguishable by phase) and **superposition of
ambiguous readings** (hold every plausible OCR reading at once instead of
committing early).

**The critique landed.** Built two fraud operations with an *identical* stage
multiset and an *identical* amount multiset, differing only in order —
threaten-then-demand versus demand-then-threaten:

| encoding | AUC | dims |
|---|---|---|
| amplitude only (aggregate) — what probes 1–7 used | **0.473** (chance) | 14 |
| amplitude + phase (HDQS style) | 0.935 | 14 |
| classical order-aware [control] | **1.000** | 70 |

The aggregate features used throughout this study are **structurally blind to
order**. Phase encoding recovers that information in the *same* 14 dimensions,
which is a genuine compression result. But the compression does not buy
anything: classical bigram features reach 1.000 at every training size tested,
including 12 samples, so even the few-shot hypothesis fails.

**On superposition of ambiguity:** enumeration over k^n readings dies past ~10⁶
branches, and the amplitude form is exact and instant — but for *separable*
questions it is a closed-form factorisation any statistician would write, with
no qubits involved. For *constrained* readings (a screenshot reference must
match the SMS), the evidence graph in a fraud case is a **chain or tree**, where
belief propagation is exact and linear: 0.12 ms versus 7 s for enumeration at
n = 12, with 0.00000% error. The exponential branch count is an illusion created
by writing the problem down badly.

**What this probe actually produced.** Not a quantum win — a bug fix. The
project's own `features.py` was order-blind, and four order-aware features were
added as a direct result. Quantum-inspired representational thinking found a
real defect in the classical pipeline. That is a legitimate and reportable
outcome, and it is the honest version of "quantum-inspired": the idea helped,
the mechanism was not needed.

### 9. The QED workflow — raw bytes to statevectors
The original SIA workflow, tested faithfully for the first time: no hand-built
features anywhere. Take the raw bytes of an evidence file, chunk into 32-byte
blocks, encode each as a normalised complex statevector with byte-derived
phases, classify with a fidelity kernel.

The comparison design is what makes this readable. "Classical model on
classical data vs quantum model on quantum data" is confounded — the encoding
is itself a transformation. So three pipelines got **the same raw bytes**:

- **P1** classical on raw bytes — char n-gram TF-IDF
- **P2** classical on the *same chunks* — identical chunking and
  normalisation, real vectors, cosine kernel ← **the decisive control**
- **P3** QED — complex statevectors, byte-derived phases, fidelity kernel

On an easy task all three saturate at 1.000. The hard regime — fraud text
leetspeaked and space-broken to evade filters, only a short excerpt available
— separates them:

| excerpt | train | P1 raw bytes | P2 same chunks | **P3 QED** |
|---|---|---|---|---|
| 120 | 20 | 0.999 | 0.512 | **0.616** |
| 120 | 60 | 1.000 | 0.541 | **0.705** |
| 256 | 20 | 1.000 | 0.575 | **0.642** |
| 256 | 60 | 1.000 | 0.609 | **0.722** |

**Three findings, and the first one is a genuine positive.**

1. **The QED phase encoding beats its own matched control by +0.10 to +0.16
   AUC, consistently and outside noise.** P2 has identical chunking and
   normalisation and differs *only* in lacking phases and the fidelity kernel.
   So the byte-derived phase structure carries real, separable information —
   the QED design claim holds under a properly matched control. This is the
   first positive result in nine probes on the analytics side.

2. **But it is recovering what its own chunking destroyed.** P2 sits at 0.51 —
   chance. Fixed 32-byte block alignment annihilates the signal in obfuscated
   text, because inserting one space shifts every subsequent block. The phase
   encoding claws back part of that loss; it does not exceed the starting
   point.

3. **A plain char n-gram model scores 1.000.** Not because it is quantum or
   classical, but because n-grams *slide* and chunks do not. The bottleneck
   was never the Hilbert space — it was alignment.

**This conclusion was wrong, and probe 9B corrected it.** Chunking was an
implementation choice I introduced, not part of the QED idea. A monolithic
encoding removes the alignment problem entirely, and when it does, the
apparent phase advantage disappears — see below.

**Cost, which stands regardless:** the fidelity kernel is O(M²) in cases and
each entry needs ~10⁴ shots on hardware — 10,000 cases is 5×10¹¹ measurements.
TF-IDF is linear and runs in milliseconds.

### 9B. Monolithic QED — the correct implementation
No chunking. The whole document accumulates into ONE statevector of 2ⁿ
amplitudes, 8–10 qubits. No block boundaries, so nothing can misalign.

Four monolithic encodings, all alignment-free, hard regime throughout
(filter-evading obfuscation + short excerpts):

| train | C1 TF-IDF | M1 hist *(no phase)* | M2 +mean-pos phase | M3 +fourier phase | M4 bigram 10q | C2 magnitudes *(classical)* | C3 mag+phase *(classical)* |
|---|---|---|---|---|---|---|---|
| 20 | 0.993 | **0.979** | 0.821 | 0.851 | 0.970 | 0.965 | 0.925 |
| 60 | 1.000 | 0.993 | 0.949 | 0.977 | 1.000 | 0.994 | 0.980 |
| 120 | 1.000 | 0.997 | 0.988 | 0.996 | 1.000 | 0.997 | 0.995 |

*(excerpt = 120 bytes; at 256 bytes everything saturates at 1.000)*

**Three findings, and the first two overturn probe 9.**

1. **Monolithic beats chunked by a mile: 0.61 → 0.98.** The chunking in probe
   9 was self-inflicted damage. An 8-qubit whole-document encoding is
   competitive with TF-IDF straight away.

2. **Phase actively HURTS once alignment is fixed: M1 0.979 → M2 0.821, M3
   0.851.** M1 and M2 have *identical amplitudes* and differ only in carrying
   phase, so this is a clean measurement. The reason is mechanical: the
   fidelity kernel |⟨ψ|φ⟩|² suppresses overlap between documents whose phases
   disagree, and phases from independently written documents do not align. On
   this task phase injects interference noise, not signal.

   So probe 9's positive was **phase repairing damage that phase-free
   chunking had caused**. Remove the damage and the benefit evaporates. This
   is exactly the failure mode this study keeps finding, and it caught itself
   this time.

3. **The quantum kernel ties its classical twin.** M1 (fidelity kernel on
   256 amplitudes) scores 0.979; C2 (a plain classical model on the *same*
   256 magnitudes) scores 0.965; C3 (classical model on magnitudes *and*
   phases as 512 real features) scores 0.925. Within noise of each other, and
   all within noise of a one-line TF-IDF baseline at 0.993.

**What this does establish, and it is worth keeping:** a monolithic 8-qubit
whole-document encoding is a *viable, compact* representation — 256 amplitudes
matching a 5,000-feature TF-IDF model. As compression that is a real property.
It is not an accuracy advantage, and the phase layer needs a task where phases
are *constructed to align* (as in the forrelation control of probe 7) before it
contributes.

### Why all seven fail together
The fraud problem is **small, structured and identifier-driven. Its
difficulty is evidential, not computational.** Nothing in it is compute-bound,
so no faster computer of any kind — quantum or classical — addresses the
bottleneck. The bottleneck is that the evidence is messy, incomplete and
sometimes fabricated.

---

## The one that survives: evidence integrity

Digital evidence must remain **verifiable** for as long as a case can be
reopened. Indian criminal appeals routinely run ten to twenty years, and
archives outlive the prosecutions that created them.

The quantum threat here runs *backwards* compared with the usual story:

- **Confidentiality attacks** ("harvest now, decrypt later") threaten secrecy.
- **Integrity attacks** threaten authenticity *retroactively*. Evidence sealed
  today with RSA or ECDSA becomes forgeable once Shor's algorithm can recover
  the private key from the public key. At that moment every signature that key
  ever made becomes repudiable — **including seals applied decades earlier.**
  A defence can then argue the archive could have been altered, and be
  technically correct.

**And the fix cannot be applied later.** An archive sealed with ECDSA in 2026
cannot be retroactively re-secured in 2040, because the seals will already be
worthless. Re-sealing in 2040 proves only that the file existed in 2040 — it
says nothing about 2026. The countermeasure has to be applied at sealing time
or not at all.

### What was built: `core/evidence_seal.py`

SHA-256 Merkle tree over the evidence bundle, signed with a Merkle–Lamport
(XMSS-style) hash-based signature. Security rests only on hash-function
properties. Shor does not apply; Grover halves the margin, which SHA-256
absorbs.

| scheme | classical | vs quantum | attack |
|---|---|---|---|
| RSA-2048 | 112 bit | **broken** | Shor |
| ECDSA P-256 | 128 bit | **broken** | Shor |
| SHA-256 Merkle+Lamport | 256 bit | 128 bit | Grover only |
| SHA-512 variant | 512 bit | 256 bit | Grover only |

Measured on a real Golden Hour bundle: **seal 0.97 ms, verify 0.64 ms**, one
published root signs 16 cases before key rotation. Altering `50,000` to
`5,000` in one SMS changes the root completely. A forged signature is
rejected. All verified by 18 tests.

### The part that pays for itself immediately

The Merkle structure gives **selective disclosure**, and this benefit exists
whether or not a quantum computer is ever built:

> A victim hands the bank the *one* payment screenshot it needs, with
> cryptographic proof it was part of the sealed case — while withholding the
> chat history, the personal note, and everything else. The bank learns
> nothing about the withheld items except how many there are.

Proof size grows logarithmically: a 65,536-item archive proves any single
item with 16 hashes — 512 bytes.

| items | tree build | proof length | proof size |
|---|---|---|---|
| 256 | 0.85 ms | 8 hashes | 256 B |
| 4,096 | 11 ms | 12 hashes | 384 B |
| 65,536 | 183 ms | 16 hashes | 512 B |

---

## The honest framing for the paper

The tempting sentence is *"we applied quantum computing to digital
forensics."* The defensible one is:

> We tested four analytics entry points for quantum computing against a real
> forensic pipeline and found no benefit in any of them, with a
> geometric-difference argument showing why kernel methods could not have
> helped regardless of labelling. We then identified the one place where
> quantum considerations do change the design — long-lived evidence integrity
> — and implemented a post-quantum sealing layer that costs a millisecond per
> case and delivers a privacy benefit today.

That is a stronger claim than an advantage result, because it is *falsifiable
and tested*, and it changed what got built. It also produces the rarer thing:
a project where the negative results are the contribution and the positive
result is deployable now.

### Two caveats to state plainly

1. **The negatives are scoped to this problem.** Fraud-case features are
   14-dimensional and aggregate. Nothing here refutes quantum methods on raw
   high-dimensional transaction graphs, on sequence structure, or on datasets
   with genuinely quantum-structured correlations. Those were not tested.
   The hybrid negative is likewise scoped: fusion fails here because the
   components' errors overlap, not because fusion cannot work in principle.
   Probe 7's negative is scoped too: GQML dequantises here because the
   relevant transform is cheap classically. Data whose correlations need a
   classically hard transform would give a different answer — that data just
   is not fraud evidence.
2. **The synthetic-data limit still applies** to probes 1–4. They measure
   representations and problem sizes, not detection performance on real fraud.
   Probe 5 is unaffected: cryptographic properties do not depend on data
   realism.

---

## Reproduce

```bash
python -m quantum.experiment          # kernel learning (probe 1)
python -m quantum.probe_hybrid        # hybrid fusion + controls (probe 6)
python -m quantum.probe_hybrid_followup  # interrogating the one positive
python -m quantum.probe_gqml_linkage  # geometric QML, similarity testing (probe 7)
python -m quantum.probe_encoding      # phase + superposition encoding (probe 8)
python -m quantum.probe_qed_workflow  # raw bytes -> statevectors, chunked (probe 9)
python -m quantum.probe_qed_monolithic # monolithic, no chunking (probe 9B)
python -m quantum.sweep               # robustness of probe 1
python -m quantum.diagnostics         # why probe 1 fails
python -m quantum.where_quantum_helps # probes 2, 3, 4
python -m quantum.probe_integrity     # probe 5, the one that survives
python -m tests.test_pipeline         # 51 tests, incl. 18 for sealing
```
