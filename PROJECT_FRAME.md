# Quantum-Era Evidence Records
### Post-quantum integrity and quantum-state similarity for digital financial fraud

**Minor project framing.** Quantum is the subject. Cryptography is the
mechanism. Digital forensics is the application domain.

Everything below is built from results that survived their own controls.
Nothing here rests on a claim that failed a check.

---

## 1. The one-sentence thesis

> A digital evidence record needs two things that no single primitive
> provides: proof that **nothing changed**, and the ability to find what is
> **related**. We build both, one from post-quantum cryptography and one from
> a quantum state representation, and show they are complementary rather
> than redundant.

---

## 2. Why quantum, stated honestly

Two independent quantum motivations, neither of them a speed claim:

**Shor's algorithm makes today's evidence seals forgeable tomorrow.** Evidence
must stay verifiable for as long as a case can be reopened — Indian criminal
appeals run ten to twenty years. A record sealed with RSA or ECDSA becomes
repudiable the moment the private key is recoverable, *retroactively*, for
every signature that key ever made. The fix must be applied at sealing time;
re-sealing in 2040 proves only that the file existed in 2040. This is the rare
case where a quantum consideration changes what you must build **now**.

**Quantum state representation is similarity-preserving where hashing is
not.** A cryptographic hash is designed to destroy all similarity structure:
change one byte and the digest is unrelated. That is exactly right for
integrity and exactly wrong for linkage. A normalised state encoding degrades
smoothly, so fidelity between states tracks similarity between documents.

---

## 3. The two layers

### Layer 1 — Integrity: post-quantum evidence seal

SHA-256 Merkle tree over the evidence bundle, signed with a Merkle–Lamport
(XMSS-style) hash-based signature. Security rests on hash-function properties
only: Shor does not apply, Grover halves the margin, SHA-256 absorbs it.

| scheme | classical | vs quantum |
|---|---|---|
| RSA-2048 / ECDSA P-256 | 112–128 bit | **broken (Shor)** |
| SHA-256 Merkle+Lamport | 256 bit | 128 bit (Grover only) |

**Measured:** seal 0.97 ms, verify 0.64 ms. One published root signs 16 cases
before key rotation. Altering `50,000` → `5,000` in one SMS changes the root
completely; forged signatures rejected. **18 tests.**

**Bonus capability that pays for itself immediately:** Merkle inclusion proofs
give **selective disclosure** — a victim proves one payment screenshot belonged
to the sealed case while withholding the chat history. A 65,536-item archive
proves any single item with 16 hashes (512 bytes). This benefit exists whether
or not a quantum computer is ever built.

### Layer 2 — Similarity: monolithic quantum state fingerprint

The whole evidence document accumulates into **one statevector, 8–10 qubits,
256–1024 amplitudes**. No chunking, therefore no alignment to break.

**Measured — representation density:** 256 amplitudes match a 5,000-feature
TF-IDF model on fraud/legitimate classification (0.979 vs 0.993 AUC at 20
training samples, hard obfuscated regime).

**Measured — similarity preservation:** given a corrupted copy, retrieve its
original from 200 cases.

| corruption | top-1 by **state** | top-1 by SHA-256 |
|---|---|---|
| 2% | **98.3%** | 1.7% |
| 5% | **95.0%** | 0.0% |
| 10% | **95.0%** | 1.7% |
| 20% | 61.7% | 0.0% |

The hash performs at chance because digest similarity carries no information
about content similarity. **This is the cleanest positive in the study.**

---

## 4. Why the two layers compose

| | proves nothing changed | finds what is related |
|---|---|---|
| **Hash / Merkle seal** | ✓ exact, brittle by design | ✗ chance |
| **Quantum state** | ✗ approximate | ✓ smooth, 95% @ 10% noise |

Neither substitutes for the other. That is the argument for carrying both in
one evidence record — and it is a design claim, not a performance claim, which
makes it defensible.

---

## 5. Stated limitations (in the paper, not hidden)

**The fingerprint leaks.** A byte-histogram state reveals character
frequencies and document length exactly — enough for frequency analysis. It
does *not* reveal order and does not permit reconstruction. Calling it
"privacy-preserving" without qualification would be false. Safe cross-agency
sharing needs a keyed permutation of the byte axis or noise on the amplitudes,
with the accuracy cost measured rather than assumed. **This is future work,
stated as an open problem.**

**No quantum speed advantage is claimed anywhere.** Nine analytics entry
points were tested and none produced one. The negative results are reported in
the companion document and are part of the contribution, not an embarrassment
to be omitted.

**The fidelity kernel is O(M²) in cases.** For linkage at national scale it
needs an approximate-nearest-neighbour index over the amplitude vectors, which
is standard and classical.

---

## 6. Application: the Golden Hour tool

The evidence record is not an abstraction — it sits inside a working system
that converts a fraud victim's raw evidence into a filing-ready complaint
package (NCRP fields, 1930 call brief, bank letter, NPCI package) within the
window where account freezing still works.

Layer 1 seals the bundle at intake. Layer 2 makes the sealed record findable:
*this complaint's evidence resembles complaints #14 and #38* — the cross-case
linkage that currently does not happen because every complaint is treated in
isolation.

---

## 7. What the panel will ask, and the answer

**"Where is the quantum?"** In the threat model that forces layer 1, and in
the state representation that is layer 2. Neither is decoration: remove
quantum computing from the world and layer 1 is unnecessary, while layer 2
would be an ordinary sketching algorithm rather than a state encoding.

**"Did you show quantum advantage?"** No, and we tested for it nine ways
before concluding that. The advantage claim would not have survived the
controls; the design claim does.

**"Did you compare against a proper classical baseline?"** Yes — including
running the *dequantised* version of a published quantum advantage and
matching it with twenty lines of classical code. That methodology is part of
the contribution.

---

## 8. Deliverables

1. Working evidence-sealing module (`core/evidence_seal.py`), 18 tests
2. Monolithic state fingerprint encoder + retrieval benchmark
3. The Golden Hour dossier tool, 51 tests
4. Companion negative-results document: nine entry points, each with a
   measured mechanism
5. Reproducible probe suite

---

*Reproduce the two positive results:*
```bash
python -m quantum.probe_integrity      # layer 1
python -m quantum.probe_qed_monolithic # layer 2 representation density
python -m quantum.probe_synthesis      # layer 2 similarity + leakage
```
