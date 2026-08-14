# Golden Hour — Final Pipeline

```
STAGE 0  INTAKE                                              [CLASSICAL]
  Victim uploads: chat export, screenshots, bank SMS

STAGE 1  PREPROCESSING                                       [CLASSICAL]
  core/preprocess.py — parse exports, OCR screenshots, hash files,
  detect language, judge quality

STAGE 2  EXTRACTION                                          [CLASSICAL]
  core/extractor.py — Gemini reads messy multilingual text -> JSON
  core/rulebased_extract.py — regex on bank SMS, cross-checks the model

STAGE 3  VALIDATION                                          [CLASSICAL]
  core/validators.py — arithmetic, impossibility checks (I1-I8),
  tamper indicators (T1-T7), NCRP format constraints

STAGE 4  SEAL                                       [QUANTUM-MOTIVATED]
  core/evidence_seal.py — SHA-256 Merkle tree + Merkle-Lamport signature
  Exists because Shor's algorithm makes RSA/ECDSA seals retroactively
  forgeable; the fix must be applied at sealing time
  -> selective disclosure comes free with the Merkle structure

STAGE 5  NEAR-DUPLICATE FINGERPRINT                 [QUANTUM-INSPIRED]
  core/fingerprint.py — 256-dim complex state, amplitude from byte
  counts, phase from mean position of each byte value
  Scope: same-artifact-reappears detection only (a screenshot
  re-uploaded, a template resurfacing, an OCR'd/edited copy of a
  document already on file) — not general script-similarity linkage,
  which loses to TF-IDF and is not used for that job
  Runs on standard hardware: numpy complex arrays, no qubits required

STAGE 6  SCRIPT / OPERATION LINKAGE                           [CLASSICAL]
  TF-IDF cosine + exact-identifier overlap
  The "different victims, same operation" job

STAGE 7  DOSSIER GENERATION                                   [CLASSICAL]
  core/generators.py — NCRP / 1930 / Bank / NPCI outputs, sealed root,
  linked-case references
```

## What is quantum here, precisely

- **Stage 4** — quantum-*motivated*. A quantum threat (Shor) is why this
  layer exists at all; the layer itself is classical cryptography.
- **Stage 5** — quantum-*inspired*. A representation built from complex
  amplitude and phase, tested against its own classical baseline
  (histogram cosine, no phase) and measured to win: **+10.9 points mean
  retrieval accuracy, Wilcoxon p = 0.016 across 8 seeds.** Implemented and
  deployed entirely on classical hardware.
- **Everywhere else** — classical, because nine earlier probes measured
  quantum representations losing or tying there.

## Measured numbers behind Stage 4 and 5

| | metric | result |
|---|---|---|
| Stage 4 | seal / verify time | 0.97 ms / 0.64 ms |
| Stage 4 | signature security vs quantum adversary | 128-bit (Grover only; Shor does not apply) |
| Stage 5 | retrieval of corrupted original, 200-case archive, 10% corruption | 95% (fingerprint) vs 92.5% (histogram cosine) vs ~2% (SHA-256) |
| Stage 5 | gain over histogram-cosine baseline, 8 seeds | +10.9 pts, p = 0.016 |

## Reproduce

```bash
python -m tests.test_pipeline              # 51 tests
python -m quantum.probe_integrity           # Stage 4 numbers
python -m quantum.probe_qed_monolithic      # Stage 5 numbers
python -c "from core.fingerprint import similarity; \
           print(similarity('doc a text', 'doc a text noised'))"
```
