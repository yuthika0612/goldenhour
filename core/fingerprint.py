"""
Stage 5 — near-duplicate evidence fingerprint.

Scope: detects the SAME artifact reappearing across cases (a screenshot
re-uploaded, a template message copy-pasted, an OCR'd or edited copy of a
document already on file). Not used for general script-similarity linkage
across differently-worded complaints — that job is TF-IDF / exact-identifier
matching (see core/generators.py linkage helpers), which measurably wins it.

Encoding: whole-document byte histogram with mean-position phase per byte
value. A classically-implementable representation structured around complex
amplitude/phase, run with standard numpy arrays -- no qubits, no quantum
hardware required to compute or deploy.

Measured (quantum/probe_synthesis.py, quantum/probe_qed_monolithic.py):
  - retrieval of a corrupted document's original among 200 cases:
      95-98% top-1 at 2-10% corruption
  - a plain histogram-cosine baseline on the same 256-dim vector, no phase:
      70-92% at the same corruption levels (8 seeds, mean gap +10.9 pts,
      Wilcoxon p = 0.016)
  - SHA-256 digest similarity on the same corrupted input: 0-2% (chance),
    because a hash is designed to destroy similarity structure
"""
import numpy as np


def fingerprint(text: str) -> np.ndarray:
    """256-dim complex state: amplitude from byte counts, phase from mean
    position of each byte value in the document."""
    b = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8)
    L = max(len(b), 1)
    cnt = np.bincount(b, minlength=256).astype(float)
    pos_sum = np.bincount(b, weights=np.arange(len(b)), minlength=256)
    mean_pos = np.divide(pos_sum, cnt, out=np.zeros(256), where=cnt > 0)
    amp = np.sqrt(cnt)
    psi = amp * np.exp(2j * np.pi * mean_pos / L)
    n = np.linalg.norm(psi)
    return psi / n if n else psi


def similarity(a: str, b: str) -> float:
    """Fidelity between two fingerprints, |<psi_a|psi_b>|^2, in [0, 1]."""
    fa, fb = fingerprint(a), fingerprint(b)
    return float(abs(np.vdot(fa, fb)) ** 2)


def nearest(query: str, archive: dict, top_k: int = 3):
    """archive: {case_id: text}. Returns top_k (case_id, similarity) pairs."""
    scores = [(cid, similarity(query, txt)) for cid, txt in archive.items()]
    return sorted(scores, key=lambda x: -x[1])[:top_k]
