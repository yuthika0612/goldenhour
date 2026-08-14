#!/usr/bin/env python3
"""
PROBE 10 — the synthesis: do the two positives compose?

Positive 1 (probe 5): post-quantum evidence sealing. Hash-based Merkle seals.
Positive 2 (probe 9B): a monolithic 8-qubit whole-document encoding, 256
amplitudes carrying what a 5,000-feature TF-IDF model carries.

These do different jobs, and the difference is the point:

  CRYPTOGRAPHIC HASH   all-or-nothing. Change one byte and the digest is
                       unrelated. Perfect for integrity, useless for
                       similarity.
  QUANTUM STATE        similarity-preserving. Near-duplicate documents have
                       high fidelity. Useless for integrity, useful for
                       linkage.

If that holds, a two-layer evidence record follows: a hash layer that proves
nothing changed, and a state layer that finds related cases — and the state
can be shared for linkage WITHOUT sharing the evidence.

Three questions, and the third decides whether it is publishable or reckless:

  Q1  Is the state actually similarity-preserving where the hash is not?
  Q2  Does it survive the noise real evidence has (OCR errors, edits)?
  Q3  HOW MUCH DOES THE STATE LEAK? A fingerprint that reconstructs the
      document is not privacy-preserving, it is a copy in disguise.
"""
import sys, hashlib, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantum.probe_qed_monolithic import m1_hist, m3_hist_fourierpos
from quantum.probe_qed_workflow import make_corpus


def sha_sim(a: str, b: str) -> float:
    """Bit agreement between two SHA-256 digests. Baseline ~0.5 always."""
    ha = hashlib.sha256(a.encode()).digest()
    hb = hashlib.sha256(b.encode()).digest()
    bits = np.unpackbits(np.frombuffer(ha, np.uint8)) == \
           np.unpackbits(np.frombuffer(hb, np.uint8))
    return float(bits.mean())


def fidelity(a: str, b: str, enc=m1_hist) -> float:
    return float(abs(np.vdot(enc(a), enc(b))) ** 2)


def corrupt(doc, rate, rng):
    """OCR-style corruption: substitutions, deletions, insertions."""
    t = list(doc)
    n = max(1, int(len(t) * rate))
    for _ in range(n):
        op = rng.integers(3)
        i = rng.integers(max(1, len(t)))
        if op == 0 and t:
            t[i] = chr(rng.integers(32, 127))
        elif op == 1 and len(t) > 10:
            del t[i]
        else:
            t.insert(i, chr(rng.integers(32, 127)))
    return "".join(t)


def main():
    rng = np.random.default_rng(0)
    docs, y = make_corpus(200, seed=5)

    print("=" * 74)
    print("Q1  IS THE STATE SIMILARITY-PRESERVING WHERE THE HASH IS NOT?")
    print("=" * 74)
    print(f"  {'corruption':>11} {'SHA-256 agreement':>19} {'state fidelity':>16}")
    base = docs[0]
    for rate in (0.0, 0.01, 0.05, 0.10, 0.25, 0.50):
        s, f = [], []
        for _ in range(30):
            c = corrupt(base, rate, rng) if rate else base
            s.append(sha_sim(base, c))
            f.append(fidelity(base, c))
        print(f"  {rate:>10.0%} {np.mean(s):>19.4f} {np.mean(f):>16.4f}")
    unrelated = np.mean([fidelity(docs[0], d) for d in docs[1:60]])
    print(f"\n  unrelated documents: SHA {sha_sim(docs[0], docs[1]):.4f}   "
          f"state fidelity {unrelated:.4f}")
    print("""
  SHA agreement sits at chance (~0.5) the moment anything changes, and stays
  there. That is the design goal of a hash and it is why a hash cannot rank
  similarity. The state degrades smoothly, which is what linkage needs.
""")

    print("=" * 74)
    print("Q2  NEAR-DUPLICATE RETRIEVAL UNDER REALISTIC NOISE")
    print("=" * 74)
    print("  Task: given a corrupted copy, find its original among 200 cases.\n")
    print(f"  {'corruption':>11} {'top-1 by state':>16} {'top-1 by SHA':>14}")
    for rate in (0.02, 0.05, 0.10, 0.20, 0.35):
        hit_q = hit_h = 0
        trials = 60
        for t in range(trials):
            i = rng.integers(len(docs))
            q = corrupt(docs[i], rate, rng)
            fid = np.array([fidelity(q, d) for d in docs])
            hit_q += int(fid.argmax() == i)
            shs = np.array([sha_sim(q, d) for d in docs])
            hit_h += int(shs.argmax() == i)
        print(f"  {rate:>10.0%} {hit_q/trials:>16.1%} {hit_h/trials:>14.1%}")

    print("""
  A hash cannot do this task at all: it retrieves at chance because digest
  similarity carries no information about content similarity.
""")

    print("=" * 74)
    print("Q3  HOW MUCH DOES THE FINGERPRINT LEAK?")
    print("=" * 74)
    print("""  This decides whether 'share the state, not the evidence' is honest.
  The M1 encoding is a normalised byte histogram, so it reveals the byte
  composition exactly. That is a real disclosure.
""")
    d = docs[0]
    st = m1_hist(d)
    counts = (np.abs(st) ** 2) * len(d.encode())
    top = np.argsort(counts)[::-1][:12]
    print("  Recovered from the fingerprint alone (top byte frequencies):")
    shown = ", ".join(
        f"{repr(chr(b)) if 32 <= b < 127 else b}x{counts[b]:.0f}" for b in top)
    print(f"    {shown}")
    print(f"""
  So the state DOES leak: character frequencies, document length, and enough
  to run frequency analysis. It does NOT leak order, and it does not permit
  reconstruction of the text — but calling it privacy-preserving without
  qualification would be false.

  Honest positioning: the state is a SIMILARITY FINGERPRINT, not a private
  one. To share it across agencies safely it needs the same treatment any
  frequency-revealing sketch needs — a keyed permutation of the byte axis,
  or differential-privacy noise on the amplitudes, with the accuracy cost
  measured rather than assumed.
""")

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("""
  The two layers compose, and they do genuinely different jobs:

    hash / Merkle seal   proves nothing changed          exact, brittle
    quantum state        finds what is related           smooth, leaky

  Neither substitutes for the other, and that is the argument for having
  both in one record. The leakage in Q3 is a design constraint to state
  openly, not a flaw to hide: it is exactly the kind of thing a forensic
  tool must disclose before anyone deploys it.
""")


if __name__ == "__main__":
    main()
