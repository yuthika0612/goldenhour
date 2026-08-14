#!/usr/bin/env python3
"""
PROBE 4 — the integrity layer.

Three analytics entry points came back negative, plus kernel learning. This
probe tests the remaining one, and it is the only place where a quantum
consideration changes what the project should build.

Run:  python -m quantum.probe_integrity
"""
import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import preprocess
from core.evidence_seal import (
    MerkleTree, MerkleSignatureScheme, seal_bundle, verify_seal,
    selective_disclosure, verify_disclosure, HASH,
    CLASSICAL_BITS, QUANTUM_BITS,
)


def sample_items():
    return preprocess.ingest(
        chat=("[14/01/2025, 10:16] +91 88XXX 11223: Transfer Rs 50,000 to "
              "verify.rbi2025@okaxis\n"
              "[14/01/2025, 10:22] Me: sent sir T2501140900123456"),
        sms=("SBI: Rs 50,000 debited from A/c XX4521 on 14-Jan-25. "
             "UPI Ref T2501140900123456. Avl Bal Rs 12,340."),
        ocr_texts=["Payment Successful  Rs 50,000  UTR T2501140900123456"],
        note="He said I would be arrested if I told anyone.")


def main():
    print("=" * 76)
    print("PROBE 4  THE INTEGRITY LAYER")
    print("=" * 76)
    print("""
  Digital evidence must stay VERIFIABLE for as long as a case can be
  reopened. Indian criminal appeals routinely run 10-20 years.

  The quantum threat to integrity runs backwards compared with the usual
  confidentiality story:

    RSA / ECDSA seals    Shor recovers the private key from the public key.
                         Every signature that key ever made becomes forgeable
                         and therefore repudiable, INCLUDING seals applied
                         decades earlier. A defence can then argue the archive
                         could have been altered, and be technically right.

    Hash-based seals     Shor does not apply. Grover halves the effective
                         bit-security, which SHA-256 absorbs.

  Nothing here needs a quantum computer to deploy. It needs one to break.
""")

    print("=" * 76)
    print("SECURITY MARGINS")
    print("=" * 76)
    print(f"  {'scheme':<26} {'classical':>11} {'vs quantum':>12} {'attack':>22}")
    rows = [
        ("RSA-2048", "112 bit", "broken", "Shor"),
        ("ECDSA P-256", "128 bit", "broken", "Shor"),
        ("SHA-256 Merkle+Lamport", f"{CLASSICAL_BITS} bit",
         f"{QUANTUM_BITS} bit", "Grover only"),
        ("SHA-512 variant", "512 bit", "256 bit", "Grover only"),
    ]
    for r in rows:
        print(f"  {r[0]:<26} {r[1]:>11} {r[2]:>12} {r[3]:>22}")

    # ------------------------------------------------------------ sealing
    print("\n" + "=" * 76)
    print("SEALING A REAL BUNDLE")
    print("=" * 76)
    items = sample_items()
    signer = MerkleSignatureScheme(height=4)
    print(f"  Public root (publish once, e.g. in the FIR): "
          f"{signer.public_root[:32]}...")
    print(f"  One key tree signs {signer.n_keys} cases before rotation.\n")

    t0 = time.perf_counter()
    bundle = seal_bundle(items, case_id="NCRP/2025/000123", signer=signer)
    t_seal = time.perf_counter() - t0

    t0 = time.perf_counter()
    v = verify_seal(bundle)
    t_ver = time.perf_counter() - t0

    print(f"  items sealed        : {bundle.n_items}")
    print(f"  merkle root         : {bundle.merkle_root[:32]}...")
    print(f"  seal time           : {t_seal*1000:.2f} ms")
    print(f"  verify time         : {t_ver*1000:.2f} ms")
    print(f"  verification        : {v}")
    print(f"  signature size      : "
          f"{len(json.dumps(bundle.signature))/1024:.1f} KB")

    # ------------------------------------------------------- tamper test
    print("\n" + "=" * 76)
    print("TAMPER DETECTION")
    print("=" * 76)
    tampered = list(items)
    original = tampered[1].raw_excerpt
    tampered[1].raw_excerpt = original.replace("50,000", "5,000")
    print(f"  altered E2: '50,000' -> '5,000' (one character class changed)")

    signer2 = MerkleSignatureScheme(height=4, seed=signer.seed)
    reseal = seal_bundle(tampered, case_id="NCRP/2025/000123", signer=signer2,
                         sealed_at=bundle.sealed_at)
    print(f"  original root : {bundle.merkle_root[:48]}")
    print(f"  after tamper  : {reseal.merkle_root[:48]}")
    print(f"  roots match   : {reseal.merkle_root == bundle.merkle_root}")
    tampered[1].raw_excerpt = original

    # ------------------------------------------- selective disclosure
    print("\n" + "=" * 76)
    print("SELECTIVE DISCLOSURE  (a benefit independent of quantum)")
    print("=" * 76)
    d = selective_disclosure(items, index=2, bundle=bundle)
    payload = json.dumps({
        "id": items[2].id, "kind": items[2].kind,
        "filename": items[2].filename, "sha256": items[2].sha256,
        "content": items[2].raw_excerpt,
    }, sort_keys=True, ensure_ascii=False).encode()
    ok = verify_disclosure(payload, d)

    print(f"  The victim gives the bank ONE item ({items[2].id}, "
          f"{items[2].kind})")
    print(f"  and withholds {d['items_withheld']} others, including the chat.")
    print(f"  proof size            : {len(d['inclusion_proof'])} sibling hashes")
    print(f"  bank verifies it belongs to the sealed case: {ok}")
    print(f"  bank learns about withheld items            : nothing but the count")

    # ------------------------------------------------------- scaling
    print("\n" + "=" * 76)
    print("SCALING")
    print("=" * 76)
    print(f"  {'items':>8} {'tree build':>12} {'proof len':>11} "
          f"{'proof bytes':>13}")
    for n in (4, 16, 256, 4096, 65536):
        data = [HASH(f"item{i}".encode()).digest() for i in range(n)]
        t0 = time.perf_counter(); tree = MerkleTree(data)
        t = time.perf_counter() - t0
        p = tree.proof(0)
        print(f"  {n:>8} {t*1000:>11.2f}ms {len(p):>11} "
              f"{len(p)*32:>12} B")
    print("\n  Proof size grows logarithmically. A 65,536-item archive still")
    print("  proves any single item with 16 hashes, half a kilobyte.")

    # ------------------------------------------------------- verdict
    print("\n" + "=" * 76)
    print("VERDICT ON PROBE 4")
    print("=" * 76)
    print("""
  This is the one entry point that survives.

  Not because quantum computing makes the analysis better, but because a
  future quantum computer makes today's evidence seals forgeable, and the
  fix has to be applied NOW, at sealing time. An archive sealed with ECDSA
  in 2026 cannot be retroactively re-secured in 2040; the seals will already
  be worthless. Re-sealing later proves only that the file existed in 2040.

  The countermeasure costs nothing: milliseconds per case, hash functions
  only, no new hardware, no libraries. And the Merkle structure delivers a
  privacy benefit that pays for itself immediately, whether or not a quantum
  computer is ever built.
""")


if __name__ == "__main__":
    main()
