"""
Post-quantum evidence sealing.

WHY THIS IS THE QUANTUM QUESTION THAT ACTUALLY MATTERS HERE

Four analytics entry points for quantum computing were tested against this
pipeline and all four came back negative (see quantum/FINDINGS.md and
quantum/where_quantum_helps.py). Fraud cases are small, structured and
identifier-driven; their difficulty is evidential, not computational.

But digital evidence has a property that ordinary data does not: it must
remain *verifiable* for as long as a case can be reopened. Indian criminal
appeals routinely run ten to twenty years, and evidence archives outlive the
prosecutions that created them.

That creates a real, dated quantum threat, and it runs the opposite way from
the usual one:

  - Confidentiality attacks ("harvest now, decrypt later") threaten secrecy.
  - INTEGRITY attacks threaten authenticity retroactively. Evidence sealed
    today with RSA or ECDSA can be FORGED once a cryptographically relevant
    quantum computer exists, because Shor's algorithm recovers the private
    key from the public key. At that point every signature that key ever made
    becomes repudiable — including ones made decades earlier. A defence can
    argue the archive could have been altered, and be technically correct.

The mitigation does not need a quantum computer and can be deployed today:
signatures whose security rests only on hash functions, which Shor does not
break. Grover halves the effective bit-security of a hash, so SHA-256 gives
roughly 128 bits against a quantum adversary, which remains sufficient.

WHAT THIS MODULE PROVIDES

  1. Merkle tree over the evidence bundle -> one root hash fingerprints the
     whole case. Any later alteration to any item changes the root.

  2. Inclusion proofs -> a victim can prove to a bank that ONE screenshot was
     part of the sealed bundle, without disclosing their whole chat history.
     This is a privacy benefit that exists regardless of quantum computing.

  3. Lamport one-time signatures + a Merkle key tree (the XMSS construction)
     -> many bundles signed under one long-lived public key, using only hash
     functions.

Reference: Lamport 1979; Merkle 1979; RFC 8391 (XMSS); NIST SP 800-208.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

HASH = hashlib.sha256
HASH_BYTES = 32
CLASSICAL_BITS = HASH_BYTES * 8          # 256
QUANTUM_BITS = CLASSICAL_BITS // 2       # 128, after Grover


def h(*parts: bytes) -> bytes:
    d = HASH()
    for p in parts:
        d.update(p)
    return d.digest()


# ═══════════════════════════════════════════════════════════ MERKLE TREE
def _leaf(data: bytes) -> bytes:
    return h(b"\x00", data)              # domain separation prevents


def _node(a: bytes, b: bytes) -> bytes:  # second-preimage attacks on
    return h(b"\x01", a, b)              # tree structure


class MerkleTree:
    """Merkle tree over evidence item digests."""

    def __init__(self, items: List[bytes]):
        if not items:
            raise ValueError("cannot seal an empty evidence bundle")
        self.levels: List[List[bytes]] = [[_leaf(x) for x in items]]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]
                nxt.append(_node(left, right))
            self.levels.append(nxt)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, index: int) -> List[Tuple[str, str]]:
        """Sibling path proving that leaf `index` is in the tree."""
        path = []
        idx = index
        for level in self.levels[:-1]:
            sib = idx ^ 1
            if sib >= len(level):
                sib = idx                       # duplicated odd node
            side = "left" if sib < idx else "right"
            path.append((side, level[sib].hex()))
            idx //= 2
        return path

    @staticmethod
    def verify(item: bytes, index: int, path: List[Tuple[str, str]],
               root: bytes) -> bool:
        cur = _leaf(item)
        for side, sib_hex in path:
            sib = bytes.fromhex(sib_hex)
            cur = _node(sib, cur) if side == "left" else _node(cur, sib)
        return hmac.compare_digest(cur, root)


# ═══════════════════════════════════════════ LAMPORT ONE-TIME SIGNATURE
class LamportOTS:
    """
    One-time signature from hash functions alone. Quantum-resistant: Shor
    does not apply, and Grover only halves the security margin.

    A key pair signs exactly ONE message. Reuse leaks the private key, which
    is why the Merkle scheme below manages many of them.
    """

    def __init__(self, seed: Optional[bytes] = None):
        seed = seed or os.urandom(32)
        self.sk = [[h(seed, b"sk", bytes([b]), i.to_bytes(2, "big"))
                    for i in range(CLASSICAL_BITS)] for b in (0, 1)]
        self.pk = [[h(x) for x in row] for row in self.sk]

    def public_key_digest(self) -> bytes:
        return h(b"".join(self.pk[0]), b"".join(self.pk[1]))

    def sign(self, message: bytes) -> List[str]:
        digest = HASH(message).digest()
        out = []
        for i in range(CLASSICAL_BITS):
            bit = (digest[i // 8] >> (7 - i % 8)) & 1
            out.append(self.sk[bit][i].hex())
        return out

    @staticmethod
    def verify(message: bytes, sig: List[str], pk: List[List[bytes]]) -> bool:
        digest = HASH(message).digest()
        if len(sig) != CLASSICAL_BITS:
            return False
        for i in range(CLASSICAL_BITS):
            bit = (digest[i // 8] >> (7 - i % 8)) & 1
            if not hmac.compare_digest(h(bytes.fromhex(sig[i])), pk[bit][i]):
                return False
        return True


class MerkleSignatureScheme:
    """
    XMSS-style: 2^height one-time key pairs under a single public root, so
    one published key can seal many cases over a long period.
    """

    def __init__(self, height: int = 4, seed: Optional[bytes] = None):
        self.height = height
        self.seed = seed or os.urandom(32)
        self.n_keys = 2 ** height
        self.used = 0
        self._ots = [LamportOTS(h(self.seed, i.to_bytes(4, "big")))
                     for i in range(self.n_keys)]
        self.tree = MerkleTree([k.public_key_digest() for k in self._ots])

    @property
    def public_root(self) -> str:
        return self.tree.root.hex()

    def sign(self, message: bytes) -> Dict:
        if self.used >= self.n_keys:
            raise RuntimeError(
                f"all {self.n_keys} one-time keys are spent; generate a new "
                f"key tree and publish the new root")
        idx = self.used
        self.used += 1
        ots = self._ots[idx]
        return {
            "scheme": "Merkle-Lamport (XMSS-style), SHA-256",
            "key_index": idx,
            "ots_signature": ots.sign(message),
            "ots_public_key": [[x.hex() for x in row] for row in ots.pk],
            "auth_path": self.tree.proof(idx),
            "public_root": self.public_root,
        }

    @staticmethod
    def verify(message: bytes, sig: Dict) -> bool:
        pk = [[bytes.fromhex(x) for x in row] for row in sig["ots_public_key"]]
        if not LamportOTS.verify(message, sig["ots_signature"], pk):
            return False
        digest = h(b"".join(pk[0]), b"".join(pk[1]))
        return MerkleTree.verify(
            digest, sig["key_index"],
            [(s, x) for s, x in sig["auth_path"]],
            bytes.fromhex(sig["public_root"]))


# ═══════════════════════════════════════════════════════ EVIDENCE SEALING
@dataclass
class SealedBundle:
    case_id: str
    sealed_at: str
    n_items: int
    item_digests: List[str]
    merkle_root: str
    signature: Dict
    algorithm_note: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def seal_bundle(evidence_items, case_id: str, signer: MerkleSignatureScheme,
                sealed_at: Optional[str] = None) -> SealedBundle:
    """
    Seal a Golden Hour evidence bundle.

    Each item is hashed, the digests form a Merkle tree, and the root is
    signed with a hash-based signature. The seal proves, to anyone holding
    the public root, that this exact set of items existed in this exact form
    at sealing time.
    """
    from datetime import datetime, timezone
    sealed_at = sealed_at or datetime.now(timezone.utc).isoformat()

    digests = []
    for it in evidence_items:
        payload = json.dumps({
            "id": it.id, "kind": it.kind, "filename": it.filename,
            "sha256": it.sha256, "content": it.raw_excerpt,
        }, sort_keys=True, ensure_ascii=False).encode()
        digests.append(HASH(payload).digest())

    tree = MerkleTree(digests)
    header = json.dumps({
        "case_id": case_id, "sealed_at": sealed_at,
        "n_items": len(digests), "root": tree.root.hex(),
    }, sort_keys=True).encode()

    return SealedBundle(
        case_id=case_id, sealed_at=sealed_at, n_items=len(digests),
        item_digests=[d.hex() for d in digests],
        merkle_root=tree.root.hex(),
        signature=signer.sign(header),
        algorithm_note=(
            f"SHA-256 Merkle tree, Merkle-Lamport signature. Security rests "
            f"only on hash-function properties: {CLASSICAL_BITS}-bit "
            f"classical, ~{QUANTUM_BITS}-bit against a quantum adversary "
            f"(Grover). Not affected by Shor's algorithm, unlike RSA/ECDSA."),
    )


def verify_seal(bundle: SealedBundle) -> Dict[str, bool]:
    digests = [bytes.fromhex(x) for x in bundle.item_digests]
    tree = MerkleTree(digests)
    root_ok = tree.root.hex() == bundle.merkle_root
    header = json.dumps({
        "case_id": bundle.case_id, "sealed_at": bundle.sealed_at,
        "n_items": bundle.n_items, "root": bundle.merkle_root,
    }, sort_keys=True).encode()
    sig_ok = MerkleSignatureScheme.verify(header, bundle.signature)
    return {"merkle_root_matches": root_ok, "signature_valid": sig_ok,
            "sealed": root_ok and sig_ok}


def selective_disclosure(evidence_items, index: int, bundle: SealedBundle
                         ) -> Dict:
    """
    Prove ONE evidence item belongs to the sealed bundle without revealing
    the others.

    A victim can hand a bank the single payment screenshot it needs, with
    proof it was part of the sealed case, while keeping the rest of their
    chat history private. This benefit does not depend on quantum computing
    at all; it comes free with the tree.
    """
    digests = [bytes.fromhex(x) for x in bundle.item_digests]
    tree = MerkleTree(digests)
    it = evidence_items[index]
    payload = json.dumps({
        "id": it.id, "kind": it.kind, "filename": it.filename,
        "sha256": it.sha256, "content": it.raw_excerpt,
    }, sort_keys=True, ensure_ascii=False).encode()
    return {
        "disclosed_item": {"id": it.id, "kind": it.kind,
                           "digest": HASH(payload).digest().hex()},
        "index": index,
        "inclusion_proof": tree.proof(index),
        "merkle_root": bundle.merkle_root,
        "items_withheld": bundle.n_items - 1,
    }


def verify_disclosure(item_payload: bytes, disclosure: Dict) -> bool:
    return MerkleTree.verify(
        HASH(item_payload).digest(), disclosure["index"],
        [(s, x) for s, x in disclosure["inclusion_proof"]],
        bytes.fromhex(disclosure["merkle_root"]))
