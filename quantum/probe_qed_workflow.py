#!/usr/bin/env python3
"""
PROBE 9 — THE QED WORKFLOW, run faithfully.

Probes 1-8 encoded hand-built FEATURES. The QED workflow is different: take
the raw BYTES of an evidence file, chunk them, encode each chunk as a
normalised complex statevector with phases derived from byte content, store
the archive, and do all downstream analysis on the statevectors.

No feature engineering anywhere. That is the point of it, and it was never
tested here.

THE COMPARISON DESIGN

The tempting comparison is "classical model on classical data" versus
"quantum model on quantum-encoded data". That comparison is confounded: the
encoding is itself a transformation, so a win could come from the chunking,
the normalisation, or the phase construction rather than from anything
quantum. Three pipelines are therefore run on THE SAME RAW BYTES:

  P1  classical on raw bytes          char n-gram TF-IDF + linear model
  P2  classical on the SAME chunks    the QED chunking and normalisation,
                                      but a classical kernel on the real
                                      vectors — isolates the chunking
  P3  QED                             chunks -> complex statevectors with
                                      byte-derived phases -> fidelity kernel

P2 is the decisive control. If P3 beats P1 but ties P2, the gain came from
chunking and normalisation, not from the quantum representation. If P3 beats
both, the phase/fidelity structure is doing real work.

Task: classify an evidence file as fraud-related or ordinary.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CHUNK = 32                      # bytes per chunk -> 5 qubits (2^5 = 32)


# ═══════════════════════════════════════════════ realistic evidence text
FRAUD_TEMPLATES = [
    "Good morning. Inspector {name} speaking from {city} Cyber Crime Cell. "
    "A parcel booked in your name has been intercepted. Your Aadhaar is "
    "linked to this case under NDPS Act. You are under digital arrest from "
    "this moment. Do not disconnect. Do not inform family or bank. We must "
    "verify your funds are not from laundering. Transfer Rs {amt} to "
    "{vpa} now. Amount is fully refundable in 24 hours after RBI clearance. "
    "Verification incomplete. Rs {amt2} more required or non bailable "
    "warrant will be issued today.",

    "Dear customer your KYC has expired today. Your account will be blocked "
    "within 2 hours. Update immediately at {url} to avoid permanent "
    "suspension. Share the OTP received on your registered mobile for "
    "verification purposes only. Our executive {name} will call you. "
    "Pay Rs {amt} processing charge to {vpa} to reactivate. This is final "
    "notice from bank security department.",

    "Congratulations you are selected for part time work from home. Rs {amt} "
    "per task. Complete 3 tasks receive payment same day no investment "
    "needed. You are upgraded to prepaid group. Today task deposit Rs {amt} "
    "complete merchant reviews receive Rs {amt2} in 2 hours. Task failed due "
    "to wrong sequence. To unlock withdrawal deposit Rs {amt2} penalty "
    "clearance within today. Account under audit wait 48 hours.",

    "Sir we are Cyber Recovery Cell authorised by government. Your case funds "
    "are traced and recovered. Pay Rs {amt} processing fee to {vpa} to "
    "release Rs {amt2} to your account. Reply YES to confirm. This offer "
    "valid today only. Do not share with anyone as case is confidential.",
]

LEGIT_TEMPLATES = [
    "Hi, I have transferred the rent for this month Rs {amt} to your account. "
    "Please confirm once received. Also the plumber came yesterday and fixed "
    "the tap in the kitchen, he charged Rs {amt2}. I will adjust it in next "
    "month rent if that is fine with you. Let me know. Thanks.",

    "Order confirmed. Your payment of Rs {amt} has been received. Order will "
    "be delivered by {name} between 4 and 6 pm today. Track your order at "
    "{url}. For any queries contact our customer support. Thank you for "
    "shopping with us. Rate your experience after delivery.",

    "Sir the material has been dispatched from {city} warehouse today. Total "
    "invoice value Rs {amt}. Please arrange the balance payment Rs {amt2} "
    "after receiving the goods. Transport receipt attached. Driver number "
    "will be shared once the vehicle leaves. Kindly confirm godown timing.",

    "Hey, I paid the electricity bill Rs {amt} online just now. Also booked "
    "the tickets for next week, total Rs {amt2} for both of us. Send me your "
    "share whenever convenient, no hurry. Mom called and asked about the "
    "weekend plan. Let me know what you decide.",
]

NAMES = ["Verma", "Sharma", "Rao", "Kumar", "Singh", "Reddy", "Nair", "Das"]
CITIES = ["Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"]
VPAS = ["verify.rbi2025@okaxis", "cbi.clearance@okicici", "kyc.update@ybl",
        "secure.pay@paytm", "refund.dept@oksbi"]
URLS = ["http://sbi-kyc-verify.co.in", "https://shop.example.in/track",
        "http://hdfc-secure-update.net", "https://orders.example.com"]


LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "5", "r": "R"}


def obfuscate(txt, rng, level):
    """
    Realistic filter evasion. Scammers do this constantly: leetspeak,
    injected spaces and zero-width breaks, unicode lookalikes. It destroys
    surface n-grams while preserving meaning, which is precisely the regime
    where a representation that is not surface-bound might earn its place.
    """
    t = list(txt)
    for i, c in enumerate(t):
        r = rng.random()
        if r < level * 0.35 and c.lower() in LEET:
            t[i] = LEET[c.lower()]
        elif r < level * 0.5 and c == " ":
            t[i] = rng.choice([".", "-", "_", "  "])
        elif r < level * 0.55 and c.isalpha():
            t[i] = c + rng.choice([" ", ""])
    return "".join(t)


def make_corpus(M, seed=0, hard=False, excerpt=None):
    """Build evidence files: raw text, as they would arrive from an export."""
    rng = np.random.default_rng(seed)
    docs, y = [], []
    for m in range(M):
        fraud = rng.integers(2)
        tpl = rng.choice(FRAUD_TEMPLATES if fraud else LEGIT_TEMPLATES)
        txt = tpl.format(
            name=rng.choice(NAMES), city=rng.choice(CITIES),
            vpa=rng.choice(VPAS), url=rng.choice(URLS),
            amt=f"{int(rng.integers(1, 400)) * 500:,}",
            amt2=f"{int(rng.integers(1, 400)) * 500:,}")
        # realistic noise: typos, casing, filler — both classes get it
        if rng.random() < 0.5:
            txt = txt.lower()
        n_typo = rng.integers(0, 6)
        t = list(txt)
        for _ in range(n_typo):
            i = rng.integers(len(t))
            t[i] = chr(rng.integers(97, 123))
        out = "".join(t)
        if hard:
            # fraud text is obfuscated to evade filters; legitimate text is
            # lightly perturbed too, so obfuscation alone is not the label
            lvl = rng.uniform(0.5, 1.0) if fraud else rng.uniform(0.0, 0.35)
            out = obfuscate(out, rng, lvl)
        if excerpt:
            start = rng.integers(0, max(1, len(out) - excerpt))
            out = out[start:start + excerpt]
        docs.append(out)
        y.append(int(fraud))
    return docs, np.array(y)


# ═══════════════════════════════════════════════════ the QED encoding
def to_chunks(doc: str, n_chunks: int = 8):
    """Bytes -> fixed number of CHUNK-byte blocks, zero-padded."""
    b = doc.encode("utf-8", errors="ignore")
    need = n_chunks * CHUNK
    b = (b + bytes(need))[:need]
    return np.frombuffer(b, dtype=np.uint8).reshape(n_chunks, CHUNK).astype(float)


def qed_statevectors(doc: str, n_chunks: int = 8):
    """
    QED encoding: each 32-byte block becomes a normalised 32-dimensional
    complex statevector. Amplitude from byte magnitude, phase derived from
    byte content — so two blocks with the same byte histogram but different
    arrangement are distinguishable, which is the stated point of the format.
    """
    ch = to_chunks(doc, n_chunks)
    amp = ch / 255.0
    # phase from content: byte value plus positional term
    pos = np.arange(CHUNK)[None, :]
    phase = 2 * np.pi * (ch / 256.0) + np.pi * pos / CHUNK
    psi = amp * np.exp(1j * phase)
    nrm = np.linalg.norm(psi, axis=1, keepdims=True)
    return psi / np.where(nrm == 0, 1, nrm)


def qed_kernel(A, B=None):
    """
    Fidelity kernel between documents, averaged over aligned chunks:
        K(i,j) = mean_c |<psi_i,c | psi_j,c>|^2
    This is what a quantum device estimates with a swap test.
    """
    B = A if B is None else B
    n_chunks = A.shape[1]
    K = np.zeros((A.shape[0], B.shape[0]))
    for c in range(n_chunks):
        G = A[:, c, :].conj() @ B[:, c, :].T
        K += np.abs(G) ** 2
    return K / n_chunks


def classical_chunk_kernel(A, B=None):
    """
    CONTROL P2: identical chunking and normalisation, but REAL vectors and a
    plain cosine kernel. No phase, no fidelity. Isolates how much of any QED
    gain is just chunking and normalising.
    """
    B = A if B is None else B
    n_chunks = A.shape[1]
    K = np.zeros((A.shape[0], B.shape[0]))
    for c in range(n_chunks):
        a, b = A[:, c, :], B[:, c, :]
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        K += (a @ b.T) ** 2
    return K / n_chunks


# ═══════════════════════════════════════════════════════════ evaluation
def eval_kernel(K, y, seed=0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        out.append(roc_auc_score(y[te], clf.decision_function(K[np.ix_(te, tr)])))
    return float(np.mean(out)), float(np.std(out))


def eval_tfidf(docs, y, seed=0, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    docs = np.array(docs, dtype=object)
    out = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              min_df=2, max_features=5000)
        Xtr = vec.fit_transform(docs[tr])
        Xte = vec.transform(docs[te])
        clf = LogisticRegression(max_iter=3000)
        clf.fit(Xtr, y[tr])
        out.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(out)), float(np.std(out))


def main():
    print("=" * 78)
    print("PROBE 9  THE QED WORKFLOW — raw bytes to statevectors")
    print("=" * 78)
    print("""  Task: classify an evidence file as fraud-related or ordinary, from RAW
  BYTES only. No hand-built features in any pipeline.

  P1 classical, raw bytes    char n-gram TF-IDF + logistic regression
  P2 classical, SAME chunks  QED chunking and normalisation, real vectors,
                             cosine kernel        <- the decisive control
  P3 QED                     complex statevectors, byte-derived phases,
                             fidelity kernel
""")

    print("=" * 78)
    print("MAIN COMPARISON")
    print("=" * 78)
    print(f"  {'documents':>10} {'P1 raw bytes':>18} {'P2 same chunks':>18} "
          f"{'P3 QED':>18}")

    for M in (100, 200, 400):
        docs, y = make_corpus(M, seed=1)
        A = np.stack([qed_statevectors(d) for d in docs])
        Areal = np.stack([to_chunks(d) for d in docs])

        p1 = eval_tfidf(docs, y, seed=1)
        p2 = eval_kernel(classical_chunk_kernel(Areal), y, seed=1)
        p3 = eval_kernel(qed_kernel(A), y, seed=1)
        print(f"  {M:>10} {p1[0]:>10.4f}+-{p1[1]:<6.4f} "
              f"{p2[0]:>10.4f}+-{p2[1]:<6.4f} {p3[0]:>10.4f}+-{p3[1]:<6.4f}")

    # ─────────────────────────────────────────────── few-shot regime
    print("\n" + "=" * 78)
    print("FEW-SHOT  (the forensic reality: almost no labelled cases)")
    print("=" * 78)
    print(f"  {'train':>7} {'P1 raw bytes':>18} {'P2 same chunks':>18} "
          f"{'P3 QED':>18}")
    for n_train in (10, 20, 40, 80):
        r = {k: [] for k in ("p1", "p2", "p3")}
        for t in range(10):
            docs, y = make_corpus(n_train + 200, seed=50 + t)
            tr = np.arange(n_train); te = np.arange(n_train, n_train + 200)
            if len(np.unique(y[tr])) < 2:
                continue
            docs_a = np.array(docs, dtype=object)

            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                  min_df=1, max_features=5000)
            Xtr = vec.fit_transform(docs_a[tr]); Xte = vec.transform(docs_a[te])
            clf = LogisticRegression(max_iter=3000).fit(Xtr, y[tr])
            r["p1"].append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))

            A = np.stack([qed_statevectors(d) for d in docs])
            Ar = np.stack([to_chunks(d) for d in docs])
            for tag, K in (("p2", classical_chunk_kernel(Ar)),
                           ("p3", qed_kernel(A))):
                c = SVC(kernel="precomputed", C=1.0)
                c.fit(K[np.ix_(tr, tr)], y[tr])
                r[tag].append(roc_auc_score(
                    y[te], c.decision_function(K[np.ix_(te, tr)])))
        print(f"  {n_train:>7} {np.mean(r['p1']):>10.4f}+-{np.std(r['p1']):<6.4f} "
              f"{np.mean(r['p2']):>10.4f}+-{np.std(r['p2']):<6.4f} "
              f"{np.mean(r['p3']):>10.4f}+-{np.std(r['p3']):<6.4f}")

    # ─────────────────────────────────── hard regime: evasion + excerpts
    print("\n" + "=" * 78)
    print("HARD REGIME  filter-evading obfuscation + short excerpts")
    print("=" * 78)
    print("""  Fraud text is leetspeaked and space-broken to defeat surface n-grams,
  and only a short excerpt of each file is available. This is the regime
  where a representation that is not surface-bound could genuinely win.
""")
    print(f"  {'excerpt':>9} {'train':>6} {'P1 raw bytes':>18} "
          f"{'P2 same chunks':>18} {'P3 QED':>18}")
    for excerpt in (120, 256):
        for n_train in (20, 60):
            r = {k: [] for k in ("p1", "p2", "p3")}
            for t in range(10):
                docs, y = make_corpus(n_train + 200, seed=300 + t,
                                      hard=True, excerpt=excerpt)
                tr = np.arange(n_train); te = np.arange(n_train, n_train + 200)
                if len(np.unique(y[tr])) < 2:
                    continue
                da = np.array(docs, dtype=object)
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                      min_df=1, max_features=5000)
                Xtr = vec.fit_transform(da[tr]); Xte = vec.transform(da[te])
                clf = LogisticRegression(max_iter=3000).fit(Xtr, y[tr])
                r["p1"].append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
                A = np.stack([qed_statevectors(d, n_chunks=4) for d in docs])
                Ar = np.stack([to_chunks(d, n_chunks=4) for d in docs])
                for tag, K in (("p2", classical_chunk_kernel(Ar)),
                               ("p3", qed_kernel(A))):
                    c = SVC(kernel="precomputed", C=1.0)
                    c.fit(K[np.ix_(tr, tr)], y[tr])
                    r[tag].append(roc_auc_score(
                        y[te], c.decision_function(K[np.ix_(te, tr)])))
            print(f"  {excerpt:>9} {n_train:>6} "
                  f"{np.mean(r['p1']):>10.4f}+-{np.std(r['p1']):<6.4f} "
                  f"{np.mean(r['p2']):>10.4f}+-{np.std(r['p2']):<6.4f} "
                  f"{np.mean(r['p3']):>10.4f}+-{np.std(r['p3']):<6.4f}")

    # ─────────────────────────────────────────────── cost
    print("\n" + "=" * 78)
    print("COST")
    print("=" * 78)
    docs, y = make_corpus(400, seed=1)
    t0 = time.perf_counter(); A = np.stack([qed_statevectors(d) for d in docs])
    t_enc = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = qed_kernel(A); t_k = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = eval_tfidf(docs, y, seed=1)
    t_p1 = time.perf_counter() - t0
    print(f"  QED encode 400 docs        : {t_enc*1000:>8.1f} ms")
    print(f"  QED kernel 400x400         : {t_k*1000:>8.1f} ms  (O(M^2))")
    print(f"  P1 full 5-fold TF-IDF eval : {t_p1*1000:>8.1f} ms  (O(M))")
    print(f"\n  On hardware the kernel needs ~1/eps^2 shots PER PAIR:")
    for M in (400, 10000, 100000):
        print(f"    {M:>7,} cases -> {M*(M-1)//2:>15,} pairs "
              f"x ~10,000 shots = {M*(M-1)//2*10000:>18,} measurements")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("""
  P2 is the number that matters. If P3 and P2 land together, the QED
  representation is doing what its chunking and normalisation do, and the
  complex phases and fidelity kernel are not adding separable value on this
  data. If P3 clears both P1 and P2, the phase structure is real.

  Either way the scaling line stands on its own: the fidelity kernel is
  quadratic in cases and each entry needs thousands of shots on hardware,
  while the TF-IDF baseline is linear and runs in milliseconds.
""")


if __name__ == "__main__":
    main()
