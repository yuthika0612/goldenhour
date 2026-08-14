"""
Deterministic checks. These run in Python, NOT in the model.

Rationale (state this to the reviewer): in prompt-only testing the model
missed an arithmetically impossible balance sequence and instead emitted
filler findings. Arithmetic, format validation and duplicate detection are
cheap, exact and reproducible in code, so they belong in code. The model is
used for language understanding; the rules are used for truth.

Each function appends Finding objects with origin="rules".
"""
from typing import List
import re
from .schema import IncidentRecord, Finding

# ---------------------------------------------------------------- helpers
_UTR_RE = re.compile(r"^[A-Za-z0-9]{12,22}$")
_RRN_RE = re.compile(r"^\d{12}$")
_UPI_RE = re.compile(r"^[a-zA-Z0-9._-]{2,64}@[a-zA-Z]{2,32}$")
_PHONE_RE = re.compile(r"^(\+?91[\s-]?)?[6-9]\d{9}$")

# handles that impersonate an authority; funds to these are never official
_AUTHORITY_TOKENS = (
    "rbi", "cbi", "police", "cyber", "customs", "income", "tax", "gov",
    "court", "ed", "trai", "npci", "clearance", "verify", "i4c",
)

_SEQ = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _fid(rec: IncidentRecord) -> str:
    return f"F{len(rec.findings) + 1}"


def _add(rec, severity, check, observed, evidence, why, resolves):
    rec.findings.append(Finding(
        id=_fid(rec), severity=severity, check=check, observed=observed,
        evidence=evidence or [], why_it_matters=why,
        resolves_with=resolves, origin="rules",
    ))


def _money(x) -> str:
    try:
        return f"Rs {float(x):,.0f}"
    except (TypeError, ValueError):
        return str(x)


# ------------------------------------------------------------ the checks
def check_balance_arithmetic(rec: IncidentRecord) -> None:
    """
    I7 — a running balance sequence that cannot occur.

    Walk transactions that carry a balance_after in evidence order. If the
    balance rises between two debits with no credit in evidence to explain
    it, the record is internally impossible: either an alert is fabricated,
    an inflow is missing from the bundle, or the ordering is wrong.
    """
    withbal = [t for t in rec.transactions if t.balance_after is not None]
    if len(withbal) < 2:
        return

    for prev, curr in zip(withbal, withbal[1:]):
        if prev.balance_after is None or curr.balance_after is None:
            continue
        expected = prev.balance_after - (curr.amount or 0) if curr.direction == "debit" \
            else prev.balance_after + (curr.amount or 0)
        drift = round(curr.balance_after - expected, 2)
        if abs(drift) < 1:
            continue

        rising = curr.direction == "debit" and curr.balance_after > prev.balance_after
        _add(
            rec,
            "HIGH", "I7/ARITHMETIC",
            (f"Balance after {curr.txn_ref or 'transaction'} is "
             f"{_money(curr.balance_after)}, but {_money(prev.balance_after)} "
             f"minus {_money(curr.amount)} should leave {_money(expected)} "
             f"(unexplained difference {_money(abs(drift))})"
             + (". The balance INCREASED across a debit with no credit in evidence."
                if rising else ".")),
            list({*prev.sources, *curr.sources}),
            "The bank alerts cannot all be genuine as recorded, or a "
            "transaction is missing from the bundle. Either changes the "
            "true loss figure.",
            "Full bank statement for the account covering the incident window.",
        )


def check_duplicate_refs(rec: IncidentRecord) -> None:
    """T2 — the same reference number on two different payments."""
    seen = {}
    for t in rec.transactions:
        if not t.txn_ref:
            continue
        key = t.txn_ref.strip().upper()
        if key in seen:
            other = seen[key]
            if (other.amount != t.amount) or (other.payee_upi_id != t.payee_upi_id):
                _add(
                    rec, "HIGH", "T2",
                    (f"Reference {t.txn_ref} appears on two different payments: "
                     f"{_money(other.amount)} to {other.payee_upi_id or 'unknown'} "
                     f"and {_money(t.amount)} to {t.payee_upi_id or 'unknown'}"),
                    list({*other.sources, *t.sources}),
                    "A UTR/RRN is unique to one transaction, so at least one of "
                    "these artifacts misstates its reference.",
                    "Bank confirmation of which amount this reference belongs to.",
                )
        else:
            seen[key] = t


def check_ref_formats(rec: IncidentRecord) -> None:
    """T1 — reference of unexpected shape. Reported as needing verification."""
    for t in rec.transactions:
        if not t.txn_ref:
            continue
        ref = t.txn_ref.strip()
        if _UTR_RE.match(ref) or _RRN_RE.match(ref):
            continue
        _add(
            rec, "MEDIUM", "T1",
            f"Reference '{ref}' does not match the usual UTR/RRN shape "
            f"(12-22 alphanumeric, or 12 digits for RRN)",
            t.sources,
            "May be an OCR misread or an artifact that was not produced by "
            "the payment app. Format alone does not prove forgery.",
            "Check the reference against the bank or app transaction history.",
        )


def check_uncorroborated(rec: IncidentRecord) -> None:
    """
    I6 — a payment with no independent corroboration.

    One source (a chat claim alone) is weak. A bank alert is strong. This
    check deliberately does not double-penalise a payment that has a bank
    alert but no screenshot.
    """
    for t in rec.transactions:
        srcs = {s for s in t.sources}
        kinds = {e.kind for e in rec.evidence_items if e.id in srcs}
        if "sms" in kinds:
            continue  # bank alert present: corroborated
        if len(srcs) <= 1:
            _add(
                rec, "MEDIUM", "I6",
                (f"Payment {t.txn_ref or ''} of {_money(t.amount)} appears in "
                 f"only one place ({', '.join(sorted(srcs)) or 'unknown'}) with "
                 f"no bank alert"),
                sorted(srcs),
                "Without a bank record this payment cannot be evidenced to "
                "the bank or the portal.",
                "Bank statement line or the app's transaction history entry.",
            )


def check_payee_drift(rec: IncidentRecord) -> None:
    """S11 — beneficiary changes mid-sequence (mule handoff)."""
    payees = [t.payee_upi_id or t.payee_name for t in rec.debits()
              if (t.payee_upi_id or t.payee_name)]
    uniq = list(dict.fromkeys(payees))
    if len(uniq) > 1:
        _add(
            rec, "HIGH", "S11",
            f"Money went to {len(uniq)} different beneficiaries: {', '.join(uniq)}",
            [s for t in rec.debits() for s in t.sources],
            "Rotating beneficiaries mid-incident indicates a mule account "
            "chain rather than a single payee, and every account in the chain "
            "is a separate freeze target.",
            "Bank to provide account details behind each beneficiary handle.",
        )


def check_authority_handle(rec: IncidentRecord) -> None:
    """I3 — official-sounding collection into a personal VPA."""
    for t in rec.debits():
        h = (t.payee_upi_id or "").lower()
        if not h:
            continue
        if any(tok in h.split("@")[0] for tok in _AUTHORITY_TOKENS):
            _add(
                rec, "HIGH", "I3",
                f"Funds were sent to '{t.payee_upi_id}', a handle that invokes "
                f"an official body",
                t.sources,
                "No government or regulatory body collects money into a "
                "personal UPI handle. This is a strong indicator of "
                "impersonation.",
                "None needed for the complaint; state it as an observed fact.",
            )


def check_upi_and_phone_formats(rec: IncidentRecord) -> None:
    for t in rec.transactions:
        for h in (t.payee_upi_id, t.payer_upi_id):
            if h and not _UPI_RE.match(h):
                _add(
                    rec, "LOW", "T1",
                    f"UPI handle '{h}' is malformed, possibly an OCR artifact",
                    t.sources,
                    "A wrong handle in the complaint delays action.",
                    "Re-read the handle from the original screenshot.",
                )
    for m in rec.suspect_mobiles:
        digits = re.sub(r"[^\d+]", "", m)
        if "X" in m.upper() or "*" in m:
            continue  # deliberately masked by the user
        if not _PHONE_RE.match(digits):
            _add(
                rec, "LOW", "T1",
                f"Suspect number '{m}' is not a valid Indian mobile format",
                [], "May be mistyped; portals reject malformed numbers.",
                "Confirm from the call log.",
            )


def check_ncrp_text_constraints(rec: IncidentRecord) -> None:
    """
    Portal rules from the formats document: description must be at least
    200 characters and must not contain # $ @ ^ * ` ' ~ | !
    """
    d = rec.incident_description or ""
    if len(d) < 200:
        _add(
            rec, "MEDIUM", "FORMAT",
            f"Incident description is {len(d)} characters; the portal requires "
            f"at least 200",
            [], "The complaint will be rejected on submission.",
            "Expand the description with the timeline detail already collected.",
        )
    bad = sorted({c for c in d if c in "#$@^*`'~|!"})
    if bad:
        _add(
            rec, "MEDIUM", "FORMAT",
            f"Description contains characters the portal disallows: {' '.join(bad)}",
            [], "The portal rejects these characters in the narrative field.",
            "Sanitise before submission (the generator does this automatically).",
        )


def check_role_consistency(rec: IncidentRecord) -> None:
    """A payee-victim has no debits to report; guard against wrong framing."""
    if rec.user_role == "PAYEE_VICTIM" and rec.debits():
        _add(
            rec, "MEDIUM", "OTHER",
            "Role is PAYEE_VICTIM (shown proof of a payment) but debits were "
            "extracted from the user's own account",
            [], "The complaint would ask the bank to freeze the wrong side of "
                "the transaction.",
            "Confirm with the user whether they also sent money.",
        )
    if rec.user_role == "PAYEE_VICTIM" and not rec.credits():
        _add(
            rec, "CRITICAL", "I6",
            "A payment was claimed to the user but no matching credit exists "
            "in the user's own records",
            [], "The claimed payment did not reach the account. Goods or "
                "services must not be released on this proof.",
            "The user's own bank statement for the claimed date and amount.",
        )


def assess_fraud_likelihood(rec: IncidentRecord) -> str:
    """
    Distinguishes 'this looks like fraud' from 'this looks ordinary'.

    Returns one of: LIKELY_FRAUD, POSSIBLE_FRAUD, NO_INDICATORS_FOUND.

    This exists because the pipeline previously had no concept of a clean
    case -- every input, including an ordinary rent payment, was narrated
    as "I am reporting an online financial fraud." That is not a cosmetic
    gap: a tool that tells people to file a false police complaint is
    actively harmful, not just unhelpful.

    This is a coarse, explainable heuristic, not a classifier -- deliberately
    so. It looks only for evidence already present in the record (impossibility
    checks I1-I8 firing, HIGH/CRITICAL findings, or fraud-typical language in
    the raw evidence) rather than guessing. When nothing fires, it says so.
    """
    strong_checks = {"I1", "I2", "I3", "I4", "I5", "I8"}
    if any(f.check.split("/")[0] in strong_checks for f in rec.findings):
        return "LIKELY_FRAUD"

    high = [f for f in rec.findings
            if f.severity in ("HIGH", "CRITICAL") and f.check != "FORMAT"]
    if high:
        return "POSSIBLE_FRAUD"

    fraud_words = ("arrest", "warrant", "cbi", "rbi clearance", "refundable",
                  "penalty", "unlock withdrawal", "digital arrest", "task",
                  "vip group", "customs", "parcel", "ndps", "kyc expire",
                  "blocked", "suspend", "fee to recover", "processing fee")
    blob = " ".join(e.raw_excerpt.lower() for e in rec.evidence_items)
    if any(w in blob for w in fraud_words):
        return "POSSIBLE_FRAUD"

    if rec.fraud_type and rec.fraud_type not in ("Other", "NOT FOUND IN EVIDENCE"):
        return "LIKELY_FRAUD"

    return "NO_INDICATORS_FOUND"


ALL_CHECKS = (
    check_balance_arithmetic,
    check_duplicate_refs,
    check_ref_formats,
    check_uncorroborated,
    check_payee_drift,
    check_authority_handle,
    check_upi_and_phone_formats,
    check_ncrp_text_constraints,
    check_role_consistency,
)


def run_all(rec: IncidentRecord) -> List[Finding]:
    """Run every deterministic check, then sort findings by severity."""
    before = len(rec.findings)
    for fn in ALL_CHECKS:
        try:
            fn(rec)
        except Exception as exc:                      # a broken check must
            _add(rec, "LOW", "OTHER",                 # never kill the run
                 f"Check {fn.__name__} could not complete: {exc}",
                 [], "Internal validation gap.", "Review input data shape.")
    rec.findings.sort(key=lambda f: _SEQ.get(f.severity, 9))
    for i, f in enumerate(rec.findings, 1):
        f.id = f"F{i}"
    rec.fraud_likelihood = assess_fraud_likelihood(rec)
    return rec.findings[before:]
