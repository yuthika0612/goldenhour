"""
Deterministic extraction from bank alert SMS.

Bank alerts are machine generated and highly formulaic, so regex handles
them exactly and for free. This layer exists for three reasons:

  1. It gives the pipeline a working fallback when no API key is available.
  2. It cross-checks the model: if the model and the regex disagree about an
     amount or a reference, that disagreement is itself reportable.
  3. Bank SMS carry the balance figures that drive the arithmetic checks,
     and those must never depend on model behaviour.

Chat and screenshot text is NOT handled here. That needs the model.
"""
import re
from typing import List

from .schema import Transaction, EvidenceItem

AMOUNT = r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)"
REF = r"(?:UPI\s*Ref|Ref(?:erence)?(?:\s*No\.?)?|UTR|RRN|Txn\s*ID)\s*[:.]?\s*([A-Za-z0-9]{6,25})"
BAL = rf"(?:Avl|Available|Avlbl)\s*Bal(?:ance)?\s*[:.]?\s*{AMOUNT}"
ACCT = r"(?:A/c|Acct|Account)\s*(?:No\.?)?\s*[:.]?\s*([X\*x]{0,6}\d{3,6})"
DATE = r"(\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
TIME = r"(\d{1,2}:\d{2}\s*(?:[APap]\.?[Mm]\.?)?)"
PAYEE = r"(?:to|To)\s+([A-Z][A-Z\s&.]{2,40}?)(?=\.|,|\s+Avl|\s+Ref|$)"
UPI_ID = r"\b([a-zA-Z0-9._-]{2,64}@[a-zA-Z]{2,32})\b"

DEBIT_WORDS = ("debited", "debit", "sent", "paid", "withdrawn", "transferred")
CREDIT_WORDS = ("credited", "credit", "received", "deposited")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _bank_from_line(line: str) -> str:
    m = re.match(r"^\s*([A-Z][A-Za-z ]{1,20}?)\s*[:\-]", line)
    if m and len(m.group(1).strip()) <= 20:
        return m.group(1).strip()
    for name in ("SBI", "HDFC", "ICICI", "Axis", "Kotak", "Union Bank",
                 "PNB", "Canara", "BOB", "Yes Bank", "IDFC", "IndusInd"):
        if name.lower() in line.lower():
            return name
    return None


def extract_from_sms(items: List[EvidenceItem]) -> List[Transaction]:
    txns: List[Transaction] = []
    for item in items:
        if item.kind != "sms":
            continue
        for line in item.raw_excerpt.splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()

            direction = None
            if any(w in low for w in DEBIT_WORDS):
                direction = "debit"
            elif any(w in low for w in CREDIT_WORDS):
                direction = "credit"
            if direction is None:
                continue                      # not a transaction alert

            amounts = re.findall(AMOUNT, line)
            if not amounts:
                continue

            bal_m = re.search(BAL, line, re.I)
            balance = _num(bal_m.group(1)) if bal_m else None
            # the transaction amount is the first amount that is not the balance
            amt = None
            for a in amounts:
                v = _num(a)
                if balance is not None and abs(v - balance) < 0.001:
                    continue
                amt = v
                break
            if amt is None:
                continue

            ref_m = re.search(REF, line, re.I)
            date_m = re.search(DATE, line)
            time_m = re.search(TIME, line)
            payee_m = re.search(PAYEE, line)
            upi_m = re.search(UPI_ID, line)

            when = " ".join(x.group(1) for x in (date_m, time_m) if x) or None

            txns.append(Transaction(
                txn_ref=ref_m.group(1) if ref_m else None,
                amount=amt,
                direction=direction,
                datetime=when,
                payee_upi_id=upi_m.group(1) if upi_m else None,
                payee_name=payee_m.group(1).strip() if payee_m else None,
                bank_wallet_merchant=_bank_from_line(line),
                balance_after=balance,
                sources=[item.id],
                confidence="HIGH",
                note="extracted deterministically from bank alert",
            ))
    return txns


def extract_account_no(items: List[EvidenceItem]) -> str:
    for item in items:
        if item.kind != "sms":
            continue
        m = re.search(ACCT, item.raw_excerpt)
        if m:
            return m.group(1)
    return None


def extract_handles(items: List[EvidenceItem]) -> dict:
    """UPI ids, phone numbers and URLs from anywhere in the bundle."""
    blob = "\n".join(i.raw_excerpt for i in items)
    upi = sorted(set(re.findall(UPI_ID, blob)))
    phones = sorted(set(re.findall(
        r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}|\+91\s?\d{2}[X\*]{3,5}\s?\d{5}",
        blob)))
    urls = sorted(set(re.findall(r"https?://[^\s,\)\]]+", blob)))
    return {"upi_ids": upi, "phones": phones, "urls": urls}


def cross_check(model_txns: List[Transaction],
                rule_txns: List[Transaction]) -> List[str]:
    """
    Compare the model's transactions against the regex ones. Disagreement on
    a bank-sourced figure is a real finding, not noise.
    """
    notes = []
    by_ref = {t.txn_ref.upper(): t for t in rule_txns if t.txn_ref}
    for mt in model_txns:
        if not mt.txn_ref:
            continue
        rt = by_ref.get(mt.txn_ref.upper())
        if not rt:
            continue
        if mt.amount is not None and rt.amount is not None and \
                abs(mt.amount - rt.amount) > 0.5:
            notes.append(
                f"Model read {mt.txn_ref} as Rs {mt.amount:,.0f} but the bank "
                f"alert states Rs {rt.amount:,.0f}. Bank alert taken as "
                f"authoritative.")
        if rt.balance_after is not None and mt.balance_after is None:
            mt.balance_after = rt.balance_after   # backfill for arithmetic
    return notes


def merge(model_txns: List[Transaction],
          rule_txns: List[Transaction]) -> List[Transaction]:
    """
    Union by reference. Bank-sourced values win on amount and balance; the
    model's values fill in payee, handle and timing detail.
    """
    out = {}
    for t in model_txns:
        key = (t.txn_ref or "").upper() or f"noref-{id(t)}"
        out[key] = t
    for rt in rule_txns:
        key = (rt.txn_ref or "").upper() or f"noref-{id(rt)}"
        if key in out:
            mt = out[key]
            mt.amount = rt.amount if rt.amount is not None else mt.amount
            mt.balance_after = (rt.balance_after if rt.balance_after is not None
                                else mt.balance_after)
            mt.direction = rt.direction
            mt.bank_wallet_merchant = (mt.bank_wallet_merchant
                                       or rt.bank_wallet_merchant)
            mt.payee_name = mt.payee_name or rt.payee_name
            mt.sources = sorted(set(mt.sources) | set(rt.sources))
            mt.confidence = "HIGH"
        else:
            out[key] = rt
    return list(out.values())
