"""
Four outputs from one incident record, mapped field-for-field to the
project formats document:

  1. NCRP financial-fraud complaint data   (official portal checklist)
  2. 1930 emergency call brief             (project template)
  3. Bank fraud complaint letter           (project template)
  4. NPCI / UPI grievance data package     (project template)

Plus the shared evidence index and timeline.

Only the NCRP field list reproduces the portal's stated checklist. The
other three are structured templates and are labelled as such in the
generated text, so nothing is passed off as an official government form.
"""
from typing import Dict
from .schema import IncidentRecord, NOT_FOUND

NCRP_FORBIDDEN = "#$@^*`'~|!"
DISCLAIMER = (
    "Prepared by Golden Hour, an assistive tool. Contents are derived from "
    "evidence supplied by the complainant and must be verified before "
    "submission. This is not legal advice."
)


# ------------------------------------------------------------------ utils
def sanitise_ncrp(text: str) -> str:
    """Portal rejects certain characters in the narrative field."""
    out = "".join(" " if c in NCRP_FORBIDDEN else c for c in text)
    return " ".join(out.split())


def _mask_account(acc: str) -> str:
    if not acc or acc == NOT_FOUND:
        return NOT_FOUND
    tail = acc[-4:]
    return f"XXXX{tail}"


def _rupees(v) -> str:
    if v in (None, ""):
        return NOT_FOUND
    try:
        return f"Rs {float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _txn_rows(rec: IncidentRecord):
    for t in rec.debits():
        yield {
            "ref": t.txn_ref or NOT_FOUND,
            "amount": _rupees(t.amount),
            "datetime": t.datetime or NOT_FOUND,
            "payee": t.payee_upi_id or t.payee_name or NOT_FOUND,
            "bank": t.bank_wallet_merchant or NOT_FOUND,
        }


def build_narrative(rec: IncidentRecord) -> str:
    """
    Chronological, plain, every sentence traceable to a timeline event.
    Padded to satisfy the portal's 200-character minimum only by adding
    real detail, never filler.
    """
    if rec.incident_description:
        return rec.incident_description

    parts = []
    if rec.timeline_events:
        opener = rec.timeline_events[0]
        parts.append(
            f"On {opener.timestamp or 'a date recorded in the evidence'}, the "
            f"complainant was contacted as follows: {opener.description}"
        )
        mid = [e for e in rec.timeline_events[1:]
               if e.stage in ("S03", "S04", "S05", "S06", "S08", "S10", "S11")]
        for e in mid[:6]:
            parts.append(e.description.rstrip("."))
    debits = rec.debits()
    if debits:
        refs = ", ".join(t.txn_ref for t in debits if t.txn_ref) or NOT_FOUND
        parts.append(
            f"The complainant transferred a total of {_rupees(rec.total_fraud_amount)} "
            f"across {len(debits)} transaction(s), with references {refs}"
        )
    if rec.credits():
        parts.append(
            f"An amount of {_rupees(rec.total_recovered_inflow)} was received back "
            f"during the incident, giving a net loss of {_rupees(rec.net_loss)}"
        )
    tail = [e for e in rec.timeline_events if e.stage in ("S13", "S14", "S15")]
    if tail:
        parts.append(tail[-1].description.rstrip("."))

    text = ". ".join(p.rstrip(".") for p in parts if p) + "."
    return sanitise_ncrp(text)


# ------------------------------------------------------------------ 1 NCRP
def generate_ncrp(rec: IncidentRecord) -> Dict:
    desc = sanitise_ncrp(build_narrative(rec))
    return {
        "form": "NCRP - National Cyber Crime Reporting Portal (financial fraud)",
        "note": "Field specification per the portal checklist. Verify the live "
                "interface before submission.",
        "complainant": {
            "name": rec.victim_name or NOT_FOUND,
            "mobile": rec.victim_mobile or NOT_FOUND,
            "email": rec.victim_email or NOT_FOUND,
            "address": rec.victim_address or NOT_FOUND,
        },
        "mandatory": {
            "incident_date_time": rec.incident_datetime or NOT_FOUND,
            "incident_details": desc,
            "incident_details_length": len(desc),
            "complainant_national_id": "[ATTACH: Aadhaar / PAN / Voter ID / DL / "
                                       "Passport, JPEG or PNG, max 5 MB]",
            "relevant_evidence": [
                f"{e.id}: {e.filename or e.kind}" for e in rec.evidence_items
            ] or [NOT_FOUND],
        },
        "financial_fraud": {
            "bank_wallet_merchant": rec.bank_wallet_merchant or NOT_FOUND,
            "transactions": list(_txn_rows(rec)),
            "fraud_amount": _rupees(rec.total_fraud_amount),
            "net_loss_after_inflows": _rupees(rec.net_loss),
        },
        "suspect_details_optional_but_desirable": {
            "mobile_numbers": rec.suspect_mobiles or [NOT_FOUND],
            "email_ids": rec.suspect_emails or [NOT_FOUND],
            "upi_ids": rec.suspect_upi_ids or [NOT_FOUND],
            "bank_accounts": rec.suspect_bank_accounts or [NOT_FOUND],
            "websites_or_handles": rec.suspect_urls_social_handles or [NOT_FOUND],
            "claimed_aliases": rec.suspect_aliases or [NOT_FOUND],
            "address": rec.suspect_address or NOT_FOUND,
        },
        "classification": {
            "fraud_type": rec.fraud_type or NOT_FOUND,
            "basis": rec.fraud_type_basis or NOT_FOUND,
            "fraud_likelihood": rec.fraud_likelihood,
        },
        "validation": {
            "min_200_chars": len(desc) >= 200,
            "forbidden_chars_present": any(c in desc for c in NCRP_FORBIDDEN),
        },
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------------------------ 2 1930
def generate_1930_brief(rec: IncidentRecord) -> str:
    """Compact, readable aloud. 1930 is a call, so this is a call script."""
    debits = rec.debits()
    no_indicators = rec.fraud_likelihood == "NO_INDICATORS_FOUND"

    lines = [
        "1930 CALL BRIEF  (project template, not an official form)",
        "Read the first three lines first. They are what gets funds held."
        if not no_indicators else
        "NOTE: no fraud indicators were found in the evidence provided. "
        "See the assessment below before calling.",
        "",
    ]
    if no_indicators:
        lines += [
            "This transaction does not show the pattern typically seen in "
            "digital fraud (no impersonation, no urgency/threat language, no "
            "unverified new payee, no report of amounts sent under pressure).",
            "If you still believe this is fraudulent, call 1930 and describe "
            "specifically why -- the checklist below is provided in case it "
            "is needed, but filing is not indicated by the evidence seen here.",
            "",
        ]
    else:
        confidence_line = (
            "Based on strong fraud indicators in the evidence."
            if rec.fraud_likelihood == "LIKELY_FRAUD" else
            "Based on some suspicious indicators; confirm details before "
            "calling."
        )
        lines += [
            f"1. I am reporting a suspected online financial fraud. I lost "
            f"{_rupees(rec.total_fraud_amount)}. ({confidence_line})",
            f"2. The money left my account on {rec.incident_datetime or NOT_FOUND}.",
            f"3. Fraud type: {rec.fraud_type or NOT_FOUND}.",
            "",
        ]
    lines += [
        f"Complainant name      : {rec.victim_name or '[TO BE FILLED]'}",
        f"Complainant mobile    : {rec.victim_mobile or '[TO BE FILLED]'}",
        f"Complainant email     : {rec.victim_email or '[TO BE FILLED]'}",
        f"Complainant address   : {rec.victim_address or '[TO BE FILLED]'}",
        f"Bank / wallet         : {rec.bank_wallet_merchant or NOT_FOUND}",
        f"My account            : {_mask_account(rec.victim_account_no or '')}",
        f"Total amount          : {_rupees(rec.total_fraud_amount)}",
        f"Net loss              : {_rupees(rec.net_loss)}",
        "",
        "TRANSACTIONS TO TRACE (most recent first, quote these references):",
    ]
    for r in sorted(_txn_rows(rec), key=lambda x: x["datetime"], reverse=True):
        lines.append(f"   {r['datetime']}  {r['amount']}  ref {r['ref']}  "
                     f"to {r['payee']}")
    if no_indicators:
        lines += [
            "",
            "SUGGESTED ACTION:",
            "   No urgent filing action indicated. If something about this",
            "   still feels wrong, describe specifically what, and consider",
            "   contacting your bank directly to ask about the transaction.",
            "",
            DISCLAIMER,
        ]
    else:
        lines += [
            "",
            f"Suspect numbers       : {', '.join(rec.suspect_mobiles) or NOT_FOUND}",
            f"Beneficiary UPI IDs   : {', '.join(rec.suspect_upi_ids) or NOT_FOUND}",
            f"Suspect accounts      : {', '.join(rec.suspect_bank_accounts) or NOT_FOUND}",
            "",
            "WHAT HAPPENED (2-3 sentences):",
            "   " + " ".join(build_narrative(rec).split()[:60]) + " ...",
            "",
            f"Already filed?        : NCRP {rec.ncrp_acknowledgement_no or 'not yet'}"
            f" / Bank {rec.bank_complaint_reference or 'not yet'}",
            f"Evidence available    : "
            f"{', '.join(sorted({e.kind for e in rec.evidence_items})) or NOT_FOUND}",
            "",
            "IMMEDIATE ACTION REQUESTED:",
            "   Urgent tracing and hold, freeze or recall of the fraudulent funds",
            "   in the beneficiary accounts listed above.",
            "",
            DISCLAIMER,
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ 3 Bank
def generate_bank_letter(rec: IncidentRecord) -> str:
    rows = list(_txn_rows(rec))
    table = "\n".join(
        f"   {i}. Ref {r['ref']} | {r['amount']} | {r['datetime']} | to {r['payee']}"
        for i, r in enumerate(rows, 1)
    ) or f"   {NOT_FOUND}"

    return f"""BANK FRAUD COMPLAINT  (project template, not an official form)

To
The Branch Manager / Fraud and Customer Support Department
{rec.victim_bank_branch or rec.bank_wallet_merchant or '[BANK AND BRANCH]'}

Subject: Reporting of unauthorised and fraudulently induced electronic
transactions, and request for urgent hold, trace and recall

Respected Sir or Madam,

Complainant name        : {rec.victim_name or '[TO BE FILLED]'}
Registered mobile       : {rec.victim_mobile or '[TO BE FILLED]'}
Email                   : {rec.victim_email or '[TO BE FILLED]'}
Address                 : {rec.victim_address or '[TO BE FILLED]'}
Account number          : {_mask_account(rec.victim_account_no or '')}
Type of fraud           : {rec.fraud_type or NOT_FOUND}
Date and time of fraud  : {rec.incident_datetime or NOT_FOUND}
Total amount involved   : {_rupees(rec.total_fraud_amount)}

I wish to report the following transactions from my account, which were
induced by fraudulent means and were not authorised by me with knowledge of
their true purpose.

TRANSACTIONS
{table}

DESCRIPTION OF FRAUD
{build_narrative(rec)}

EVIDENCE ATTACHED
{chr(10).join(f'   {e.id}. {e.filename or e.kind} ({e.quality.lower()})' for e in rec.evidence_items) or '   ' + NOT_FOUND}

REQUEST
   1. Mark the above transactions as fraudulent and unauthorised.
   2. Urgently trace and place a hold, freeze or recall on the funds in the
      beneficiary accounts, in accordance with applicable RBI guidance on
      unauthorised electronic banking transactions.
   3. Preserve all transaction records, beneficiary details and logs
      relating to the above references.
   4. Provide a written complaint reference number for follow up.

Reported to the cyber crime helpline and portal:
   NCRP acknowledgement : {rec.ncrp_acknowledgement_no or 'To be updated'}
   1930 reference       : {rec.bank_complaint_reference or 'To be updated'}

Date  : [DATE]
Place : [PLACE]
Signature : ______________________
Name : {rec.victim_name or '[TO BE FILLED]'}

{DISCLAIMER}
"""


# ------------------------------------------------------------------ 4 NPCI
def generate_npci_package(rec: IncidentRecord) -> Dict:
    """
    NPCI routes complaints to the member bank, so this is a per-transaction
    data package rather than a letter.
    """
    return {
        "form": "NPCI / UPI grievance data package (project template)",
        "note": "NPCI routes complaints to the relevant member bank. Fraudulent "
                "transactions should also be pursued with the bank directly.",
        "product_payment_system": "UPI",
        "bank_member": rec.bank_wallet_merchant or NOT_FOUND,
        "complaint_issue_type": "Fraudulent / unauthorised transaction",
        "transactions": [
            {
                "transaction_type": "P2P" if (t.payee_upi_id and not t.payee_name)
                                    else "P2M",
                "rrn_or_reference": t.txn_ref or NOT_FOUND,
                "transaction_datetime": t.datetime or NOT_FOUND,
                "amount": _rupees(t.amount),
                "payer_name": rec.victim_name or NOT_FOUND,
                "payer_upi_id": t.payer_upi_id or NOT_FOUND,
                "payee_name": t.payee_name or NOT_FOUND,
                "payee_upi_id": t.payee_upi_id or NOT_FOUND,
                "beneficiary_bank_ifsc": t.beneficiary_bank_ifsc or NOT_FOUND,
                "remarks": "Transaction induced by fraud. Request trace and "
                           "action on beneficiary account.",
            }
            for t in rec.debits()
        ],
        "complaint_reference_number": rec.ncrp_acknowledgement_no or "[CRN after submission]",
        "disclaimer": DISCLAIMER,
    }


# ------------------------------------------------- shared: index + urgency
def generate_evidence_index(rec: IncidentRecord) -> str:
    lines = ["EVIDENCE INDEX", ""]
    for e in rec.evidence_items:
        lines.append(
            f"{e.id} | {e.kind} | {e.filename or '-'} | quality {e.quality} | "
            f"lang {','.join(e.languages) or '-'}"
        )
        if e.sha256:
            lines.append(f"     sha256 {e.sha256}")
    lines += ["", "TIMELINE", ""]
    for ev in rec.timeline_events:
        lines.append(
            f"{ev.timestamp or 'AMBIGUOUS':<22} {ev.stage:<5} "
            f"{'/'.join(ev.tactics):<28} {ev.description}  "
            f"[{','.join(ev.sources)}]"
        )
    return "\n".join(lines)


def generate_urgent_actions(rec: IncidentRecord) -> str:
    """The golden hour section: what to do first, ranked."""
    debits = sorted(rec.debits(), key=lambda t: t.datetime or "", reverse=True)
    lines = [
        "URGENT ACTIONS",
        "",
        "Freezing outcomes depend on speed. Act on these in order.",
        "",
        "1. Call 1930 now and read the call brief.",
        "2. Send the bank letter to your bank through its official channel.",
        "3. File on cybercrime.gov.in using the NCRP field data.",
        "",
        "FREEZE TARGETS (most recent first: money moves outward from here)",
    ]
    for i, t in enumerate(debits, 1):
        lines.append(
            f"   {i}. {t.payee_upi_id or t.payee_name or NOT_FOUND}  "
            f"{_rupees(t.amount)}  quote ref {t.txn_ref or NOT_FOUND}"
        )
    crit = [f for f in rec.findings if f.severity == "CRITICAL"]
    if crit:
        lines += ["", "ACTIVE DANGER"]
        lines += [f"   - {f.observed}" for f in crit]
    lines += ["", "Do not pay anyone who offers to recover this money for a fee.",
              "No agency asks for payment to return your funds."]
    return "\n".join(lines)


def generate_all(rec: IncidentRecord) -> Dict:
    return {
        "ncrp": generate_ncrp(rec),
        "brief_1930": generate_1930_brief(rec),
        "bank_letter": generate_bank_letter(rec),
        "npci": generate_npci_package(rec),
        "evidence_index": generate_evidence_index(rec),
        "urgent_actions": generate_urgent_actions(rec),
    }
