"""
Classical pre-extraction — runs BEFORE the model, using standard libraries
only. Purpose: pull out everything a library can get reliably, so the model
call has less work to do and less text to read.

This reduces token spend two ways:
  1. The extracted entity list is passed to the model as a compact JSON hint
     block instead of asking it to find every phone number and UPI ID itself
     inside a long noisy transcript.
  2. Evidence is deduplicated and trimmed of clearly irrelevant boilerplate
     (promotional SMS, OTP-only messages) before it reaches the prompt.

The model is still responsible for the parts libraries cannot do: reading
meaning, assigning scam stages, judging tactics, resolving relative dates
against context. Nothing here replaces validators.py — arithmetic and
impossibility checks still run in Python after extraction, regardless of
source.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import phonenumbers
import dateparser
from email_validator import validate_email, EmailNotValidError

UPI_RE = re.compile(r"\b[a-zA-Z0-9._-]{1,64}@[a-zA-Z]{2,32}\b")
URL_RE = re.compile(r"https?://[^\s,\)\]]+")
REF_RE = re.compile(r"\b(?:UTR|RRN|Ref(?:erence)?(?:\s*No\.?)?|Txn\s*ID)"
                    r"\s*[:.]?\s*([A-Za-z0-9]{8,25})\b", re.I)
AMOUNT_RE = re.compile(r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)")
DATE_HINT_RE = re.compile(
    r"\b\d{1,2}[/-][A-Za-z]{3}[/-]\d{2,4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

# boilerplate that never carries case-relevant information: safe to drop
# before the text reaches the model, saving tokens without losing evidence
PROMO_PATTERNS = [
    re.compile(r"is your OTP", re.I),
    re.compile(r"unsubscribe", re.I),
    re.compile(r"\bsale\b.*\boff\b", re.I),
    re.compile(r"congratulations.*won.*lottery", re.I),  # kept as a scam
]                                                          # signal elsewhere,
                                                            # not dropped here


@dataclass
class PreExtracted:
    upi_ids: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    transaction_refs: List[str] = field(default_factory=list)
    amounts: List[float] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    date_hints: List[str] = field(default_factory=list)
    resolved_dates: List[str] = field(default_factory=list)

    def to_hint_block(self) -> str:
        """Compact text block for the model prompt — cheaper than raw scanning."""
        parts = []
        if self.upi_ids:
            parts.append(f"UPI IDs seen: {', '.join(self.upi_ids)}")
        if self.transaction_refs:
            parts.append(f"Transaction references seen: "
                         f"{', '.join(self.transaction_refs)}")
        if self.amounts:
            amt = ", ".join(f"Rs {a:,.0f}" for a in self.amounts)
            parts.append(f"Amounts mentioned: {amt}")
        if self.phone_numbers:
            parts.append(f"Phone numbers seen: {', '.join(self.phone_numbers)}")
        if self.urls:
            parts.append(f"URLs seen: {', '.join(self.urls)}")
        if self.resolved_dates:
            parts.append(f"Dates resolved: {', '.join(self.resolved_dates)}")
        if not parts:
            return ""
        return ("[Pre-extracted by classical tools — verify against the text, "
                "do not re-derive from scratch]\n" + "\n".join(parts))


def extract_phones(text: str, region: str = "IN") -> List[str]:
    out = []
    for m in phonenumbers.PhoneNumberMatcher(text, region):
        out.append(phonenumbers.format_number(
            m.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL))
    return sorted(set(out))


def extract_dates(text: str, reported_on: Optional[str] = None) -> List[str]:
    """Resolve absolute date strings. Relative phrases ('last Tuesday') are
    left to the model, which has reported_on in its prompt context."""
    hints = DATE_HINT_RE.findall(text)
    settings = {"PREFER_DATES_FROM": "past"}
    if reported_on:
        base = dateparser.parse(reported_on)
        if base:
            settings["RELATIVE_BASE"] = base
    out = []
    for h in hints:
        d = dateparser.parse(h, settings=settings)
        if d:
            out.append(d.strftime("%d %b %Y"))
    return sorted(set(out))


def validate_email_address(addr: str) -> Optional[str]:
    try:
        return validate_email(addr, check_deliverability=False).normalized
    except EmailNotValidError:
        return None


def strip_boilerplate(text: str) -> str:
    """Drop lines that are pure promotional noise, to save tokens. Anything
    resembling a scam pretext (lottery, prize) is deliberately kept."""
    keep = []
    for line in text.splitlines():
        if any(p.search(line) for p in PROMO_PATTERNS[:3]):
            continue
        keep.append(line)
    return "\n".join(keep)


def pre_extract(text: str, reported_on: Optional[str] = None) -> PreExtracted:
    upi = sorted(set(UPI_RE.findall(text)))
    urls = sorted(set(URL_RE.findall(text)))
    refs = sorted(set(REF_RE.findall(text)))
    amounts = sorted(set(float(a.replace(",", "")) for a in AMOUNT_RE.findall(text)))
    phones = extract_phones(text)
    date_hints = sorted(set(DATE_HINT_RE.findall(text)))
    resolved = extract_dates(text, reported_on)

    return PreExtracted(
        upi_ids=upi, urls=urls, transaction_refs=refs, amounts=amounts,
        phone_numbers=phones, date_hints=date_hints, resolved_dates=resolved,
    )


def pre_extract_bundle(evidence_items, reported_on: Optional[str] = None
                       ) -> PreExtracted:
    """Run pre-extraction across an entire evidence bundle and merge."""
    combined = PreExtracted()
    seen_upi, seen_ref, seen_phone, seen_url = set(), set(), set(), set()
    amounts = set()
    for item in evidence_items:
        text = strip_boilerplate(item.raw_excerpt) if item.kind == "sms" \
            else item.raw_excerpt
        r = pre_extract(text, reported_on)
        seen_upi.update(r.upi_ids)
        seen_ref.update(r.transaction_refs)
        seen_phone.update(r.phone_numbers)
        seen_url.update(r.urls)
        amounts.update(r.amounts)
        combined.resolved_dates.extend(r.resolved_dates)
    combined.upi_ids = sorted(seen_upi)
    combined.transaction_refs = sorted(seen_ref)
    combined.phone_numbers = sorted(seen_phone)
    combined.urls = sorted(seen_url)
    combined.amounts = sorted(amounts)
    combined.resolved_dates = sorted(set(combined.resolved_dates))
    return combined
