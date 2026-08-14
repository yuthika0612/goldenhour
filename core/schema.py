"""
Unified Incident Record — the single normalised object that all four
outputs (NCRP / 1930 / Bank / NPCI) are generated from.

Field names follow the project formats document section 5.
Everything is Optional: real evidence is incomplete, and a missing field
must survive to the output as an explicit gap, never as a guess.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json

NOT_FOUND = "NOT FOUND IN EVIDENCE"


@dataclass
class Transaction:
    """One money movement, corroborated by one or more evidence items."""
    txn_ref: Optional[str] = None          # UTR / RRN / transaction id
    amount: Optional[float] = None
    direction: str = "debit"               # debit | credit
    datetime: Optional[str] = None         # ISO-ish string as found
    payer_upi_id: Optional[str] = None
    payee_upi_id: Optional[str] = None
    payee_name: Optional[str] = None
    bank_wallet_merchant: Optional[str] = None
    beneficiary_bank_ifsc: Optional[str] = None
    balance_after: Optional[float] = None  # from bank SMS, drives arithmetic
    sources: List[str] = field(default_factory=list)   # evidence ids E1..
    confidence: str = "MEDIUM"             # HIGH | MEDIUM | LOW
    note: Optional[str] = None


@dataclass
class TimelineEvent:
    timestamp: Optional[str] = None
    stage: str = ""                        # S01..S16
    tactics: List[str] = field(default_factory=list)
    description: str = ""
    sources: List[str] = field(default_factory=list)


@dataclass
class Finding:
    """A flagged problem: contradiction, impossibility, or tamper signal."""
    id: str = ""
    severity: str = "MEDIUM"               # CRITICAL | HIGH | MEDIUM | LOW
    check: str = ""                        # I1..I8, T1..T7, ARITHMETIC, GAP
    observed: str = ""
    evidence: List[str] = field(default_factory=list)
    why_it_matters: str = ""
    resolves_with: str = ""
    origin: str = "rules"                  # rules | model


@dataclass
class EvidenceItem:
    id: str = ""                           # E1, E2...
    kind: str = ""                         # chat | screenshot_ocr | sms | note
    filename: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    quality: str = "CLEAN"                 # CLEAN | DEGRADED | UNREADABLE
    sha256: Optional[str] = None
    raw_excerpt: str = ""


@dataclass
class IncidentRecord:
    # --- complainant (usually supplied by the user, not the evidence) ---
    victim_name: Optional[str] = None
    victim_mobile: Optional[str] = None
    victim_email: Optional[str] = None
    victim_address: Optional[str] = None
    victim_account_no: Optional[str] = None
    victim_bank_branch: Optional[str] = None
    user_role: str = "UNCLEAR"             # PAYER_VICTIM | PAYEE_VICTIM |
                                           # THIRD_PARTY | UNCLEAR
    # --- incident core ---
    incident_datetime: Optional[str] = None
    incident_window_end: Optional[str] = None
    incident_description: str = ""
    fraud_type: Optional[str] = None
    fraud_type_basis: str = ""
    fraud_likelihood: str = "NOT_ASSESSED"   # LIKELY_FRAUD | POSSIBLE_FRAUD |
                                             # NO_INDICATORS_FOUND
    total_fraud_amount: Optional[float] = None
    total_recovered_inflow: Optional[float] = None
    net_loss: Optional[float] = None
    bank_wallet_merchant: Optional[str] = None

    # --- suspect artifacts ---
    suspect_mobiles: List[str] = field(default_factory=list)
    suspect_emails: List[str] = field(default_factory=list)
    suspect_upi_ids: List[str] = field(default_factory=list)
    suspect_bank_accounts: List[str] = field(default_factory=list)
    suspect_urls_social_handles: List[str] = field(default_factory=list)
    suspect_aliases: List[str] = field(default_factory=list)
    suspect_address: Optional[str] = None

    # --- references after filing ---
    ncrp_acknowledgement_no: Optional[str] = None
    bank_complaint_reference: Optional[str] = None

    # --- structured content ---
    transactions: List[Transaction] = field(default_factory=list)
    timeline_events: List[TimelineEvent] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    evidence_items: List[EvidenceItem] = field(default_factory=list)

    # --- audit ---
    inferred_fields: Dict[str, str] = field(default_factory=dict)
    ocr_corrections: List[Dict[str, str]] = field(default_factory=list)
    missing_evidence: List[Dict[str, str]] = field(default_factory=list)
    weakest_link: str = ""

    # ---------- helpers ----------
    def debits(self) -> List[Transaction]:
        return [t for t in self.transactions if t.direction == "debit"]

    def credits(self) -> List[Transaction]:
        return [t for t in self.transactions if t.direction == "credit"]

    def compute_totals(self) -> None:
        out = sum(t.amount or 0 for t in self.debits())
        inn = sum(t.amount or 0 for t in self.credits())
        self.total_fraud_amount = out
        self.total_recovered_inflow = inn
        self.net_loss = out - inn

    def value_or_missing(self, attr: str) -> str:
        v = getattr(self, attr, None)
        if v in (None, "", [], 0):
            return NOT_FOUND
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IncidentRecord":
        rec = IncidentRecord()
        for k, v in d.items():
            if not hasattr(rec, k):
                continue
            if k == "transactions":
                rec.transactions = [Transaction(**_filter(Transaction, x)) for x in v]
            elif k == "timeline_events":
                rec.timeline_events = [TimelineEvent(**_filter(TimelineEvent, x)) for x in v]
            elif k == "findings":
                rec.findings = [Finding(**_filter(Finding, x)) for x in v]
            elif k == "evidence_items":
                rec.evidence_items = [EvidenceItem(**_filter(EvidenceItem, x)) for x in v]
            else:
                setattr(rec, k, v)
        return rec


def _filter(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unexpected keys so a chatty model response cannot crash parsing."""
    allowed = set(cls.__dataclass_fields__.keys())
    return {k: v for k, v in (d or {}).items() if k in allowed}
