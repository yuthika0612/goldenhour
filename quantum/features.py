"""
IncidentRecord -> fixed-length feature vector.

This is the bridge between the Golden Hour pipeline and any downstream
learning experiment. The features are things the pipeline already computes,
not new instrumentation: transaction structure, timing, stage/tactic profile
and the deterministic findings.

Feature vector (14 dimensions):

  0  n_debits                number of outgoing payments
  1  log_total_out           log1p of gross outflow
  2  escalation_ratio        largest payment / first payment
  3  n_unique_payees         distinct beneficiaries
  4  payee_switch_rate       unique payees / payments  (mule handoff signal)
  5  span_hours              first to last payment, in hours
  6  mean_gap_minutes        mean interval between payments
  7  burstiness              std(gap) / mean(gap)
  8  n_stages                distinct scam stages present
  9  frac_pressure_stages    THREAT+ISOLATION+ESCALATION share of events
 10  frac_trust_stages       TRUST_BUILD+CREDIBILITY share of events
 11  tactic_diversity        distinct persuasion tactics used
 12  n_high_findings         HIGH+CRITICAL deterministic findings
 13  inflow_ratio            money returned / money sent (trust-build tell)
"""
from typing import List
import math
import numpy as np

FEATURE_NAMES = [
    "n_debits", "log_total_out", "escalation_ratio", "n_unique_payees",
    "payee_switch_rate", "span_hours", "mean_gap_minutes", "burstiness",
    "n_stages", "frac_pressure_stages", "frac_trust_stages",
    "tactic_diversity", "n_high_findings", "inflow_ratio",
    # order-aware block
    "pressure_before_demand", "trust_before_demand",
    "escalation_monotonicity", "stage_bigram_entropy",
]
N_FEATURES = len(FEATURE_NAMES)

PRESSURE_STAGES = {"S05", "S06", "S10"}
TRUST_STAGES = {"S02", "S07"}


def _parse_hours(ts: str):
    """Best-effort hour extraction; the experiment tolerates missing times."""
    if not ts:
        return None
    import re
    m = re.search(r"(\d{1,2}):(\d{2})", ts)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if re.search(r"[Pp]\.?[Mm]", ts) and h < 12:
        h += 12
    return h + mi / 60.0


def record_to_vector(rec) -> np.ndarray:
    debits = [t for t in rec.transactions if t.direction == "debit"]
    credits = [t for t in rec.transactions if t.direction == "credit"]
    amounts = [t.amount or 0.0 for t in debits]

    n_debits = float(len(debits))
    total_out = float(sum(amounts))
    escalation = (max(amounts) / amounts[0]) if amounts and amounts[0] else 0.0

    payees = [t.payee_upi_id or t.payee_name for t in debits
              if (t.payee_upi_id or t.payee_name)]
    n_payees = float(len(set(payees)))
    switch_rate = n_payees / n_debits if n_debits else 0.0

    hours = [h for h in (_parse_hours(t.datetime) for t in debits) if h is not None]
    span = (max(hours) - min(hours)) if len(hours) >= 2 else 0.0
    gaps = [abs(b - a) * 60 for a, b in zip(sorted(hours), sorted(hours)[1:])]
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    burst = float(np.std(gaps) / mean_gap) if gaps and mean_gap > 0 else 0.0

    stages = [e.stage for e in rec.timeline_events if e.stage]
    n_stages = float(len(set(stages)))
    n_ev = max(len(stages), 1)
    frac_pressure = sum(s in PRESSURE_STAGES for s in stages) / n_ev
    frac_trust = sum(s in TRUST_STAGES for s in stages) / n_ev

    tactics = {t for e in rec.timeline_events for t in (e.tactics or [])}
    n_high = float(sum(1 for f in rec.findings
                       if f.severity in ("HIGH", "CRITICAL")))

    total_in = float(sum(t.amount or 0.0 for t in credits))
    inflow_ratio = (total_in / total_out) if total_out > 0 else 0.0

    # ---- order-aware block: aggregates above cannot see sequence ----
    first_demand = next((i for i, s in enumerate(stages) if s == "S08"), None)
    if first_demand is None or not stages:
        pressure_before = trust_before = 0.0
    else:
        pre = stages[:first_demand]
        n_pre = max(len(pre), 1)
        pressure_before = sum(s in PRESSURE_STAGES for s in pre) / n_pre
        trust_before = sum(s in TRUST_STAGES for s in pre) / n_pre

    if len(amounts) >= 2:
        rises = sum(b > a for a, b in zip(amounts, amounts[1:]))
        monotonic = rises / (len(amounts) - 1)
    else:
        monotonic = 0.0

    if len(stages) >= 2:
        from collections import Counter
        bigrams = Counter(zip(stages, stages[1:]))
        tot_b = sum(bigrams.values())
        probs = np.array([c / tot_b for c in bigrams.values()])
        bigram_entropy = float(-(probs * np.log2(probs)).sum())
    else:
        bigram_entropy = 0.0

    return np.array([
        n_debits, math.log1p(total_out), escalation, n_payees, switch_rate,
        span, mean_gap, burst, n_stages, frac_pressure, frac_trust,
        float(len(tactics)), n_high, inflow_ratio,
        pressure_before, trust_before, monotonic, bigram_entropy,
    ], dtype=float)


def batch(records: List) -> np.ndarray:
    return np.vstack([record_to_vector(r) for r in records])
