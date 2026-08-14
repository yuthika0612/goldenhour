#!/usr/bin/env python3
"""
Golden Hour pipeline.

    python pipeline.py --case cases/case1.txt              # uses Gemini
    python pipeline.py --case cases/case1.txt --offline    # rules only
    python pipeline.py --case cases/case1.txt --out out/   # write files

A case file is a plain text bundle with sections marked:
    ###CHAT / ###SMS / ###OCR / ###NOTE / ###REPORTED_ON
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core import (preprocess, extractor, validators, generators,   # noqa: E402
                  rulebased_extract as rbx, preextract)
from core.schema import IncidentRecord                            # noqa: E402

SECTIONS = ("CHAT", "SMS", "OCR", "NOTE", "REPORTED_ON")


def split_case_file(text: str) -> dict:
    """Split a bundle on ###SECTION markers. OCR may repeat."""
    parts, current, buf = {"OCR": []}, None, []

    def flush():
        if current is None:
            return
        body = "\n".join(buf).strip()
        if current == "OCR":
            parts["OCR"].append(body)
        else:
            parts[current] = body

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("###"):
            name = stripped.lstrip("#").strip().upper()
            if name in SECTIONS:
                flush()
                current, buf = name, []
                continue
        buf.append(line)
    flush()
    return parts


def run(case_text: str, offline: bool = False, api_key: str = None
        ) -> IncidentRecord:
    parts = split_case_file(case_text)

    items = preprocess.ingest(
        chat=parts.get("CHAT", ""),
        sms=parts.get("SMS", ""),
        note=parts.get("NOTE", ""),
        ocr_texts=parts.get("OCR", []),
    )
    if not items:
        raise SystemExit("No evidence found. Check the ### section markers.")

    # Deterministic layer always runs: bank alerts are formulaic, and the
    # balance figures that drive the arithmetic checks must not depend on
    # model behaviour.
    rule_txns = rbx.extract_from_sms(items)
    handles = rbx.extract_handles(items)

    if offline:
        rec = IncidentRecord()
        rec.evidence_items = items
        rec.transactions = rule_txns
        rec.user_role = "PAYER_VICTIM" if rule_txns else "UNCLEAR"
        rec.weakest_link = ("Offline mode: only bank alerts were parsed. Chat "
                            "and screenshot content was not interpreted, so "
                            "stages, tactics and narrative are absent.")
    else:
        # classical pre-extraction FIRST, so the single model call gets a
        # verified hint block instead of scanning raw text for everything
        pre = preextract.pre_extract_bundle(items, parts.get("REPORTED_ON"))
        block = extractor.build_evidence_block(items, pre.to_hint_block())
        data = extractor.call_gemini(block, api_key=api_key,
                                     reported_on=parts.get("REPORTED_ON"))
        rec = extractor.record_from_model(data, items)
        for u in pre.urls:               # classical extraction backstops
            if u not in rec.suspect_urls_social_handles:
                rec.suspect_urls_social_handles.append(u)
        for note in rbx.cross_check(rec.transactions, rule_txns):
            rec.findings.append(__import__("core.schema", fromlist=["Finding"])
                                .Finding(severity="HIGH", check="CROSSCHECK",
                                         observed=note, origin="rules",
                                         why_it_matters="The two extraction "
                                         "paths disagree on a bank-sourced "
                                         "figure.",
                                         resolves_with="Bank statement."))
        rec.transactions = rbx.merge(rec.transactions, rule_txns)

    # backfill identifiers found deterministically
    rec.victim_account_no = rec.victim_account_no or rbx.extract_account_no(items)
    for h in handles["upi_ids"]:
        if h not in rec.suspect_upi_ids:
            rec.suspect_upi_ids.append(h)
    for u in handles["urls"]:
        if u not in rec.suspect_urls_social_handles:
            rec.suspect_urls_social_handles.append(u)
    if not rec.bank_wallet_merchant:
        banks = [t.bank_wallet_merchant for t in rec.transactions
                 if t.bank_wallet_merchant]
        rec.bank_wallet_merchant = banks[0] if banks else None
    if not rec.incident_datetime and rec.transactions:
        rec.incident_datetime = rec.transactions[0].datetime

    rec.compute_totals()
    rec.incident_description = generators.build_narrative(rec)
    validators.run_all(rec)          # deterministic checks, always run
    return rec


def summarise(rec: IncidentRecord) -> str:
    by_sev = {}
    for f in rec.findings:
        by_sev.setdefault(f.severity, []).append(f)
    lines = [
        "=" * 68,
        f"ROLE {rec.user_role}   TYPE {rec.fraud_type or '-'}",
        f"OUT  Rs {rec.total_fraud_amount or 0:,.0f}    "
        f"IN Rs {rec.total_recovered_inflow or 0:,.0f}    "
        f"NET LOSS Rs {rec.net_loss or 0:,.0f}",
        f"{len(rec.transactions)} transactions   "
        f"{len(rec.timeline_events)} timeline events   "
        f"{len(rec.evidence_items)} evidence items",
        "=" * 68, "",
        "FINDINGS",
    ]
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        for f in by_sev.get(sev, []):
            tag = "rule" if f.origin == "rules" else "model"
            lines.append(f"  [{sev:<8}] ({tag}) {f.check}: {f.observed}")
    if not rec.findings:
        lines.append("  none")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Golden Hour fraud dossier builder")
    ap.add_argument("--case", required=True, help="path to a case bundle file")
    ap.add_argument("--offline", action="store_true",
                    help="skip the model, run deterministic checks only")
    ap.add_argument("--out", help="directory to write generated outputs")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"))
    args = ap.parse_args()

    text = Path(args.case).read_text(encoding="utf-8")
    rec = run(text, offline=args.offline, api_key=args.api_key)

    print(summarise(rec))
    outputs = generators.generate_all(rec)
    print("\n" + outputs["urgent_actions"])

    if args.out:
        d = Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "incident_record.json").write_text(rec.to_json(), encoding="utf-8")
        (d / "ncrp_complaint.json").write_text(
            json.dumps(outputs["ncrp"], indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "npci_package.json").write_text(
            json.dumps(outputs["npci"], indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "1930_call_brief.txt").write_text(outputs["brief_1930"], encoding="utf-8")
        (d / "bank_letter.txt").write_text(outputs["bank_letter"], encoding="utf-8")
        (d / "evidence_index.txt").write_text(outputs["evidence_index"], encoding="utf-8")
        (d / "urgent_actions.txt").write_text(outputs["urgent_actions"], encoding="utf-8")
        print(f"\nWrote 7 files to {d}/")


if __name__ == "__main__":
    main()
