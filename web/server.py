"""
Golden Hour web app.

    pip install -r requirements.txt
    export GEMINI_API_KEY=...
    python web/server.py
    open http://127.0.0.1:8000

Privacy: nothing is written to disk except a transient temp file per image,
deleted before the response is sent.

DESIGN RULE: there is no user-facing "mode". The pipeline always tries every
available source of signal and degrades silently and automatically:

    each image  -> local OCR first
                -> if OCR yields nothing usable, that image (not the whole
                   batch) is queued for the model's own vision instead
    OCR text + pasted text + hint block  -> the single model call
    model unavailable or the call fails  -> the deterministic rules engine
                   (bank-SMS parsing, arithmetic, format checks) still runs,
                   and a complaint is still produced from whatever could be
                   established, with gaps marked rather than blocked on

A complaint is ALWAYS generated. The system's job is to get as far as the
evidence allows and say plainly what it could not establish -- never to stop
and ask the victim to pick a technical option.
"""
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

from core import preprocess, extractor, validators, generators, preextract
from core import rulebased_extract as rbx
from core.schema import IncidentRecord, Finding, EvidenceItem

app = FastAPI(title="Golden Hour")
INDEX = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
IST = ZoneInfo("Asia/Kolkata")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/health")
def health():
    return {"ok": True, "model_configured": bool(os.environ.get("GEMINI_API_KEY"))}


@app.get("/today")
def today():
    """IST 'now', for the form to pre-fill the incident date. The person can
    still change it -- this is a default, not a constraint."""
    now = datetime.now(IST)
    return {"date": now.strftime("%d %B %Y"), "iso": now.isoformat()}


def _ocr_each_image(image_bytes_list: List[bytes], start_id: int):
    """
    OCR-first, per image. Returns (evidence_items, unresolved_image_bytes,
    tmp_paths). An image only falls through to unresolved -- for the
    model's own vision later -- if local OCR could not read it at all.
    """
    items, unresolved, tmp_paths = [], [], []
    for i, data in enumerate(image_bytes_list):
        eid = f"E{start_id + i}"
        fh = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fh.write(data)
        fh.close()
        tmp_paths.append(fh.name)
        try:
            item = preprocess.ocr_image(fh.name, eid)
        except Exception:
            item = EvidenceItem(id=eid, kind="screenshot_ocr",
                               filename="screenshot", quality="UNREADABLE",
                               raw_excerpt="[OCR raised an error]")
        if item.quality == "UNREADABLE" or not item.raw_excerpt.strip():
            unresolved.append((eid, data))
        else:
            items.append(item)
    return items, unresolved, tmp_paths


def _run_model(items, hint_block, reported_on, unresolved_images):
    """
    The one model call. Unreadable images ride along as native image parts
    so the model sees exactly the screenshots OCR could not read -- not all
    of them, only the ones that actually needed it.
    """
    image_bytes = [b for _, b in unresolved_images]
    data = extractor.call_gemini(
        extractor.build_evidence_block(items, hint_block),
        reported_on=reported_on or None, image_bytes=image_bytes)
    return extractor.record_from_model(data, items)


@app.post("/analyse")
async def analyse(
    chat: str = Form(""),
    sms: str = Form(""),
    note: str = Form(""),
    reported_on: str = Form(""),
    victim_name: str = Form(""),
    victim_mobile: str = Form(""),
    victim_email: str = Form(""),
    victim_address: str = Form(""),
    images: Optional[List[UploadFile]] = File(None),
):
    tmp_paths = []
    degradation_notes = []
    seal_info, fp = None, None
    try:
        image_bytes_list = []
        for up in (images or []):
            data = await up.read()
            if data:
                image_bytes_list.append(data)

        text_items = preprocess.ingest(chat=chat, sms=sms, note=note)
        img_items, unresolved, tmp_paths = _ocr_each_image(
            image_bytes_list, start_id=len(text_items) + 1)
        items = text_items + img_items      # unresolved images are NOT
                                            # in here; the model reads them
                                            # directly, and any recovered
                                            # content is folded back in

        if not items and not unresolved:
            return JSONResponse(
                {"error": "No evidence supplied. Add at least a chat "
                          "export, bank SMS, a note, or a screenshot."},
                status_code=400)

        if unresolved:
            degradation_notes.append(
                f"{len(unresolved)} screenshot(s) could not be read by "
                f"local text recognition and were sent to the AI model "
                f"directly instead.")

        rule_txns = rbx.extract_from_sms(items)
        handles = rbx.extract_handles(items)
        have_key = bool(os.environ.get("GEMINI_API_KEY"))

        rec = None
        if have_key:
            try:
                pre = preextract.pre_extract_bundle(items, reported_on or None)
                rec = _run_model(items, pre.to_hint_block(), reported_on,
                                 unresolved)
                for n in rbx.cross_check(rec.transactions, rule_txns):
                    rec.findings.append(Finding(
                        severity="HIGH", check="CROSSCHECK", observed=n,
                        origin="rules",
                        why_it_matters="Extraction paths disagree on a "
                                       "bank-sourced figure.",
                        resolves_with="Bank statement."))
                rec.transactions = rbx.merge(rec.transactions, rule_txns)
                for u in pre.urls:
                    if u not in rec.suspect_urls_social_handles:
                        rec.suspect_urls_social_handles.append(u)
                for eid, _ in unresolved:
                    rec.evidence_items.append(EvidenceItem(
                        id=eid, kind="screenshot_ocr", filename="screenshot",
                        quality="CLEAN",
                        raw_excerpt="[Read directly by the AI model]"))
            except Exception:
                degradation_notes.append(
                    "The AI reading step could not complete, so this "
                    "complaint is built from bank-SMS parsing and format "
                    "checks only. Conversation content and any unresolved "
                    "screenshots are not reflected below.")
                rec = None
        else:
            degradation_notes.append(
                "No AI model is configured for this deployment, so this "
                "complaint is built from bank-SMS parsing and format "
                "checks only. Conversation content and screenshots that "
                "could not be read locally are not reflected below.")

        if rec is None:
            # ALWAYS produce something: rules-only reconstruction.
            rec = IncidentRecord()
            rec.evidence_items = items + [
                EvidenceItem(id=eid, kind="screenshot_ocr",
                            filename="screenshot", quality="UNREADABLE",
                            raw_excerpt="[Not read: no AI model available]")
                for eid, _ in unresolved
            ]
            rec.transactions = rule_txns
            rec.user_role = "PAYER_VICTIM" if rule_txns else "UNCLEAR"

        rec.weakest_link = rec.weakest_link or " ".join(degradation_notes)

        rec.victim_name = victim_name or None
        rec.victim_mobile = victim_mobile or None
        rec.victim_email = (preextract.validate_email_address(victim_email)
                            or victim_email or None)
        rec.victim_address = victim_address or None
        rec.victim_account_no = rec.victim_account_no or rbx.extract_account_no(items)
        for h in handles["upi_ids"]:
            if h not in rec.suspect_upi_ids:
                rec.suspect_upi_ids.append(h)
        for u in handles["urls"]:
            if u not in rec.suspect_urls_social_handles:
                rec.suspect_urls_social_handles.append(u)
        if not rec.incident_datetime:
            rec.incident_datetime = (rec.transactions[0].datetime
                                     if rec.transactions else
                                     (reported_on or None))
        if not rec.incident_window_end and rec.transactions:
            rec.incident_window_end = rec.transactions[-1].datetime
        if not rec.bank_wallet_merchant:
            banks = [t.bank_wallet_merchant for t in rec.transactions
                     if t.bank_wallet_merchant]
            rec.bank_wallet_merchant = banks[0] if banks else None

        rec.compute_totals()
        rec.incident_description = generators.build_narrative(rec)
        validators.run_all(rec)

        # Quantum layer: seal + near-duplicate fingerprint. Runs regardless
        # of how much of the record above could be established.
        try:
            from core.evidence_seal import MerkleSignatureScheme, seal_bundle
            from core.fingerprint import fingerprint as make_fingerprint
            signer = MerkleSignatureScheme(height=3)
            seal = seal_bundle(rec.evidence_items,
                               case_id=f"GH-{datetime.now(IST):%Y%m%d%H%M%S}",
                               signer=signer)
            seal_info = {"merkle_root": seal.merkle_root,
                        "sealed_at": seal.sealed_at,
                        "n_items": seal.n_items}
            all_text = "\n".join(e.raw_excerpt for e in rec.evidence_items)
            fp = make_fingerprint(all_text).tolist() if all_text.strip() else None
        except Exception:
            pass

        outputs = generators.generate_all(rec)

        # Additional suggestions, shown plainly alongside the complaint --
        # never a technical mode, never a block on generating it.
        suggestions = list(degradation_notes)
        for m in rec.missing_evidence[:5]:
            if isinstance(m, dict) and m.get("item"):
                suggestions.append(f"{m['item']}: {m.get('why', '')}".strip(": "))
        high = [f for f in rec.findings if f.severity in ("HIGH", "CRITICAL")]
        if high:
            suggestions.append(
                f"{len(high)} inconsistency(ies) found in the evidence -- "
                f"see the flags below before filing.")

        return JSONResponse({
            "summary": {
                "user_role": rec.user_role,
                "fraud_type": rec.fraud_type,
                "fraud_type_basis": rec.fraud_type_basis,
                "fraud_likelihood": rec.fraud_likelihood,
                "gross_out": rec.total_fraud_amount,
                "inflow": rec.total_recovered_inflow,
                "net_loss": rec.net_loss,
                "n_transactions": len(rec.transactions),
                "n_events": len(rec.timeline_events),
                "n_evidence": len(rec.evidence_items),
            },
            "suggestions": suggestions,
            "findings": [f.__dict__ for f in rec.findings],
            "timeline": [e.__dict__ for e in rec.timeline_events],
            "transactions": [t.__dict__ for t in rec.transactions],
            "outputs": outputs,
            "seal": seal_info,
            "fingerprint_dims": len(fp) if fp else 0,
            "record": rec.to_json(),
        })
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
