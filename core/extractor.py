"""
Evidence -> IncidentRecord.

Division of labour, and this is the defensible design claim of the project:

  MODEL (Gemini)  reads messy multilingual text, resolves who said what,
                  assigns scam stages and psychological tactics, and pulls
                  entities out of noisy OCR.
  CODE (Python)   validates every number, every format and every
                  contradiction, and writes the four outputs.

The model is never asked to compute a total or to decide whether a balance
sequence is possible. Those failed in prompt-only testing and are exact in
code.
"""
import json
import os
import re
from typing import List, Optional

from .schema import IncidentRecord, EvidenceItem

MODEL_NAME = os.environ.get("GOLDENHOUR_MODEL", "gemini-3.5-flash")

STAGES = """S01 CONTACT | S02 CREDIBILITY_PROP | S03 AUTHORITY_CLAIM |
S04 PRETEXT | S05 THREAT | S06 ISOLATION | S07 TRUST_BUILD |
S08 PAYMENT_DEMAND | S09 PAYMENT_EXECUTED | S10 ESCALATION |
S11 MULE_HANDOFF | S12 CHANNEL_SHIFT | S13 GHOSTING | S14 REALISATION |
S15 REVICTIMISATION | S16 REPORTING_ACTION"""

TACTICS = """AUTHORITY | FEAR | URGENCY | SCARCITY | ISOLATION | RECIPROCITY |
SOCIAL_PROOF | COMMITMENT_ESCALATION | LIKING | CONFUSION_OVERLOAD |
FALSE_LEGITIMACY"""

SYSTEM_PROMPT = f"""You are the extraction stage of Golden Hour, a tool that
helps victims of digital financial fraud in India assemble evidence for a
police complaint. You read messy evidence and return structured JSON. You do
not write advice, narrative or commentary.

EVIDENCE DISCIPLINE
- Every value must come from the evidence. Never infer a value to fill a
  field. Use null for anything not present.
- If text is garbled by OCR, put the raw string in "raw" and your best
  reading in "normalised", and set confidence to LOW. Never silently repair
  digits inside a transaction reference; if you correct one, record it in
  ocr_corrections.
- The evidence may mix English, Hindi, Telugu and romanised transliteration.
  Read all of them. Write descriptions in English.

DEDUPLICATION
- The same payment usually appears in a chat message, a screenshot and a
  bank SMS. That is ONE transaction with three sources, not three
  transactions.
- A victim message saying "sent" or "paid" is NOT a separate payment event.
  Attach it to the payment it refers to.

ROLE
Decide user_role from the evidence:
  PAYER_VICTIM  the user sent money
  PAYEE_VICTIM  the user was shown proof of a payment that was supposedly
                sent TO them (fake payment screenshot)
  THIRD_PARTY   someone is reporting on behalf of the victim
  UNCLEAR       insufficient evidence

DO NOT COMPUTE
Do not total amounts, do not compute net loss, and do not judge whether a
balance sequence is arithmetically possible. Downstream code does that
exactly. Just report each transaction faithfully, including balance_after
when a bank message states a balance.

A block labelled "[Pre-extracted by classical tools]" may appear before the
evidence. It lists phone numbers, UPI IDs, references, amounts and dates
already found by regex/library extraction. Treat it as a checklist to verify
against the text, not as ground truth to copy blindly -- a regex can
false-positive on something that only looks like a UPI ID. Cross-check each
entry against its source before including it in your output.

STAGES: {STAGES}
TACTICS: {TACTICS}

Return ONLY a JSON object with exactly these keys:

{{
  "user_role": "...",
  "fraud_type": "Digital Arrest | UPI Fraud | Phishing | Smishing | Job or Task Scam | Investment Scam | Fake Payment Proof | Loan App | Recovery Scam | Other",
  "fraud_type_basis": "one sentence citing what in the evidence supports this",
  "incident_datetime": "first event, as found in evidence, or null",
  "incident_window_end": "last event, or null",
  "bank_wallet_merchant": "the user's own bank or wallet, or null",
  "victim_account_no": "user's own account as shown in bank messages, or null",
  "suspect_mobiles": [], "suspect_emails": [], "suspect_upi_ids": [],
  "suspect_bank_accounts": [], "suspect_urls_social_handles": [],
  "suspect_aliases": ["names the suspect claimed, never treated as identified"],
  "transactions": [
    {{"txn_ref": "...", "amount": 50000, "direction": "debit|credit",
      "datetime": "...", "payer_upi_id": null, "payee_upi_id": "...",
      "payee_name": null, "bank_wallet_merchant": "...",
      "balance_after": 12340, "sources": ["E1","E3"],
      "confidence": "HIGH|MEDIUM|LOW", "note": null}}
  ],
  "timeline_events": [
    {{"timestamp": "...", "stage": "S03", "tactics": ["AUTHORITY","FEAR"],
      "description": "plain factual sentence", "sources": ["E1"]}}
  ],
  "model_findings": [
    {{"severity": "CRITICAL|HIGH|MEDIUM", "check": "I1..I8|T1..T7|OTHER",
      "observed": "...", "evidence": ["E1"], "why_it_matters": "one sentence",
      "resolves_with": "the document or step that would settle it"}}
  ],
  "unrelated_items": ["one line per evidence item that is not part of this incident"],
  "ocr_corrections": [{{"raw": "...", "corrected": "...", "where": "E2"}}],
  "inferred_fields": {{"field": "basis for the inference"}},
  "missing_evidence": [{{"item": "...", "why": "...", "how_to_get": "..."}}],
  "weakest_link": "the single weakest point in this reconstruction"
}}

IMPOSSIBILITY CHECKS for model_findings. Report only those actually present,
and name the ID:
I1 arrest, custody or interrogation conducted over a call. No such procedure
   exists in Indian law.
I2 any official body demanding funds for verification, clearance, refundable
   security or case closure.
I3 funds requested to a personal UPI handle while claiming to be an official
   collection.
I4 a promise that transferred money is automatically refundable.
I5 a fee demanded to recover money already lost.
I6 a payment claimed with no corresponding bank record.
I8 instructions to install remote access or screen sharing software, or to
   keep the matter secret from family, bank or police.

TAMPER INDICATORS for model_findings:
T3 spelling or wording errors in what should be app generated text.
T4 a field missing that the app always shows.
T5 a screenshot timestamp inconsistent with its own transaction or the chat.
T7 a chat export showing signs of editing.

Report findings, never the absence of findings. Do not include an entry that
says nothing is wrong.
"""


# --------------------------------------------------------------- packaging
def build_evidence_block(items: List[EvidenceItem],
                         hint_block: str = "") -> str:
    """
    Evidence text plus an optional classical pre-extraction hint block.
    Putting verified entities up front lets the model spend its attention on
    meaning, stages and tactics rather than re-scanning for every phone
    number and UPI ID -- fewer output tokens spent restating what a regex
    already found reliably, same JSON schema either way.
    """
    parts = []
    if hint_block:
        parts.append(hint_block)
    for e in items:
        parts.append(
            f"--- {e.id} | kind={e.kind} | file={e.filename or '-'} ---\n"
            f"{e.raw_excerpt}\n--- end {e.id} ---"
        )
    return "\n\n".join(parts)


def _strip_fences(text: str) -> str:
    text = re.sub(r"^\s*```(?:json)?", "", text.strip())
    text = re.sub(r"```\s*$", "", text.strip())
    return text.strip()


def _first_json_object(text: str) -> str:
    """Salvage a JSON object even if the model wrapped it in prose."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unterminated JSON object in model response")


def parse_model_json(text: str) -> dict:
    try:
        return json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        return json.loads(_first_json_object(text))


# ------------------------------------------------------------- model calls
def call_gemini(evidence_text: str, api_key: Optional[str] = None,
                reported_on: Optional[str] = None,
                image_bytes: Optional[List[bytes]] = None) -> dict:
    """
    Single extraction call, using the current unified Google Gen AI SDK
    (`google-genai`, `from google import genai`).

    NOTE ON SDK HISTORY: this file previously used the `google-generativeai`
    package (`import google.generativeai as genai`). Google has deprecated
    that package in favour of this one. If you see import errors mentioning
    `google.generativeai`, the environment has the OLD package installed
    instead of `google-genai` -- run `pip install -U google-genai` (and
    `pip uninstall google-generativeai` if both are present).

    Screenshots are sent as native image parts alongside the text, using
    the model's own vision -- not a local OCR binary. Local OCR is still
    tried first (see web/server.py:_ocr_each_image); this path is only for
    images OCR could not read.

    Raises RuntimeError with the ORIGINAL error message attached on
    failure. Callers must not swallow this silently -- a stale model name,
    a bad key, or a quota error becomes invisible and looks identical to
    "the AI reading step could not complete" with no way to diagnose it.
    """
    from google import genai
    from google.genai import types
    from PIL import Image
    import io

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in your environment (locally) "
            "or in your host's dashboard (e.g. Render's Environment tab)."
        )

    client = genai.Client(api_key=key)

    prompt = evidence_text
    if reported_on:
        prompt = (f"Reported on: {reported_on}. Treat this as the present date "
                  f"when resolving relative dates.\n\n{evidence_text}")

    contents: List = [prompt]
    for b in (image_bytes or []):
        try:
            contents.append(Image.open(io.BytesIO(b)))
        except Exception:
            continue      # a corrupt upload should not fail the whole call

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    except Exception as exc:
        # NOT swallowed here on purpose. web/server.py catches this at a
        # higher level and degrades gracefully for the user-facing output,
        # but the real cause must survive in the message for logs -- a
        # generic "could not complete" with no detail is exactly what makes
        # a stale MODEL_NAME or SDK/version mismatch invisible.
        raise RuntimeError(
            f"Gemini API call failed (model={MODEL_NAME}): {exc}") from exc

    if not resp.text:
        reason = "?"
        if resp.candidates:
            reason = getattr(resp.candidates[0], "finish_reason", "?")
        raise RuntimeError(
            f"Gemini API returned an empty response (model={MODEL_NAME}, "
            f"finish_reason={reason}).")

    return parse_model_json(resp.text)


def record_from_model(data: dict, items: List[EvidenceItem]) -> IncidentRecord:
    """Map the model's JSON onto the unified record, then attach evidence."""
    from .schema import Transaction, TimelineEvent, Finding, _filter

    rec = IncidentRecord()
    for k in ("user_role", "fraud_type", "fraud_type_basis", "incident_datetime",
              "incident_window_end", "bank_wallet_merchant", "victim_account_no",
              "suspect_mobiles", "suspect_emails", "suspect_upi_ids",
              "suspect_bank_accounts", "suspect_urls_social_handles",
              "suspect_aliases", "inferred_fields", "ocr_corrections",
              "missing_evidence", "weakest_link"):
        if data.get(k) is not None:
            setattr(rec, k, data[k])

    rec.transactions = [Transaction(**_filter(Transaction, t))
                        for t in data.get("transactions") or []]
    rec.timeline_events = [TimelineEvent(**_filter(TimelineEvent, e))
                           for e in data.get("timeline_events") or []]
    rec.findings = [Finding(**{**_filter(Finding, f), "origin": "model"})
                    for f in data.get("model_findings") or []]
    rec.evidence_items = items
    rec.compute_totals()
    return rec
