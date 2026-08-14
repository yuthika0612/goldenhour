# Golden Hour

Turns a fraud victim's raw evidence into a filing-ready complaint package.

**One evidence upload → one normalised incident record → four outputs**
(NCRP field data, 1930 call brief, bank complaint letter, NPCI/UPI package),
plus a shared evidence index, timeline, and an urgent-actions sheet.

---

## Why this exists

Defrauded money is recoverable, but the window is short — funds move through
mule accounts within hours. At exactly that moment the victim has to assemble
transaction IDs, UPI IDs, timestamps and screenshots into four different
complaints while panicking. The delay, not the detection, is what loses the
money.

This tool does the assembly.

---

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env    # then edit .env and paste your key
export $(cat .env | xargs)   # or use python-dotenv / your shell's own method
python web/server.py
# open http://127.0.0.1:8000  and press "Load a sample case"
```

## Deploy on Render

1. Push this repo to GitHub. `.gitignore` already excludes `.env` — the key
   is never committed.
2. In Render: **New → Blueprint**, connect the repo. Render reads
   `render.yaml` automatically and creates the web service.
3. Open the service's **Environment** tab and add `GEMINI_API_KEY` with your
   key from Google AI Studio. This is the ONLY place the real key exists —
   not in the repo, not in `render.yaml` (which only declares the variable
   name via `sync: false`), not in any committed file.
4. Deploy. The app binds `0.0.0.0:$PORT`, which is what Render expects.

Without a key set, `/analyse` automatically falls back to checks-only mode
(no crash) — see `GEMINI_API_KEY` handling in `web/server.py`.

Command line:

```bash
python pipeline.py --case cases/case1.txt              # with the model
python pipeline.py --case cases/case1.txt --offline    # checks only, no key
python pipeline.py --case cases/case1.txt --out out/   # writes 7 files
```

Tests (no API key needed — the model response is mocked):

```bash
python -m tests.test_pipeline
```

Optional OCR for screenshot images: `pip install pytesseract pillow` plus the
`tesseract-ocr` system package. Without it, paste screenshot text instead —
the pipeline records the item as UNREADABLE and continues rather than failing.

---

## Architecture

```
    upload / paste
          |
  [1] preprocess.py        our code: parse exports, hash files, OCR,
          |                          detect language, judge quality
          +-------------------------------+
          |                               |
  [2] rulebased_extract.py        [3] extractor.py
      regex on bank SMS:              Gemini: reads messy multilingual
      amounts, refs, balances         chat, assigns scam stages and
      (deterministic, exact)          persuasion tactics, pulls entities
          |                               |
          +---------- merge --------------+
                      |            bank figures win on amounts/balances
                      |            model fills payee, timing, meaning
              [4] validators.py    our code: arithmetic, duplicate refs,
                      |                      formats, impossibility checks
              [5] schema.py        one IncidentRecord
                      |
              [6] generators.py    NCRP · 1930 · bank letter · NPCI
                                   + evidence index + urgent actions
```

### Classical pre-extraction, before the model

Before the single model call, `core/preextract.py` runs standard libraries
(`phonenumbers`, `dateparser`, `email_validator`, plus regex for UPI IDs,
transaction references, and URLs) over the evidence bundle. Two effects:

1. **Fewer tokens spent on things a library already gets exactly right.**
   The verified entity list is passed to the model as a compact hint block
   ("Pre-extracted by classical tools — verify against the text") instead of
   asking the model to re-scan raw, noisy, multilingual text for every phone
   number and UPI handle.
2. **Boilerplate is stripped** (pure promotional SMS, OTP-only messages)
   before the bundle reaches the prompt.

The model still does everything a library cannot: read meaning, assign scam
stages, judge tactics, resolve relative dates using context. The model is
called **exactly once per analysis** — extraction, stage/tactic tagging, and
model-side findings all come back in one structured JSON response, verified
by `test_single_model_call_per_run`. Deterministic generation of the four
complaint documents afterward makes zero further model calls.

### Why the split is what it is

Prompt-only testing of this task failed in a specific, reproducible way: given
a bank alert sequence where the balance *rose* across a debit with no credit —
arithmetically impossible, and the strongest signal in the whole bundle — the
model missed it and instead emitted filler findings, including one whose
content was that nothing was wrong.

So the division is deliberate:

| Job | Where | Reason |
|---|---|---|
| Reading garbled, code-mixed text | Model | Regex cannot read Telugu-English scam chat |
| Assigning scam stages and tactics | Model | Requires language understanding |
| Amounts, balances, totals, net loss | Code | Must be exact and reproducible |
| Phone numbers, emails, UPI IDs, URLs, dates | Code (pre-extraction) | Libraries get these exactly right, cheaper than a model call |
| Reference format and duplicate checks | Code | Cheap, deterministic, testable |
| Contradiction detection | Code | The model demonstrably misses these |
| Generating the four documents | Code | Formats are fixed specifications |

The model is never asked to compute a total or judge whether a balance
sequence is possible. It is told explicitly not to.

---

## What the checks catch

**Impossibility checks** (India-specific, cited by ID in output)

- `I1` arrest or custody conducted over a call — no such procedure in Indian law
- `I2` an official body demanding funds for verification or clearance
- `I3` funds to a personal UPI handle while claiming an official collection
- `I4` a promise that transferred money is automatically refundable
- `I5` a fee demanded to recover money already lost (recovery scam)
- `I6` a claimed payment with no corresponding bank record
- `I7` a running balance sequence that cannot occur
- `I8` remote-access software, or secrecy instructions

**Tamper indicators** `T1`–`T7`: reference-format anomalies, a reference reused
on two payments, wording errors in app-generated text, missing fields,
screenshot/chat timestamp conflicts, signs of an edited export.

Every finding names what document would settle it. Formats alone are never
asserted as proof of forgery.

**No padding rule:** LOW-severity variance is summarised in one line, never
itemised, and a clean record produces zero findings. This is enforced by a
test.

---

## Complainant details

The victim's name, mobile, email, and address are collected once (web form
or CLI) and flow into every generated document — NCRP fields, the 1930 call
brief, and the bank letter — without needing to be re-extracted from
evidence text. Missing fields render as `[TO BE FILLED]` rather than being
guessed.

## Roles handled

| Role | Situation |
|---|---|
| `PAYER_VICTIM` | The user sent money |
| `PAYEE_VICTIM` | The user was shown fake proof of a payment *to* them |
| `THIRD_PARTY` | A relative reporting for the victim |
| `UNCLEAR` | Insufficient evidence — stated, not guessed |

The payee case matters: a shopkeeper shown a forged "payment successful"
screenshot needs the opposite advice from a victim who sent money, and the
tool raises CRITICAL when a claimed incoming payment has no matching credit.

---

## Evidence discipline

- Absent from evidence → `NOT FOUND IN EVIDENCE`
- Present but uncheckable → `UNVERIFIABLE FROM EVIDENCE`
- Garbled → raw form kept, best reading offered, confidence `LOW`
- Every OCR correction is logged with raw → corrected
- Account numbers are masked in generated documents
- SHA-256 of every evidence item is recorded for chain of custody
- Nothing is written to disk by the web app

---

## Test coverage

36 tests, no API key required. Highlights:

- the balance impossibility is detected, and a *valid* sequence produces
  nothing (no false positive)
- gross outflow and net loss computed separately when money flows back
- a clean record yields zero findings
- model JSON is salvaged from fences or surrounding prose
- unexpected keys in a model response cannot crash parsing
- NCRP forbidden characters stripped, 200-character minimum enforced

---

## Layout

```
pipeline.py                CLI orchestrator
core/schema.py             IncidentRecord — the unified data model
core/preprocess.py         parsing, hashing, OCR, language, quality
core/rulebased_extract.py  deterministic bank-SMS extraction
core/extractor.py          Gemini prompt and response mapping
core/validators.py         the checks
core/generators.py         the four outputs
web/server.py              FastAPI app
web/index.html             single-page UI
cases/case1.txt            sample bundle
tests/test_pipeline.py     test suite
```

---

## Limits, stated plainly

- Prepares paperwork; does not file anything or give legal advice
- Portal field layouts change — verify against the live interface
- Only the NCRP section reflects an official checklist; the 1930, bank and
  NPCI outputs are project templates
- Image tamper analysis is limited to what OCR text reveals; pixel-level
  forensics is not implemented
- Model output varies between runs; the deterministic layer does not
