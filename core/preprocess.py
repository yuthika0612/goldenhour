"""
Raw uploads -> EvidenceItem list.

This is the part the team writes themselves rather than delegating to the
model: parsing exports, hashing files for chain of custody, running OCR, and
judging whether an item is readable at all.
"""
import hashlib
import re
from pathlib import Path
from typing import List, Optional

from .schema import EvidenceItem

# WhatsApp export line: [14/01/2025, 10:02] Sender: text     (several variants)
WA_LINE = re.compile(
    r"^\[?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})[,\s]+"
    r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?)\]?\s*[-–]?\s*"
    r"([^:]{1,60}?):\s*(.*)$"
)

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
TELUGU = re.compile(r"[\u0C00-\u0C7F]")

OCR_NOISE = re.compile(r"[|~^]{2,}|\uFFFD")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def detect_languages(text: str) -> List[str]:
    langs = ["en"]
    if DEVANAGARI.search(text):
        langs.append("hi")
    if TELUGU.search(text):
        langs.append("te")
    # crude romanised-Indic signal: common transliterated tokens
    if re.search(r"\b(naaku|sar|bhaiya|paisa|bhej|kar do|ledu|sir ji|maal)\b",
                 text, re.I):
        langs.append("romanised")
    return langs


def judge_quality(text: str) -> str:
    if not text or len(text.strip()) < 10:
        return "UNREADABLE"
    noise = len(OCR_NOISE.findall(text))
    # digit/letter confusion signature: capital O inside numbers, l inside digits
    confusions = len(re.findall(r"\d[Ol]\d|\dO|O\d|\bl\d", text))
    if noise > 3 or confusions > 4:
        return "DEGRADED"
    return "CLEAN"


def parse_chat_export(text: str, eid: str, filename: Optional[str] = None
                      ) -> EvidenceItem:
    """
    Normalise a WhatsApp/Telegram export into 'timestamp | sender | text'
    lines. Unparseable lines are kept verbatim: a continuation line is often
    where the transaction reference lives.
    """
    out = []
    for line in text.splitlines():
        m = WA_LINE.match(line.strip())
        if m:
            date, time, sender, msg = m.groups()
            out.append(f"{date} {time} | {sender.strip()} | {msg.strip()}")
        elif line.strip():
            out.append(f"           | (cont.) | {line.strip()}")
    body = "\n".join(out) if out else text
    return EvidenceItem(
        id=eid, kind="chat", filename=filename,
        languages=detect_languages(text), quality=judge_quality(text),
        sha256=sha256_bytes(text.encode()), raw_excerpt=body,
    )


def parse_sms_block(text: str, eid: str, filename: Optional[str] = None
                    ) -> EvidenceItem:
    return EvidenceItem(
        id=eid, kind="sms", filename=filename,
        languages=detect_languages(text), quality=judge_quality(text),
        sha256=sha256_bytes(text.encode()), raw_excerpt=text.strip(),
    )


def parse_note(text: str, eid: str, filename: Optional[str] = None
               ) -> EvidenceItem:
    return EvidenceItem(
        id=eid, kind="note", filename=filename,
        languages=detect_languages(text), quality=judge_quality(text),
        sha256=sha256_bytes(text.encode()), raw_excerpt=text.strip(),
    )


def ocr_image(path: str, eid: str) -> EvidenceItem:
    """
    Screenshot -> text.

    Tries pytesseract, then easyocr. If neither is installed the item is
    still registered as UNREADABLE so the pipeline continues and the missing
    OCR shows up honestly in the output rather than crashing the run.
    """
    text, engine = "", "none"
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(path))
        engine = "pytesseract"
    except Exception:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            text = "\n".join(reader.readtext(path, detail=0))
            engine = "easyocr"
        except Exception:
            text = ""

    data = Path(path).read_bytes()
    item = EvidenceItem(
        id=eid, kind="screenshot_ocr", filename=Path(path).name,
        languages=detect_languages(text), quality=judge_quality(text),
        sha256=sha256_bytes(data), raw_excerpt=text.strip(),
    )
    if not text.strip():
        item.quality = "UNREADABLE"
        item.raw_excerpt = (f"[OCR produced no text. Engine: {engine}. "
                            f"Send this image to the model as an image, or "
                            f"install pytesseract.]")
    return item


def ingest(chat: str = "", sms: str = "", note: str = "",
           ocr_texts: Optional[List[str]] = None,
           image_paths: Optional[List[str]] = None) -> List[EvidenceItem]:
    """Assemble an evidence bundle, assigning stable E1..En ids."""
    items: List[EvidenceItem] = []
    n = 0

    def nid() -> str:
        nonlocal n
        n += 1
        return f"E{n}"

    if chat.strip():
        items.append(parse_chat_export(chat, nid(), "chat_export.txt"))
    for t in (ocr_texts or []):
        if t.strip():
            items.append(EvidenceItem(
                id=nid(), kind="screenshot_ocr", filename="screenshot",
                languages=detect_languages(t), quality=judge_quality(t),
                sha256=sha256_bytes(t.encode()), raw_excerpt=t.strip(),
            ))
    for p in (image_paths or []):
        items.append(ocr_image(p, nid()))
    if sms.strip():
        items.append(parse_sms_block(sms, nid(), "sms_inbox.txt"))
    if note.strip():
        items.append(parse_note(note, nid(), "victim_note.txt"))
    return items
