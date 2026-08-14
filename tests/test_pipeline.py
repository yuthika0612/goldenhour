"""
Test suite. Runs without an API key by mocking the model response, so the
deterministic layer and the generators are testable in CI and in a viva.

    python3 -m tests.test_pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import validators, generators, rulebased_extract as rbx, preprocess
from core.extractor import parse_model_json, record_from_model
from core.schema import IncidentRecord, Transaction, EvidenceItem

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((PASS if condition else FAIL, name, detail))


# --------------------------------------------------------------- unit tests
def test_balance_arithmetic():
    """The check the model failed in prompt-only testing."""
    rec = IncidentRecord(user_role="PAYER_VICTIM")
    rec.evidence_items = [EvidenceItem(id="E1", kind="sms")]
    rec.transactions = [
        Transaction(txn_ref="T1", amount=50000, balance_after=12340,
                    sources=["E1"], direction="debit"),
        Transaction(txn_ref="T2", amount=150000, balance_after=462340,
                    sources=["E1"], direction="debit"),
    ]
    validators.check_balance_arithmetic(rec)
    hit = [f for f in rec.findings if "ARITHMETIC" in f.check]
    check("balance impossibility detected", len(hit) == 1)
    check("balance finding is HIGH", hit and hit[0].severity == "HIGH")
    check("finding explains the rise",
          hit and "INCREASED" in hit[0].observed)


def test_balance_valid_sequence_is_silent():
    """No false positive on a correct sequence."""
    rec = IncidentRecord()
    rec.transactions = [
        Transaction(txn_ref="T1", amount=5000, balance_after=95000,
                    direction="debit"),
        Transaction(txn_ref="T2", amount=15000, balance_after=80000,
                    direction="debit"),
    ]
    validators.check_balance_arithmetic(rec)
    check("valid balance sequence produces no finding", len(rec.findings) == 0)


def test_duplicate_reference():
    rec = IncidentRecord()
    rec.transactions = [
        Transaction(txn_ref="T2501141052987654", amount=150000,
                    payee_upi_id="verify.rbi2025@okaxis", direction="debit"),
        Transaction(txn_ref="T2501141052987654", amount=300000,
                    payee_upi_id="cbi.clearance@okicici", direction="debit"),
    ]
    validators.check_duplicate_refs(rec)
    check("duplicate reference on different payments detected",
          any(f.check == "T2" for f in rec.findings))


def test_authority_handle():
    rec = IncidentRecord()
    rec.transactions = [Transaction(payee_upi_id="verify.rbi2025@okaxis",
                                    amount=50000, direction="debit")]
    validators.check_authority_handle(rec)
    check("official-sounding VPA flagged (I3)",
          any(f.check == "I3" for f in rec.findings))


def test_payee_drift():
    rec = IncidentRecord()
    rec.transactions = [
        Transaction(payee_upi_id="a@okaxis", amount=1, direction="debit"),
        Transaction(payee_upi_id="b@okicici", amount=2, direction="debit"),
    ]
    validators.check_payee_drift(rec)
    check("mule handoff detected (S11)",
          any(f.check == "S11" for f in rec.findings))


def test_payee_victim_no_credit():
    """Case 2: shopkeeper shown fake proof. Must be CRITICAL."""
    rec = IncidentRecord(user_role="PAYEE_VICTIM")
    validators.check_role_consistency(rec)
    crit = [f for f in rec.findings if f.severity == "CRITICAL"]
    check("fake payment proof raises CRITICAL", len(crit) == 1)


def test_sms_regex_extraction():
    items = preprocess.ingest(sms=(
        "SBI: Rs 50,000 debited from A/c XX4521 on 14-Jan-25. "
        "UPI Ref T2501140900123456. Avl Bal Rs 12,340.\n"
        "HDFC: Rs 6,500 credited 30-05-26 UPI 445120500122. Avl Bal 94,900."))
    txns = rbx.extract_from_sms(items)
    check("regex extracted 2 transactions", len(txns) == 2, f"got {len(txns)}")
    debit = [t for t in txns if t.direction == "debit"]
    credit = [t for t in txns if t.direction == "credit"]
    check("debit amount correct", debit and debit[0].amount == 50000)
    check("balance captured, not confused with amount",
          debit and debit[0].balance_after == 12340)
    check("credit direction correct", len(credit) == 1)
    check("reference captured",
          debit and debit[0].txn_ref == "T2501140900123456")


def test_net_loss_with_inflow():
    """Case 4: an inflow must reduce net loss but not the gross figure."""
    rec = IncidentRecord()
    rec.transactions = [
        Transaction(amount=5000, direction="debit"),
        Transaction(amount=50000, direction="debit"),
        Transaction(amount=75000, direction="debit"),
        Transaction(amount=450, direction="credit"),
        Transaction(amount=6500, direction="credit"),
    ]
    rec.compute_totals()
    check("gross outflow 130000", rec.total_fraud_amount == 130000)
    check("net loss 123050", rec.net_loss == 123050,
          f"got {rec.net_loss}")


def test_ncrp_sanitisation():
    rec = IncidentRecord()
    rec.incident_description = "Caller said pay @ once! Cost #urgent " * 8
    out = generators.generate_ncrp(rec)
    desc = out["mandatory"]["incident_details"]
    check("forbidden characters stripped",
          not any(c in desc for c in "#$@^*`'~|!"))
    check("validation flag reports clean",
          out["validation"]["forbidden_chars_present"] is False)


def test_short_description_flagged():
    rec = IncidentRecord()
    rec.incident_description = "Too short."
    validators.check_ncrp_text_constraints(rec)
    check("under-200-char description flagged",
          any(f.check == "FORMAT" for f in rec.findings))


def test_no_padding_rule():
    """A clean record must produce no findings at all."""
    rec = IncidentRecord(user_role="PAYER_VICTIM")
    rec.evidence_items = [EvidenceItem(id="E1", kind="sms")]
    rec.transactions = [Transaction(txn_ref="T25011409001234", amount=5000,
                                    balance_after=95000, direction="debit",
                                    payee_upi_id="ramesh.kumar@okhdfc",
                                    sources=["E1"])]
    rec.incident_description = "x" * 250
    validators.run_all(rec)
    check("clean record yields zero findings (no padding)",
          len(rec.findings) == 0,
          "; ".join(f.check for f in rec.findings))


def test_model_json_salvage():
    """Model wrapped JSON in prose or fences must still parse."""
    a = parse_model_json('```json\n{"user_role": "PAYER_VICTIM"}\n```')
    b = parse_model_json('Sure! Here you go:\n{"user_role": "THIRD_PARTY"}\nHope that helps')
    check("fenced JSON parsed", a["user_role"] == "PAYER_VICTIM")
    check("prose-wrapped JSON salvaged", b["user_role"] == "THIRD_PARTY")


def test_model_extra_keys_do_not_crash():
    data = {"user_role": "PAYER_VICTIM",
            "transactions": [{"txn_ref": "T1", "amount": 100,
                              "hallucinated_key": "boom"}],
            "timeline_events": [], "model_findings": []}
    rec = record_from_model(data, [EvidenceItem(id="E1", kind="sms")])
    check("unexpected model keys ignored safely", len(rec.transactions) == 1)


def test_full_offline_pipeline():
    import pipeline
    case = Path(__file__).resolve().parent.parent / "cases" / "case1.txt"
    rec = pipeline.run(case.read_text(encoding="utf-8"), offline=True)
    check("case1 offline: 3 transactions", len(rec.transactions) == 3,
          f"got {len(rec.transactions)}")
    check("case1 offline: total 500000", rec.total_fraud_amount == 500000,
          f"got {rec.total_fraud_amount}")
    check("case1 offline: arithmetic impossibility caught",
          any("ARITHMETIC" in f.check for f in rec.findings))
    check("case1 offline: phishing URL captured",
          any("sbi-kyc-verify" in u for u in rec.suspect_urls_social_handles))
    outs = generators.generate_all(rec)
    check("all six outputs generated", len(outs) == 6)
    check("1930 brief names the references",
          "T2501140900123456" in outs["brief_1930"])
    check("bank letter includes request section",
          "REQUEST" in outs["bank_letter"])
    check("account number masked in bank letter",
          "XXXX4521" in outs["bank_letter"] or "XXX4521" in outs["bank_letter"],
          "masking applied")


def test_mocked_model_path():
    """Full model path with a canned response: no API key needed."""
    mock = """```json
    {"user_role":"PAYER_VICTIM","fraud_type":"Digital Arrest",
     "fraud_type_basis":"Caller claimed police authority and demanded refundable verification transfers.",
     "incident_datetime":"14/01/2025 10:02","bank_wallet_merchant":"SBI",
     "suspect_mobiles":["+91 88XXX 11223"],
     "suspect_upi_ids":["verify.rbi2025@okaxis","cbi.clearance@okicici"],
     "suspect_aliases":["Inspector Verma"],
     "transactions":[
       {"txn_ref":"T2501140900123456","amount":50000,"direction":"debit",
        "datetime":"14 Jan 2025 10:20","payee_upi_id":"verify.rbi2025@okaxis",
        "sources":["E1","E2","E4"],"confidence":"HIGH"},
       {"txn_ref":"T2501141052987654","amount":150000,"direction":"debit",
        "datetime":"14 Jan 2025 10:50","payee_upi_id":"verify.rbi2025@okaxis",
        "sources":["E1","E4"],"confidence":"MEDIUM"}],
     "timeline_events":[
       {"timestamp":"14/01/2025 10:02","stage":"S01","tactics":["AUTHORITY"],
        "description":"Caller introduced himself as a cyber crime officer","sources":["E1"]},
       {"timestamp":"14/01/2025 10:07","stage":"S06","tactics":["ISOLATION","FEAR"],
        "description":"Complainant was told not to disconnect or inform family","sources":["E1"]}],
     "model_findings":[
       {"severity":"HIGH","check":"I1","observed":"Caller asserted a digital arrest over a call",
        "evidence":["E1"],"why_it_matters":"No such procedure exists in Indian law.",
        "resolves_with":"None needed."}],
     "ocr_corrections":[{"raw":"verify.rbi2O25@okaxis","corrected":"verify.rbi2025@okaxis","where":"E2"}],
     "weakest_link":"The third payment appears only in a bank alert."}
    ```"""
    data = parse_model_json(mock)
    items = preprocess.ingest(sms=(
        "SBI: Rs 50,000 debited from A/c XX4521 on 14-Jan-25. UPI Ref "
        "T2501140900123456. Avl Bal Rs 12,340.\n"
        "SBI: Rs 1,50,000 debited from A/c XX4521 on 14-Jan-25. UPI Ref "
        "T2501141052987654. Avl Bal Rs 4,62,340."))
    rec = record_from_model(data, items)
    rule_txns = rbx.extract_from_sms(items)
    rbx.cross_check(rec.transactions, rule_txns)
    rec.transactions = rbx.merge(rec.transactions, rule_txns)
    rec.compute_totals()
    rec.incident_description = generators.build_narrative(rec)
    validators.run_all(rec)

    check("mocked path: stages preserved", len(rec.timeline_events) == 2)
    check("mocked path: model finding retained",
          any(f.origin == "model" for f in rec.findings))
    check("mocked path: balances backfilled from SMS",
          all(t.balance_after is not None for t in rec.transactions))
    check("mocked path: arithmetic still caught",
          any("ARITHMETIC" in f.check for f in rec.findings))
    check("mocked path: findings sorted by severity",
          [f.severity for f in rec.findings] ==
          sorted([f.severity for f in rec.findings],
                 key=lambda s: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[s]))
    ncrp = generators.generate_ncrp(rec)
    check("mocked path: NCRP classification carried",
          ncrp["classification"]["fraud_type"] == "Digital Arrest")


# ------------------------------------------------------------------- runner
def main():
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            try:
                fn()
            except Exception as exc:
                results.append((FAIL, fn.__name__, f"raised {exc!r}"))

    width = max(len(n) for _, n, _ in results) + 2
    print("=" * (width + 12))
    for status, name, detail in results:
        line = f"  {status}  {name:<{width}}"
        if detail and status == FAIL:
            line += f"  <- {detail}"
        print(line)
    passed = sum(1 for s, _, _ in results if s == PASS)
    print("=" * (width + 12))
    print(f"  {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


# ------------------------------------------------- post-quantum sealing
def test_evidence_seal_roundtrip():
    from core.evidence_seal import (MerkleSignatureScheme, seal_bundle,
                                    verify_seal)
    items = preprocess.ingest(chat="[01/01/2026, 10:00] A: hi",
                              sms="SBI: Rs 100 debited. Ref X1. Avl Bal Rs 5.")
    signer = MerkleSignatureScheme(height=3)
    b = seal_bundle(items, "CASE/1", signer)
    v = verify_seal(b)
    check("seal verifies", v["sealed"] is True)
    check("merkle root recomputes", v["merkle_root_matches"] is True)
    check("hash-based signature valid", v["signature_valid"] is True)


def test_evidence_seal_detects_tampering():
    from core.evidence_seal import MerkleSignatureScheme, seal_bundle
    items = preprocess.ingest(sms="SBI: Rs 50,000 debited. Ref A1. Bal Rs 10.")
    signer = MerkleSignatureScheme(height=3, seed=b"fixed-seed-for-test-only!")
    before = seal_bundle(items, "CASE/2", signer, sealed_at="2026-01-01T00:00Z")
    items[0].raw_excerpt = items[0].raw_excerpt.replace("50,000", "5,000")
    signer2 = MerkleSignatureScheme(height=3, seed=b"fixed-seed-for-test-only!")
    after = seal_bundle(items, "CASE/2", signer2, sealed_at="2026-01-01T00:00Z")
    check("tampering changes the merkle root",
          before.merkle_root != after.merkle_root)


def test_forged_signature_rejected():
    from core.evidence_seal import MerkleSignatureScheme, seal_bundle, verify_seal
    items = preprocess.ingest(sms="SBI: Rs 10 debited. Ref B1. Bal Rs 1.")
    b = seal_bundle(items, "CASE/3", MerkleSignatureScheme(height=3))
    b.signature["ots_signature"][0] = "00" * 32
    check("forged signature rejected", verify_seal(b)["signature_valid"] is False)


def test_selective_disclosure():
    import json as _json
    from core.evidence_seal import (MerkleSignatureScheme, seal_bundle,
                                    selective_disclosure, verify_disclosure)
    items = preprocess.ingest(
        chat="[01/01/2026, 10:00] A: private conversation",
        sms="SBI: Rs 900 debited. Ref C1. Bal Rs 4.",
        note="private note")
    b = seal_bundle(items, "CASE/4", MerkleSignatureScheme(height=3))
    d = selective_disclosure(items, 1, b)
    payload = _json.dumps({
        "id": items[1].id, "kind": items[1].kind,
        "filename": items[1].filename, "sha256": items[1].sha256,
        "content": items[1].raw_excerpt,
    }, sort_keys=True, ensure_ascii=False).encode()
    check("disclosed item proves inclusion", verify_disclosure(payload, d))
    check("other items stay withheld", d["items_withheld"] == len(items) - 1)

    wrong = payload.replace(b"900", b"9000")
    check("altered disclosed item fails the proof",
          verify_disclosure(wrong, d) is False)


def test_merkle_proof_all_positions():
    from core.evidence_seal import MerkleTree, HASH
    for n in (1, 2, 3, 5, 8, 17):
        leaves = [HASH(f"x{i}".encode()).digest() for i in range(n)]
        t = MerkleTree(leaves)
        ok = all(MerkleTree.verify(leaves[i], i, t.proof(i), t.root)
                 for i in range(n))
        check(f"merkle proofs valid at every index (n={n})", ok)


def test_ots_key_exhaustion():
    from core.evidence_seal import MerkleSignatureScheme
    s = MerkleSignatureScheme(height=2)          # 4 keys
    for _ in range(4):
        s.sign(b"m")
    try:
        s.sign(b"m")
        check("key exhaustion raises", False, "no error raised")
    except RuntimeError:
        check("key exhaustion raises", True)


# ------------------------------------------------- classical pre-extraction
def test_preextract_finds_entities():
    from core import preextract
    txt = ("Transfer Rs 50,000 to verify.rbi2025@okaxis. UTR T2501140900123456. "
          "Call +91 88888 11223. See http://sbi-kyc-verify.co.in on 14/01/2025.")
    r = preextract.pre_extract(txt, reported_on="20 January 2025")
    check("pre-extract finds UPI id", "verify.rbi2025@okaxis" in r.upi_ids)
    check("pre-extract finds amount", 50000.0 in r.amounts)
    check("pre-extract finds URL",
          any("sbi-kyc-verify" in u for u in r.urls))
    check("pre-extract finds phone number", len(r.phone_numbers) >= 1)
    check("pre-extract resolves a date", len(r.resolved_dates) >= 1)


def test_preextract_hint_block_format():
    from core import preextract
    r = preextract.pre_extract("Pay Rs 1,000 to x@okaxis. Ref ABCDEFGH1234.")
    hint = r.to_hint_block()
    check("hint block flags itself as pre-extracted",
          "Pre-extracted by classical tools" in hint)
    check("hint block contains the UPI id", "x@okaxis" in hint)


def test_preextract_email_validation():
    from core import preextract
    ok = preextract.validate_email_address("victim@example.com")
    bad = preextract.validate_email_address("not-an-email")
    check("valid email normalises", ok == "victim@example.com")
    check("invalid email returns None", bad is None)


def test_preextract_bundle_merges_across_items():
    from core import preextract
    items = preprocess.ingest(
        chat="[01/01/2026, 10:00] X: pay to a@okaxis ref REF12345678",
        sms="SBI: Rs 5,000 debited. Ref REF87654321. Avl Bal Rs 100.")
    r = preextract.pre_extract_bundle(items)
    check("bundle merges refs from multiple items",
          {"REF12345678", "REF87654321"}.issubset(set(r.transaction_refs)))


def test_single_model_call_per_run():
    """Confirm the architecture claim: one Gemini call per analysis."""
    src = Path("pipeline.py").read_text()
    check("pipeline.py calls the model exactly once",
          src.count("extractor.call_gemini(") == 1)
    src2 = Path("web/server.py").read_text()
    check("server.py calls the model exactly once",
          src2.count("extractor.call_gemini(") == 1)


def test_complainant_fields_reach_ncrp():
    rec = IncidentRecord(victim_name="Test Person", victim_mobile="9999999999",
                         victim_email="t@example.com", victim_address="123 Road")
    out = generators.generate_ncrp(rec)
    check("NCRP carries complainant name",
          out["complainant"]["name"] == "Test Person")
    check("NCRP carries complainant email",
          out["complainant"]["email"] == "t@example.com")


# ------------------------------------------------- fraud-likelihood gating
def test_clean_transaction_no_indicators():
    from core.validators import assess_fraud_likelihood
    rec = IncidentRecord()
    rec.evidence_items = [EvidenceItem(id="E1", kind="sms",
                          raw_excerpt="SBI: Rs 2,499 debited. Ref X1. Avl Bal Rs 8,120.")]
    rec.transactions = [Transaction(amount=2499, direction="debit",
                                    payee_upi_id="landlord@okaxis")]
    check("clean transaction assessed as NO_INDICATORS_FOUND",
          assess_fraud_likelihood(rec) == "NO_INDICATORS_FOUND")


def test_scam_language_flags_possible_fraud():
    from core.validators import assess_fraud_likelihood
    rec = IncidentRecord()
    rec.evidence_items = [EvidenceItem(id="E1", kind="chat",
                          raw_excerpt="You are under digital arrest, pay penalty clearance now")]
    check("scam language assessed as POSSIBLE_FRAUD or stronger",
          assess_fraud_likelihood(rec) in ("POSSIBLE_FRAUD", "LIKELY_FRAUD"))


def test_impossibility_check_flags_likely_fraud():
    from core.validators import assess_fraud_likelihood
    from core.schema import Finding
    rec = IncidentRecord()
    rec.findings = [Finding(severity="HIGH", check="I3",
                            observed="funds to official-sounding personal handle")]
    check("I-check firing assessed as LIKELY_FRAUD",
          assess_fraud_likelihood(rec) == "LIKELY_FRAUD")


def test_1930_brief_does_not_assert_fraud_when_none_found():
    rec = IncidentRecord(fraud_likelihood="NO_INDICATORS_FOUND")
    rec.transactions = [Transaction(amount=100, direction="debit")]
    rec.compute_totals()
    brief = generators.generate_1930_brief(rec)
    check("no false 'I am reporting a fraud' line when no indicators found",
          "I am reporting an online financial fraud" not in brief)
    check("brief states no indicators were found",
          "no fraud indicators were found" in brief)


def test_fingerprint_included_in_response_path():
    """Regression test: fingerprint was computed but silently dropped."""
    src = Path("web/server.py").read_text()
    check("server response includes fingerprint_dims",
          '"fingerprint_dims"' in src)


if __name__ == "__main__":
    raise SystemExit(main())
