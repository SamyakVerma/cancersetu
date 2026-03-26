"""
Tests for pdf_service.py
Run from backend/ with: python tests/test_pdf_service.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.disclaimers import NEVER_SAY
from services.pdf_service import generate_asha_batch_report, generate_report

# ── Mock data ─────────────────────────────────────────────────────────────────

_CENTERS = [
    {
        "name": "AIIMS New Delhi",
        "address": "Ansari Nagar, New Delhi - 110029",
        "distance": "3.2 km",
        "maps_link": "https://www.google.com/maps/search/?api=1&query=AIIMS+New+Delhi",
    },
    {
        "name": "Rajiv Gandhi Cancer Institute",
        "address": "Sector 5, Rohini, New Delhi",
        "distance": "12.5 km",
        "maps_link": "https://www.google.com/maps/search/?api=1&query=RGCI+Delhi",
    },
    {
        "name": "Tata Memorial Hospital",
        "address": "Dr Ernest Borges Road, Mumbai",
        "distance": "1200 km",
        "maps_link": "https://www.google.com/maps/search/?api=1&query=Tata+Memorial+Mumbai",
    },
]

MOCK_DATA = {
    "phone_hash":     "abc12345def67890",
    "scan_date":      "26 Mar 2026, 10:00 AM",
    "scan_type":      "oral",
    "image_bytes":    None,
    "risk_level":     "MEDIUM_RISK",
    "confidence":     0.74,
    "hindi_message":  (
        "आपकी तस्वीर में कुछ बातें हैं जिन पर ध्यान देना चाहिए। "
        "कृपया एक सप्ताह के अंदर डॉक्टर से मिलें।"
    ),
    "english_message": (
        "Some signs have been detected that need medical attention. "
        "Please see a doctor within a week for further evaluation."
    ),
    "action_urgency":    "within_week",
    "centers":           _CENTERS,
    "disclaimer_hindi":  "यह एक AI स्क्रीनिंग है, डॉक्टर से मिलें।",
    "disclaimer_english": "This is an AI screening tool, not a medical diagnosis.",
}

MOCK_SCANS = [
    {**MOCK_DATA, "risk_level": "HIGH_RISK",   "action_urgency": "urgent",
     "phone_hash": "aaa111"},
    {**MOCK_DATA, "risk_level": "MEDIUM_RISK",  "action_urgency": "within_week",
     "phone_hash": "bbb222"},
    {**MOCK_DATA, "risk_level": "LOW_RISK",     "action_urgency": "monitor",
     "phone_hash": "ccc333"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_no_banned_words(pdf_bytes: bytes, test_name: str) -> None:
    text = pdf_bytes.decode("latin-1", errors="ignore").lower()
    for word in NEVER_SAY:
        assert word.lower() not in text, (
            f"[{test_name}] Banned word {word!r} found in PDF text layer"
        )


# ── Test cases ────────────────────────────────────────────────────────────────

def test_medium_risk_report():
    pdf = generate_report(MOCK_DATA)
    assert pdf[:4] == b"%PDF", "Not a valid PDF (wrong magic bytes)"
    size = len(pdf)
    assert 50_000 <= size <= 2_000_000, f"PDF size out of range: {size} bytes"
    _check_no_banned_words(pdf, "medium_risk_report")
    print(f"  PASS  test_medium_risk_report        — {size // 1024} KB")


def test_high_risk_report():
    data = {**MOCK_DATA, "risk_level": "HIGH_RISK", "action_urgency": "urgent"}
    pdf  = generate_report(data)
    assert pdf[:4] == b"%PDF"
    size = len(pdf)
    assert 50_000 <= size <= 2_000_000, f"PDF size out of range: {size} bytes"
    _check_no_banned_words(pdf, "high_risk_report")
    print(f"  PASS  test_high_risk_report           — {size // 1024} KB")


def test_low_risk_no_action():
    data = {**MOCK_DATA, "risk_level": "LOW_RISK", "action_urgency": "monitor",
            "centers": []}
    pdf  = generate_report(data)
    assert pdf[:4] == b"%PDF"
    print(f"  PASS  test_low_risk_no_action         — {len(pdf) // 1024} KB")


def test_report_no_centers():
    data = {**MOCK_DATA, "centers": []}
    pdf  = generate_report(data)
    assert pdf[:4] == b"%PDF"
    print(f"  PASS  test_report_no_centers          — {len(pdf) // 1024} KB")


def test_asha_batch_report():
    pdf  = generate_asha_batch_report(MOCK_SCANS)
    assert pdf[:4] == b"%PDF"
    size = len(pdf)
    assert 50_000 <= size <= 2_000_000, f"PDF size out of range: {size} bytes"
    _check_no_banned_words(pdf, "asha_batch_report")
    print(f"  PASS  test_asha_batch_report          — {size // 1024} KB")


def test_asha_batch_empty():
    pdf = generate_asha_batch_report([])
    assert pdf[:4] == b"%PDF"
    print(f"  PASS  test_asha_batch_empty           — {len(pdf) // 1024} KB")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_medium_risk_report,
        test_high_risk_report,
        test_low_risk_no_action,
        test_report_no_centers,
        test_asha_batch_report,
        test_asha_batch_empty,
    ]

    passed = failed = 0
    print("\nRunning PDF service tests…\n")
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__:40s} — {exc}")
            failed += 1

    print(f"\n{'=' * 52}")
    print(f"  Results: {passed} PASS  |  {failed} FAIL")
    print(f"{'=' * 52}\n")
    sys.exit(0 if failed == 0 else 1)
