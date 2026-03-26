import hashlib
import logging
import os
from datetime import datetime

from config.disclaimers import check_for_banned_words
from services.firebase_service import save_screening, upload_pdf
from services.gemini_service import analyze_image_with_gemini, check_image_quality
from services.maps_service import get_maps_link
from services.pdf_service import generate_report
from services.whatsapp_service import download_media, send_buttons, send_document, send_message

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_PHOTO_BUTTONS = [
    {"id": "btn_oral", "title": "Oral cavity photo"},
    {"id": "btn_skin", "title": "Skin lesion photo"},
    {"id": "btn_asha", "title": "Talk to ASHA"},
]

_WELCOME_TEXT = (
    "नमस्ते! JanArogya में आपका स्वागत है 🙏\n"
    "मुंह या त्वचा की तस्वीर भेजें।\n"
    "---\n"
    "Hello! Welcome to JanArogya.\n"
    "Send a photo of your mouth or skin for screening."
)

_PDF_OFFER = (
    "📄 रिपोर्ट PDF चाहिए? Reply *YES* भेजें।\n"
    "Want a PDF report? Reply *YES*."
)

_AUDIO_REDIRECT = (
    "कृपया तस्वीर भेजें / Please send a photo for screening 📷"
)

_DEFAULT_MAPS_LINK = get_maps_link("cancer screening hospital")

# In-memory store: phone → report data, waiting for YES reply
_pending_reports: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _urgency(risk_level: str, action_required: bool) -> str:
    if risk_level == "HIGH_RISK":
        return "urgent"
    if risk_level == "MEDIUM_RISK" or action_required:
        return "within_week"
    return "monitor"


# ── Stub — replace with TFLite model inference in the ml/ sprint ───────────────

async def _run_ml_inference(image_bytes: bytes) -> tuple[str, float]:
    """Stub: returns MEDIUM_RISK at 74% confidence."""
    return "MEDIUM_RISK", 0.74


# ── PDF delivery (called when user replies YES) ────────────────────────────────

async def _deliver_pdf(sender: str) -> None:
    """Generate, send, and log the PDF for a pending report."""
    report_data = _pending_reports.pop(sender, None)
    if not report_data:
        await send_message(sender, "कोई pending रिपोर्ट नहीं। / No pending report found.")
        return

    phone_hash = report_data["phone_hash"]
    patient_id = sender[-4:] if len(sender) >= 4 else sender
    tmp_path   = f"/tmp/report_{phone_hash[:8]}.pdf"

    try:
        pdf_bytes = generate_report(report_data)

        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)

        await send_document(
            sender,
            pdf_bytes,
            filename=f"CancerSetu_Report_{patient_id}.pdf",
            caption="📄 आपकी स्क्रीनिंग रिपोर्ट / Your AI screening report",
        )
        logger.info("PDF delivered: sender=%s size=%d bytes", sender, len(pdf_bytes))

        # Log to Firebase
        doc_id = await save_screening(
            phone=sender,
            risk_level=report_data["risk_level"],
            confidence=report_data["confidence"],
            hindi_message=report_data["hindi_message"],
            english_message=report_data["english_message"],
        )
        if doc_id:
            pdf_url = await upload_pdf(pdf_bytes, sender, doc_id)
            if pdf_url:
                from services.firebase_service import _db
                db = _db()
                if db:
                    db.collection("screenings").document(doc_id).update(
                        {"pdf_url": pdf_url, "pdf_sent": True}
                    )

    except Exception as exc:
        logger.error("PDF delivery failed for %s: %s", sender, exc)
        await send_message(
            sender,
            "रिपोर्ट बनाने में समस्या हुई। / Error generating report. Please try again."
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_text(sender: str, text: str) -> None:
    """Handle incoming text — check for YES reply first, then send welcome."""
    logger.info("handle_text: sender=%s text=%r", sender, text[:80])

    # YES reply → deliver pending PDF
    if text.strip().upper() in ("YES", "हाँ", "HAN", "HAAN", "Y") and sender in _pending_reports:
        await _deliver_pdf(sender)
        return

    # Default: welcome flow
    await send_message(sender, _WELCOME_TEXT)
    await send_buttons(sender, "क्या चाहिए? / What would you like?", _PHOTO_BUTTONS)


async def handle_image(sender: str, image_id: str) -> None:
    """Download image → quality check → inference → Gemini → reply → offer PDF."""
    logger.info("handle_image: sender=%s media_id=%s", sender, image_id)

    # 1. Download
    image_bytes = await download_media(image_id)

    # 2. Quality check via Gemini
    quality_result = await check_image_quality(image_bytes)
    if quality_result["quality"] != "GOOD":
        reason = quality_result.get("reason", "तस्वीर साफ नहीं है")
        await send_message(
            sender,
            f"📷 तस्वीर ठीक नहीं है: {reason}\n"
            "कृपया अच्छी रोशनी में, सीधे कैमरे से दोबारा लें।\n\n"
            f"Photo issue: {reason}\n"
            "Please retake in good lighting, camera facing the area directly.",
        )
        return

    # 3. ML inference
    risk_level, confidence = await _run_ml_inference(image_bytes)

    # 4. Gemini — structured bilingual analysis
    analysis = await analyze_image_with_gemini(image_bytes, risk_level, confidence)

    # 5. Nearest center (no GPS — generic fallback)
    maps_link = _DEFAULT_MAPS_LINK
    nearest_center = {
        "name": "Nearest Cancer Screening Center",
        "address": "Search for hospitals near you on Google Maps",
        "maps_link": maps_link,
        "distance": "",
    }

    # 6. Format WhatsApp text reply
    reply = (
        f"{analysis['risk_emoji']} *CancerSetu स्क्रीनिंग रिपोर्ट*\n\n"
        f"{analysis['hindi_message']}\n\n"
        f"📍 *नजदीकी केंद्र:* {maps_link}\n\n"
        f"_{analysis['disclaimer']}_"
    )
    check_for_banned_words(reply)
    await send_message(sender, reply)

    # 7. Store report data for on-demand PDF
    phone_hash = hashlib.sha256(sender.encode()).hexdigest()
    _pending_reports[sender] = {
        "phone_hash":        phone_hash,
        "scan_date":         datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "scan_type":         "oral/skin",
        "risk_level":        risk_level,
        "confidence":        confidence,
        "hindi_message":     analysis["hindi_message"],
        "english_message":   analysis["english_message"],
        "action_urgency":    _urgency(risk_level, analysis.get("action_required", False)),
        "centers":           [nearest_center],
        "disclaimer_hindi":  "यह एक AI स्क्रीनिंग है, डॉक्टर से मिलें।",
        "disclaimer_english": "This is an AI screening tool, not a medical diagnosis.",
    }

    # 8. Offer PDF
    await send_message(sender, _PDF_OFFER)

    logger.info(
        "handle_image complete: sender=%s risk=%s confidence=%.2f urgency=%s",
        sender, risk_level, confidence,
        _pending_reports[sender]["action_urgency"],
    )


async def handle_audio(sender: str, audio_id: str) -> None:
    """Audio messages — redirect user to send a photo instead."""
    logger.info("handle_audio: sender=%s media_id=%s", sender, audio_id)
    await send_message(sender, _AUDIO_REDIRECT)
