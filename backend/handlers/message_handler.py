import logging
import os
from datetime import datetime

from config.disclaimers import check_for_banned_words
from services.gemini_service import analyze_image_with_gemini, check_image_quality
from services.maps_service import get_maps_link
from services.firebase_service import save_screening, upload_pdf
from services.pdf_service import generate_report
from services.whatsapp_service import download_media, send_buttons, send_document, send_message

logger = logging.getLogger(__name__)

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

_AUDIO_REDIRECT = (
    "कृपया तस्वीर भेजें / Please send a photo for screening 📷"
)

# Fallback Maps link — used when the user's GPS location is unavailable
_DEFAULT_MAPS_LINK = get_maps_link("cancer screening hospital")


# ---------------------------------------------------------------------------
# Stub — replace with TFLite model inference in the ml/ sprint
# ---------------------------------------------------------------------------

async def _run_ml_inference(image_bytes: bytes) -> tuple[str, float]:
    """Stub: returns MEDIUM_RISK at 74% confidence."""
    return "MEDIUM_RISK", 0.74


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_text(sender: str, text: str) -> None:
    """Handle any incoming text message — send welcome + action buttons."""
    logger.info("handle_text: sender=%s text=%r", sender, text[:80])
    await send_message(sender, _WELCOME_TEXT)
    await send_buttons(sender, "क्या चाहिए? / What would you like?", _PHOTO_BUTTONS)


async def handle_image(sender: str, image_id: str) -> None:
    """Download image, run quality check + inference, send reply + PDF report."""
    logger.info("handle_image: sender=%s media_id=%s", sender, image_id)

    # 1. Download
    image_bytes = await download_media(image_id)

    # 2. Quality check via Gemini — reject blurry/unusable photos early
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

    # 5. Nearest center — no GPS from image messages, use a generic fallback
    maps_link = _DEFAULT_MAPS_LINK
    nearest_center = {
        "name": "Nearest Cancer Screening Center",
        "address": "Search for hospitals near you on Google Maps",
        "maps_link": maps_link,
        "distance": "",
    }

    # 6. Format WhatsApp reply
    reply = (
        f"{analysis['risk_emoji']} *CancerSetu स्क्रीनिंग रिपोर्ट*\n\n"
        f"{analysis['hindi_message']}\n\n"
        f"📍 *नजदीकी केंद्र:* {maps_link}\n\n"
        f"_{analysis['disclaimer']}_"
    )

    # 7. Safety gate — scan for banned words before sending
    check_for_banned_words(reply)

    # 8. Send WhatsApp text result
    await send_message(sender, reply)

    # 9. Generate PDF, save temporarily, send as document, then clean up
    patient_id = sender[-4:] if len(sender) >= 4 else sender
    safe_sender = sender.replace("+", "").replace(" ", "")
    tmp_path = f"/tmp/report_{safe_sender}.pdf"

    try:
        patient_data = {
            "phone_number": sender,
            "scan_date": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "risk_level": risk_level,
            "confidence": confidence,
            "risk_emoji": analysis.get("risk_emoji", "🟡"),
            "hindi_message": analysis["hindi_message"],
            "english_message": analysis["english_message"],
            "nearest_center": nearest_center,
        }
        pdf_bytes = generate_report(patient_data)

        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)

        await send_document(
            sender,
            pdf_bytes,
            filename=f"CancerSetu_Report_{patient_id}.pdf",
            caption="📄 आपकी स्क्रीनिंग रिपोर्ट / Your AI screening report",
        )
        logger.info("PDF report sent: sender=%s size=%d bytes", sender, len(pdf_bytes))

    except Exception as exc:
        logger.error("PDF report failed for %s: %s", sender, exc)
        # Non-fatal — user already received the text result above

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info(
        "handle_image complete: sender=%s risk=%s confidence=%.2f action_required=%s",
        sender,
        risk_level,
        confidence,
        analysis.get("action_required"),
    )


async def handle_audio(sender: str, audio_id: str) -> None:
    """Audio messages — redirect user to send a photo instead."""
    logger.info("handle_audio: sender=%s media_id=%s", sender, audio_id)
    await send_message(sender, _AUDIO_REDIRECT)
