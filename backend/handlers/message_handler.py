import logging

from config.disclaimers import check_for_banned_words
from services.gemini_service import analyze_image_with_gemini, check_image_quality
from services.maps_service import find_nearest_cancer_center, get_maps_link
from services.whatsapp_service import download_media, send_buttons, send_message

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

_REPORT_PROMPT = (
    "रिपोर्ट चाहिए? Reply YES\n"
    "Want a PDF report? Reply YES"
)

_AUDIO_REDIRECT = (
    "कृपया तस्वीर भेजें / Please send a photo for screening 📷"
)

# Fallback Maps link used when the user's location is not available
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
    """Download image, run quality check + inference, compose safe reply, send result."""
    logger.info("handle_image: sender=%s media_id=%s", sender, image_id)

    # 1. Download
    image_bytes = await download_media(image_id)

    # 2. Quality check via Gemini — ask before running expensive inference
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

    # 5. Maps link — no lat/lng from image messages, use generic fallback
    maps_link = _DEFAULT_MAPS_LINK

    # 6. Format final reply
    reply = (
        f"{analysis['risk_emoji']} *CancerSetu स्क्रीनिंग रिपोर्ट*\n\n"
        f"{analysis['hindi_message']}\n\n"
        f"📍 *नजदीकी केंद्र:* {maps_link}\n\n"
        f"_{analysis['disclaimer']}_"
    )

    # 7. Safety gate — scan for banned words before sending
    check_for_banned_words(reply)

    # 8. Send result
    await send_message(sender, reply)

    # 9. Offer PDF report
    await send_message(sender, _REPORT_PROMPT)

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
