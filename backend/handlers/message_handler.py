import logging

from config.disclaimers import (
    ENGLISH_DISCLAIMER,
    HINDI_DISCLAIMER,
    NEVER_SAY,
    RISK_LANGUAGE,
    check_for_banned_words,
)
from services.whatsapp_service import (
    download_media,
    send_buttons,
    send_message,
)

logger = logging.getLogger(__name__)

_PHOTO_BUTTONS = [
    {"id": "btn_oral", "title": "Oral cavity photo"},
    {"id": "btn_skin", "title": "Skin lesion photo"},
    {"id": "btn_asha", "title": "Talk to ASHA"},
]

_WELCOME_TEXT = (
    "नमस्ते! CancerSetu में आपका स्वागत है 🙏\n"
    "मुंह या त्वचा की तस्वीर भेजें।\n"
    "---\n"
    "Hello! Welcome to CancerSetu.\n"
    "Send a photo of your mouth or skin for screening."
)

_REPORT_PROMPT = (
    "रिपोर्ट चाहिए? Reply YES\n"
    "Want a PDF report? Reply YES"
)

_AUDIO_REDIRECT = (
    "कृपया तस्वीर भेजें / Please send a photo for screening 📷"
)


# ---------------------------------------------------------------------------
# Stubs — replace with real implementations in later sprints
# ---------------------------------------------------------------------------

async def _check_image_quality(image_bytes: bytes) -> str:
    """Stub: always returns GOOD. Replace with blur-detection logic."""
    return "GOOD"


async def _run_ml_inference(image_bytes: bytes) -> tuple[str, float]:
    """Stub: returns MEDIUM_RISK at 74% confidence.
    Replace with TFLite model inference in ml/ sprint.
    """
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
    """Download image, run inference, compose safe reply, send result."""
    logger.info("handle_image: sender=%s media_id=%s", sender, image_id)

    # 1. Download
    image_bytes = await download_media(image_id)

    # 2. Quality check
    quality = await _check_image_quality(image_bytes)
    if quality != "GOOD":
        await send_message(
            sender,
            "तस्वीर साफ नहीं है। कृपया अच्छी रोशनी में दोबारा लें।\n"
            "Image is unclear. Please retake in good lighting.",
        )
        return

    # 3. ML inference
    risk_level, confidence = await _run_ml_inference(image_bytes)

    # 4. Build reply from disclaimer constants
    risk = RISK_LANGUAGE[risk_level]
    reply = (
        f"{risk['emoji']} *Screening Result*\n\n"
        f"{risk['hindi']}\n\n"
        f"{risk['english']}\n\n"
        f"Confidence: {round(confidence * 100)}%\n\n"
        f"{HINDI_DISCLAIMER}\n"
        f"{ENGLISH_DISCLAIMER}"
    )

    # 5. Safety gate — scan for banned words before sending
    check_for_banned_words(reply)

    # 6. Send result
    await send_message(sender, reply)

    # 7. Offer PDF report
    await send_message(sender, _REPORT_PROMPT)

    logger.info(
        "handle_image complete: sender=%s risk=%s confidence=%.2f",
        sender, risk_level, confidence,
    )


async def handle_audio(sender: str, audio_id: str) -> None:
    """Audio messages — redirect user to send a photo instead."""
    logger.info("handle_audio: sender=%s media_id=%s", sender, audio_id)
    await send_message(sender, _AUDIO_REDIRECT)
