import json
import logging
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL = "gemini-1.5-pro"

_ANALYSIS_PROMPT_TEMPLATE = (
    "You are a medical AI assistant helping rural Indians screen for \n"
    "early cancer signs. A CNN model has classified this image as \n"
    "{risk_level} with {confidence}% confidence.\n\n"
    "Respond ONLY in this exact JSON format:\n"
    "{{\n"
    '    "hindi_message": "...",\n'
    '    "english_message": "...",\n'
    '    "risk_emoji": "...",\n'
    '    "action_required": true/false,\n'
    '    "disclaimer": "..."\n'
    "}}\n\n"
    "Rules:\n"
    "- hindi_message: Simple Hindi (no complex words), max 3 sentences, \n"
    "  explain the risk in plain language a villager would understand\n"
    "- english_message: Same but English, for ASHA workers\n"
    "- risk_emoji: single emoji representing risk level\n"
    "- action_required: true if MEDIUM or HIGH risk\n"
    "- disclaimer: always add 'यह एक AI स्क्रीनिंग है, डॉक्टर से मिलें'\n\n"
    "Be compassionate, not alarming. Never say 'you have cancer'.\n"
    "Say 'this needs attention' or 'please see a doctor'."
)

_QUALITY_PROMPT = (
    "Is this a clear photo suitable for medical screening? "
    "Look for: blur, poor lighting, wrong angle, not showing "
    "mouth/skin properly. Reply only: GOOD or BAD: [reason]"
)

_FALLBACK_ANALYSIS = {
    "hindi_message": "यह तस्वीर जांची गई। कृपया डॉक्टर से मिलें।",
    "english_message": "This image has been screened. Please consult a doctor.",
    "risk_emoji": "🟡",
    "action_required": True,
    "disclaimer": "यह एक AI स्क्रीनिंग है, डॉक्टर से मिलें",
}

_ERROR_ANALYSIS = {
    "hindi_message": "जांच में समस्या हुई। कृपया डॉक्टर से सलाह लें।",
    "english_message": "Screening encountered an issue. Please consult a doctor.",
    "risk_emoji": "⚠️",
    "action_required": True,
    "disclaimer": "यह एक AI स्क्रीनिंग है, डॉक्टर से मिलें",
}


def _extract_json(text: str) -> dict:
    """Extract JSON object from Gemini response, handling markdown code blocks."""
    # Try ```json ... ``` or ``` ... ``` blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fall back to first bare JSON object in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in Gemini response: {text!r}")


async def analyze_image_with_gemini(
    image_bytes: bytes,
    risk_level: str,
    confidence: float,
) -> dict:
    """Send image + risk context to Gemini 1.5 Pro, return structured analysis.

    Returns a dict with keys:
        hindi_message, english_message, risk_emoji, action_required, disclaimer
    Falls back to safe defaults on parse or API errors.
    """
    prompt = _ANALYSIS_PROMPT_TEMPLATE.format(
        risk_level=risk_level,
        confidence=round(confidence * 100),
    )
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    try:
        response = await _client.aio.models.generate_content(
            model=_MODEL,
            contents=[prompt, image_part],
        )
        result = _extract_json(response.text)
        logger.info(
            "Gemini analysis done: risk=%s action_required=%s",
            risk_level,
            result.get("action_required"),
        )
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini JSON parse failed (%s). Using fallback.", exc)
        return _FALLBACK_ANALYSIS
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return _ERROR_ANALYSIS


async def check_image_quality(image_bytes: bytes) -> dict:
    """Ask Gemini whether the image is suitable for medical screening.

    Returns:
        {"quality": "GOOD"}
        {"quality": "BAD", "reason": "<reason>"}
    Fails open (returns GOOD) on API errors so the pipeline can continue.
    """
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    try:
        response = await _client.aio.models.generate_content(
            model=_MODEL,
            contents=[_QUALITY_PROMPT, image_part],
        )
        text = response.text.strip()

        if text.upper().startswith("GOOD"):
            return {"quality": "GOOD"}

        # Expected format: "BAD: <reason>"
        reason = text.split(":", 1)[1].strip() if ":" in text else text
        return {"quality": "BAD", "reason": reason}

    except Exception as exc:
        logger.error("Gemini quality check error: %s — failing open", exc)
        return {"quality": "GOOD"}
