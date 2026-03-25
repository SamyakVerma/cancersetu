# NEVER MODIFY THESE — medical-legal compliance
# All user-facing risk language lives here.
# We never diagnose. We only triage.

HINDI_DISCLAIMER = "⚠️ यह एक AI स्क्रीनिंग टूल है, कोई डॉक्टरी निदान नहीं। कृपया किसी योग्य डॉक्टर से मिलें।"
ENGLISH_DISCLAIMER = "⚠️ This is an AI screening tool, not a medical diagnosis. Please consult a qualified doctor."

RISK_LANGUAGE = {
    "HIGH_RISK": {
        "hindi": "इस तस्वीर में कुछ चीज़ें हैं जिन पर ध्यान देना ज़रूरी है। जल्द से जल्द डॉक्टर से मिलें।",
        "english": "This image shows some areas that need medical attention. Please see a doctor soon.",
        "emoji": "🔴",
    },
    "MEDIUM_RISK": {
        "hindi": "कुछ संकेत हैं जिन्हें जांचना चाहिए। एक हफ्ते में डॉक्टर से मिलें।",
        "english": "There are some signs that should be checked. See a doctor within a week.",
        "emoji": "🟡",
    },
    "LOW_RISK": {
        "hindi": "अभी कोई गंभीर संकेत नहीं दिखा। लेकिन नियमित जांच करते रहें।",
        "english": "No serious signs detected. But keep monitoring and do regular checkups.",
        "emoji": "🟢",
    },
}

# These words are banned from ALL user-facing output.
# Scan every outbound message before sending.
NEVER_SAY = ["cancer", "tumor", "malignant", "कैंसर"]


def check_for_banned_words(text: str) -> None:
    """Raise ValueError if any banned word is found in text.

    Call this on every string before it leaves the system.
    """
    text_lower = text.lower()
    for word in NEVER_SAY:
        if word.lower() in text_lower:
            raise ValueError(
                f"Banned word '{word}' detected in outbound message. "
                "Remove it before sending. We never diagnose — we only triage."
            )
