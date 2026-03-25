"""
Tests for the WhatsApp webhook router and disclaimer safety system.

Run from the backend/ directory:
    pytest tests/ -v
"""
import os
import sys

# Ensure backend/ is on the path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import AsyncMock, patch

# Set required env vars before importing the app
os.environ.setdefault("WHATSAPP_VERIFY_TOKEN", "test_verify_token_123")
os.environ.setdefault("WHATSAPP_TOKEN", "test_wa_token")
os.environ.setdefault("WHATSAPP_PHONE_ID", "test_phone_id")

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. GET /api/v1/webhook — Meta verification handshake
# ---------------------------------------------------------------------------

class TestWebhookVerification:
    def test_valid_verification_returns_challenge(self):
        resp = client.get(
            "/api/v1/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test_verify_token_123",
                "hub.challenge": "challenge_abc_789",
            },
        )
        assert resp.status_code == 200
        assert resp.text == "challenge_abc_789"

    def test_wrong_token_returns_403(self):
        resp = client.get(
            "/api/v1/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "challenge_xyz",
            },
        )
        assert resp.status_code == 403

    def test_wrong_mode_returns_403(self):
        resp = client.get(
            "/api/v1/webhook",
            params={
                "hub.mode": "unsubscribe",
                "hub.verify_token": "test_verify_token_123",
                "hub.challenge": "challenge_xyz",
            },
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 2. POST /api/v1/webhook — inbound message handling
# ---------------------------------------------------------------------------

def _make_image_payload(sender: str = "919876543210", media_id: str = "media_001"):
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": sender,
                        "type": "image",
                        "image": {"id": media_id},
                    }]
                }
            }]
        }]
    }


def _make_text_payload(sender: str = "919876543210", text: str = "hello"):
    return {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "from": sender,
                        "type": "text",
                        "text": {"body": text},
                    }]
                }
            }]
        }]
    }


class TestWebhookPost:
    def test_image_message_returns_200(self):
        """POST with a mock image message must always return 200."""
        with (
            patch("handlers.message_handler.download_media", new_callable=AsyncMock) as mock_dl,
            patch("handlers.message_handler.send_message", new_callable=AsyncMock),
        ):
            mock_dl.return_value = b"fake_image_bytes"
            resp = client.post("/api/v1/webhook", json=_make_image_payload())

        assert resp.status_code == 200

    def test_text_message_returns_200(self):
        """POST with a text message must return 200."""
        with (
            patch("handlers.message_handler.send_message", new_callable=AsyncMock),
            patch("handlers.message_handler.send_buttons", new_callable=AsyncMock),
        ):
            resp = client.post("/api/v1/webhook", json=_make_text_payload())

        assert resp.status_code == 200

    def test_malformed_body_still_returns_200(self):
        """Even unparseable payloads must return 200 — never let Meta retry."""
        resp = client.post(
            "/api/v1/webhook",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

    def test_empty_entry_returns_200(self):
        """Webhook status events (no messages key) must return 200."""
        resp = client.post("/api/v1/webhook", json={"entry": []})
        assert resp.status_code == 200

    def test_image_handler_sends_reply(self):
        """handle_image should call send_message at least twice (result + report prompt)."""
        with (
            patch("handlers.message_handler.download_media", new_callable=AsyncMock) as mock_dl,
            patch("handlers.message_handler.send_message", new_callable=AsyncMock) as mock_send,
        ):
            mock_dl.return_value = b"fake_image_bytes"
            client.post("/api/v1/webhook", json=_make_image_payload())

        assert mock_send.call_count >= 2, (
            f"Expected at least 2 send_message calls (result + report prompt), got {mock_send.call_count}"
        )


# ---------------------------------------------------------------------------
# 3. Disclaimer safety — banned-word scanner
# ---------------------------------------------------------------------------

class TestDisclaimerScanner:
    def test_clean_text_passes(self):
        from config.disclaimers import check_for_banned_words
        # Should not raise
        check_for_banned_words("No serious signs detected. See a doctor.")

    def test_banned_word_cancer_raises(self):
        from config.disclaimers import check_for_banned_words
        with pytest.raises(ValueError, match="cancer"):
            check_for_banned_words("You might have cancer, please see a doctor.")

    def test_banned_word_tumor_raises(self):
        from config.disclaimers import check_for_banned_words
        with pytest.raises(ValueError, match="tumor"):
            check_for_banned_words("A tumor was detected in the image.")

    def test_banned_word_malignant_raises(self):
        from config.disclaimers import check_for_banned_words
        with pytest.raises(ValueError, match="malignant"):
            check_for_banned_words("This lesion appears malignant.")

    def test_banned_hindi_word_raises(self):
        from config.disclaimers import check_for_banned_words
        with pytest.raises(ValueError):
            check_for_banned_words("आपको कैंसर हो सकता है।")

    def test_case_insensitive_detection(self):
        from config.disclaimers import check_for_banned_words
        with pytest.raises(ValueError):
            check_for_banned_words("This is CANCER risk assessment.")

    def test_risk_language_constants_are_clean(self):
        """Verify that our own RISK_LANGUAGE strings don't contain banned words."""
        from config.disclaimers import RISK_LANGUAGE, check_for_banned_words
        for level, content in RISK_LANGUAGE.items():
            for lang in ("hindi", "english"):
                check_for_banned_words(content[lang])  # must not raise

    def test_disclaimers_themselves_are_clean(self):
        """The disclaimer strings must not contain banned words."""
        from config.disclaimers import (
            ENGLISH_DISCLAIMER,
            HINDI_DISCLAIMER,
            check_for_banned_words,
        )
        check_for_banned_words(HINDI_DISCLAIMER)
        check_for_banned_words(ENGLISH_DISCLAIMER)
