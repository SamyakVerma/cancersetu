import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import APIRouter, Query, Request, Response

from handlers.message_handler import handle_audio, handle_image, handle_text

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook", tags=["WhatsApp"])

_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")


# ---------------------------------------------------------------------------
# GET /webhook — Meta verification handshake
# ---------------------------------------------------------------------------

@router.get("")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode", default=""),
    hub_verify_token: str = Query(alias="hub.verify_token", default=""),
    hub_challenge: str = Query(alias="hub.challenge", default=""),
) -> Response:
    """Respond to Meta's one-time webhook verification request."""
    if hub_mode == "subscribe" and hub_verify_token == _VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.warning(
        "Webhook verification failed: mode=%r token_match=%s",
        hub_mode,
        hub_verify_token == _VERIFY_TOKEN,
    )
    return Response(content="Forbidden", status_code=403)


# ---------------------------------------------------------------------------
# POST /webhook — receives all inbound messages
# ---------------------------------------------------------------------------

@router.post("")
async def receive_message(request: Request) -> Response:
    """Handle every inbound WhatsApp event.

    ALWAYS returns 200 — Meta will retry endlessly on non-200 responses,
    which would duplicate messages.
    """
    try:
        body = await request.json()
    except Exception:
        logger.error("Failed to parse webhook body")
        return Response(status_code=200)

    try:
        _log_incoming(body)
        await _dispatch(body)
    except Exception as exc:
        # Never let an exception propagate — always return 200 to Meta
        logger.exception("Error processing webhook: %s", exc)

    return Response(status_code=200)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_incoming(body: dict) -> None:
    ts = datetime.now(tz=timezone.utc).isoformat()
    try:
        entry = body["entry"][0]
        change = entry["changes"][0]["value"]
        msg = change.get("messages", [{}])[0]
        sender = msg.get("from", "unknown")
        msg_type = msg.get("type", "unknown")
        logger.info("[%s] Incoming message — from=%s type=%s", ts, sender, msg_type)
    except (KeyError, IndexError):
        logger.info("[%s] Incoming non-message webhook event", ts)


async def _dispatch(body: dict) -> None:
    """Extract message fields and route to the correct handler."""
    try:
        messages = body["entry"][0]["changes"][0]["value"].get("messages")
    except (KeyError, IndexError):
        # Status updates, read receipts, etc. — ignore silently
        return

    if not messages:
        return

    msg = messages[0]
    sender: str = msg["from"]
    msg_type: str = msg.get("type", "")

    if msg_type == "text":
        text = msg.get("text", {}).get("body", "")
        await handle_text(sender, text)

    elif msg_type == "image":
        image_id = msg.get("image", {}).get("id", "")
        await handle_image(sender, image_id)

    elif msg_type == "audio":
        audio_id = msg.get("audio", {}).get("id", "")
        await handle_audio(sender, audio_id)

    else:
        logger.info("Unhandled message type: %s from %s", msg_type, sender)
