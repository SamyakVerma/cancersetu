import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_GRAPH_API_BASE = "https://graph.facebook.com/v18.0"
_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

_HEADERS = {
    "Authorization": f"Bearer {_TOKEN}",
    "Content-Type": "application/json",
}


def _messages_url() -> str:
    return f"{_GRAPH_API_BASE}/{_PHONE_ID}/messages"


async def send_message(to: str, text: str) -> dict[str, Any]:
    """Send a plain text WhatsApp message."""
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(_messages_url(), json=payload, headers=_HEADERS)
    if resp.status_code != 200:
        logger.error("send_message failed: %s — %s", resp.status_code, resp.text)
    return resp.json()


async def download_media(media_id: str) -> bytes:
    """Fetch media URL from Graph API, then download and return raw bytes."""
    media_url_endpoint = f"{_GRAPH_API_BASE}/{media_id}"
    auth_headers = {"Authorization": f"Bearer {_TOKEN}"}

    async with httpx.AsyncClient(timeout=15) as client:
        # Step 1: get the download URL
        meta_resp = await client.get(media_url_endpoint, headers=auth_headers)
        meta_resp.raise_for_status()
        download_url = meta_resp.json()["url"]

        # Step 2: download the actual bytes
        media_resp = await client.get(download_url, headers=auth_headers)
        media_resp.raise_for_status()

    logger.info("Downloaded media %s — %d bytes", media_id, len(media_resp.content))
    return media_resp.content


async def send_document(
    to: str,
    file_bytes: bytes,
    filename: str,
    caption: str = "",
) -> dict[str, Any]:
    """Upload and send a document (e.g. PDF report) to a WhatsApp user."""
    upload_url = f"{_GRAPH_API_BASE}/{_PHONE_ID}/media"
    upload_headers = {"Authorization": f"Bearer {_TOKEN}"}

    async with httpx.AsyncClient(timeout=30) as client:
        # Upload the file first to get a media_id
        upload_resp = await client.post(
            upload_url,
            headers=upload_headers,
            files={"file": (filename, file_bytes, "application/pdf")},
            data={"messaging_product": "whatsapp"},
        )
        upload_resp.raise_for_status()
        media_id = upload_resp.json()["id"]

        # Send the document message
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "document",
            "document": {
                "id": media_id,
                "filename": filename,
                "caption": caption,
            },
        }
        send_resp = await client.post(_messages_url(), json=payload, headers=_HEADERS)

    if send_resp.status_code != 200:
        logger.error("send_document failed: %s — %s", send_resp.status_code, send_resp.text)
    return send_resp.json()


async def send_buttons(
    to: str,
    body_text: str,
    buttons_list: list[dict[str, str]],
) -> dict[str, Any]:
    """Send an interactive button message.

    buttons_list format: [{"id": "btn_id", "title": "Button Label"}, ...]
    Maximum 3 buttons (WhatsApp API limit).
    """
    if len(buttons_list) > 3:
        raise ValueError("WhatsApp allows a maximum of 3 interactive buttons.")

    buttons = [
        {"type": "reply", "reply": {"id": b["id"], "title": b["title"]}}
        for b in buttons_list
    ]

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {"buttons": buttons},
        },
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(_messages_url(), json=payload, headers=_HEADERS)

    if resp.status_code != 200:
        logger.error("send_buttons failed: %s — %s", resp.status_code, resp.text)
    return resp.json()
