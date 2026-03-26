import io
import logging
import os
import urllib.request
from datetime import datetime

import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

# ── Page geometry ─────────────────────────────────────────────────────────────
_W, _H = A4          # 595.27 x 841.89 pt
_MARGIN = 40.0
_CONTENT_W = _W - 2 * _MARGIN

# ── Devanagari font setup ─────────────────────────────────────────────────────
_FONT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
)
_NOTO_PATH = os.path.join(_FONT_DIR, "NotoSansDevanagari-Regular.ttf")
_NOTO_URL = (
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
    "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
)
_HINDI_FONT = "NotoDevanagari"

_hindi_ready: bool | None = None   # None = unchecked, True/False = result


def _setup_hindi_font() -> bool:
    """Register NotoSansDevanagari with reportlab, downloading if absent.
    Returns True on success, False if the font is unavailable.
    """
    global _hindi_ready
    if _hindi_ready is not None:
        return _hindi_ready

    if not os.path.exists(_NOTO_PATH):
        os.makedirs(_FONT_DIR, exist_ok=True)
        try:
            logger.info("Downloading NotoSansDevanagari font…")
            urllib.request.urlretrieve(_NOTO_URL, _NOTO_PATH)
            logger.info("Font saved to %s", _NOTO_PATH)
        except Exception as exc:
            logger.warning("Font download failed (%s) — Hindi section will be skipped", exc)
            _hindi_ready = False
            return False

    try:
        pdfmetrics.registerFont(TTFont(_HINDI_FONT, _NOTO_PATH))
        _hindi_ready = True
        logger.info("NotoSansDevanagari registered with reportlab")
    except Exception as exc:
        logger.warning("Font registration failed (%s) — Hindi section will be skipped", exc)
        _hindi_ready = False

    return _hindi_ready


# ── Risk styling ──────────────────────────────────────────────────────────────
_RISK_COLORS = {
    "LOW_RISK":    colors.HexColor("#27AE60"),
    "MEDIUM_RISK": colors.HexColor("#E67E22"),
    "HIGH_RISK":   colors.HexColor("#E74C3C"),
}
_RISK_LABELS = {
    "LOW_RISK":    "LOW",
    "MEDIUM_RISK": "MEDIUM",
    "HIGH_RISK":   "HIGH",
}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _wrap_text(
    c: canvas.Canvas,
    text: str,
    max_w: float,
    font: str,
    size: float,
) -> list[str]:
    """Split *text* into lines that fit within *max_w* points."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_w = 0.0

    for word in words:
        w = c.stringWidth(word, font, size)
        gap = c.stringWidth(" ", font, size) if current else 0.0
        if current and current_w + gap + w > max_w:
            lines.append(" ".join(current))
            current = [word]
            current_w = w
        else:
            current.append(word)
            current_w += gap + w

    if current:
        lines.append(" ".join(current))
    return lines


def _text_block(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_w: float,
    font: str,
    size: float,
    leading: float = 16.0,
    fill_color=colors.HexColor("#1A1A2E"),
) -> float:
    """Draw word-wrapped text; return the y position below the last line."""
    c.setFillColor(fill_color)
    c.setFont(font, size)
    for line in _wrap_text(c, text, max_w, font, size):
        c.drawString(x, y, line)
        y -= leading
    return y


def _section_header(c: canvas.Canvas, label: str, y: float) -> float:
    """Draw a bold section title with a light divider. Returns y below divider."""
    c.setFillColor(colors.HexColor("#1A1A2E"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(_MARGIN, y, label)
    y -= 6
    c.setStrokeColor(colors.HexColor("#E0E0E0"))
    c.setLineWidth(0.5)
    c.line(_MARGIN, y, _W - _MARGIN, y)
    return y - 14


def _make_qr(url: str) -> ImageReader:
    """Render a QR code for *url* and return a reportlab-compatible image."""
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(patient_data: dict) -> bytes:
    """Build an A4 PDF screening report and return it as bytes.

    Expected keys in *patient_data*:
        phone_number   – sender's phone (last 4 digits used as patient ID)
        scan_date      – human-readable date string (defaults to now)
        risk_level     – "LOW_RISK" | "MEDIUM_RISK" | "HIGH_RISK"
        confidence     – float 0-1
        risk_emoji     – single emoji
        hindi_message  – patient-facing Hindi text
        english_message – ASHA-worker-facing English text
        nearest_center – dict {name, address, maps_link, distance}
    """
    hindi_ok = _setup_hindi_font()

    # Unpack
    phone      = patient_data.get("phone_number", "unknown")
    patient_id = phone[-4:] if len(phone) >= 4 else phone
    scan_date  = patient_data.get("scan_date", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    risk_level = patient_data.get("risk_level", "MEDIUM_RISK")
    confidence = patient_data.get("confidence", 0.0)
    risk_emoji = patient_data.get("risk_emoji", "🟡")
    hindi_msg  = patient_data.get("hindi_message", "")
    eng_msg    = patient_data.get("english_message", "")
    center     = patient_data.get("nearest_center", {})

    risk_color = _RISK_COLORS.get(risk_level, colors.orange)
    risk_label = _RISK_LABELS.get(risk_level, risk_level)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("AI Cancer Screening Report — CancerSetu")

    y = _H - _MARGIN   # cursor starts at top

    # ── HEADER ────────────────────────────────────────────────────────────────
    logo_w, logo_h = 72.0, 28.0

    # Logo placeholder (blue rounded rect)
    c.setFillColor(colors.HexColor("#1A73E8"))
    c.roundRect(_MARGIN, y - logo_h, logo_w, logo_h, 4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(_MARGIN + logo_w / 2, y - logo_h / 2 - 3, "CancerSetu")

    # Title + subtitle
    c.setFillColor(colors.HexColor("#1A1A2E"))
    c.setFont("Helvetica-Bold", 17)
    c.drawString(_MARGIN + logo_w + 12, y - 11, "AI Cancer Screening Report")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#5F6368"))
    c.drawString(_MARGIN + logo_w + 12, y - 24, "Powered by Google Gemini + TensorFlow")

    # Date / Patient ID (right-aligned)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#5F6368"))
    c.drawRightString(_W - _MARGIN, y - 8,  f"Date: {scan_date}")
    c.drawRightString(_W - _MARGIN, y - 20, f"Patient ID: ****{patient_id}")

    y -= logo_h + 10

    # Divider
    c.setStrokeColor(colors.HexColor("#BDBDBD"))
    c.setLineWidth(1)
    c.line(_MARGIN, y, _W - _MARGIN, y)
    y -= 20

    # ── RISK RESULT BOX ───────────────────────────────────────────────────────
    box_h = 66.0
    c.setFillColor(risk_color)
    c.roundRect(_MARGIN, y - box_h, _CONTENT_W, box_h, 8, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(_W / 2, y - 28, f"RISK LEVEL: {risk_label}")
    c.setFont("Helvetica", 13)
    c.drawCentredString(_W / 2, y - 50, f"Confidence: {round(confidence * 100)}%   {risk_emoji}")

    y -= box_h + 22

    # ── HINDI EXPLANATION ─────────────────────────────────────────────────────
    if hindi_ok and hindi_msg:
        y = _section_header(c, "रिपोर्ट सारांश (Hindi)", y)
        y = _text_block(c, hindi_msg, _MARGIN, y, _CONTENT_W, _HINDI_FONT, 11, leading=18)
        y -= 16
    elif hindi_msg:
        # Font unavailable — note it and skip
        y = _section_header(c, "Hindi Summary (Devanagari font unavailable)", y)
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.HexColor("#5F6368"))
        c.drawString(_MARGIN, y, "Please see the English summary below.")
        y -= 20

    # ── ENGLISH EXPLANATION ───────────────────────────────────────────────────
    y = _section_header(c, "Report Summary (English)", y)
    y = _text_block(c, eng_msg, _MARGIN, y, _CONTENT_W, "Helvetica", 11, leading=16)
    y -= 20

    # ── NEAREST SCREENING CENTER ──────────────────────────────────────────────
    if center:
        y = _section_header(c, "Nearest Screening Center", y)

        center_name = center.get("name", "N/A")
        center_addr = center.get("address", "")
        maps_link   = center.get("maps_link", "")
        distance    = center.get("distance", "")

        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.HexColor("#1A1A2E"))
        c.drawString(_MARGIN, y, center_name)
        y -= 15

        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#5F6368"))
        if center_addr:
            c.drawString(_MARGIN, y, center_addr)
            y -= 13
        if distance:
            c.drawString(_MARGIN, y, f"Distance: {distance}")
            y -= 13

        # QR code for Google Maps
        if maps_link:
            try:
                qr_size = 80.0
                qr_img = _make_qr(maps_link)
                c.drawImage(qr_img, _MARGIN, y - qr_size, qr_size, qr_size)
                c.setFont("Helvetica", 8)
                c.setFillColor(colors.HexColor("#5F6368"))
                c.drawString(_MARGIN + qr_size + 10, y - 18, "Scan to open in Google Maps")
                c.setFillColor(colors.HexColor("#1A73E8"))
                c.drawString(_MARGIN + qr_size + 10, y - 32, maps_link[:70])
                y -= qr_size + 10
            except Exception as exc:
                logger.warning("QR code generation failed: %s", exc)

        y -= 10

    # ── DISCLAIMER (pinned near bottom) ───────────────────────────────────────
    disclaimer_y = _MARGIN + 52
    c.setStrokeColor(colors.HexColor("#E0E0E0"))
    c.setLineWidth(0.5)
    c.line(_MARGIN, disclaimer_y, _W - _MARGIN, disclaimer_y)
    disclaimer_y -= 14

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#5F6368"))
    c.drawString(_MARGIN, disclaimer_y, "This is an AI screening tool, not a medical diagnosis.")
    disclaimer_y -= 12
    c.drawString(_MARGIN, disclaimer_y, "Please consult a qualified doctor.")

    # Watermark
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#1A73E8"))
    c.drawRightString(_W - _MARGIN, _MARGIN + 10, "CancerSetu — AI Health Screening")

    c.save()
    return buf.getvalue()
