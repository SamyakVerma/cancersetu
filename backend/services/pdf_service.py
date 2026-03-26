"""
PDF report generator for CancerSetu screening results.
Supports Hindi (Devanagari) via NotoSansDevanagari font.
"""

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
PAGE_W, PAGE_H = A4          # 595.27 × 841.89 pt
MARGIN = 36.0
CONTENT_W = PAGE_W - 2 * MARGIN

# ── Palette ───────────────────────────────────────────────────────────────────
C_DARK_GREEN  = colors.HexColor("#1B5E20")
C_MED_GREEN   = colors.HexColor("#388E3C")
C_RED         = colors.HexColor("#B71C1C")
C_ORANGE      = colors.HexColor("#E65100")
C_YELLOW_BG   = colors.HexColor("#FFF9C4")
C_YELLOW_BDR  = colors.HexColor("#F9A825")
C_ACTION_BG   = colors.HexColor("#FFEBEE")
C_ACTION_BDR  = colors.HexColor("#C62828")
C_GRAY        = colors.HexColor("#616161")
C_LIGHT_GRAY  = colors.HexColor("#F5F5F5")

RISK_BG = {
    "HIGH_RISK":   C_RED,
    "MEDIUM_RISK": C_ORANGE,
    "LOW_RISK":    C_DARK_GREEN,
}
RISK_LABEL = {
    "HIGH_RISK": "HIGH",
    "MEDIUM_RISK": "MEDIUM",
    "LOW_RISK": "LOW",
}

ACTION_TIMEFRAME = {
    "urgent":       "24-48 hours",
    "within_week":  "within 7 days",
    "within_month": "within 30 days",
}

CANCER_HELPLINE = "1800-11-2345  (Toll-Free Cancer Helpline)"

# ── Devanagari font ───────────────────────────────────────────────────────────
ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets"))
FONT_PATH   = os.path.join(ASSETS_DIR, "NotoSansDevanagari.ttf")
FONT_URL    = (
    "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
    "NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
)
HINDI_FONT = "NotoDevanagari"

_hindi_ready: bool | None = None


def _setup_hindi_font() -> bool:
    global _hindi_ready
    if _hindi_ready is not None:
        return _hindi_ready

    if not os.path.exists(FONT_PATH):
        os.makedirs(ASSETS_DIR, exist_ok=True)
        try:
            logger.info("Downloading NotoSansDevanagari font…")
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
            logger.info("Font saved: %s", FONT_PATH)
        except Exception as exc:
            logger.warning("Font download failed: %s — Hindi sections will be skipped", exc)
            _hindi_ready = False
            return False

    try:
        pdfmetrics.registerFont(TTFont(HINDI_FONT, FONT_PATH))
        _hindi_ready = True
        logger.info("NotoSansDevanagari registered with reportlab")
    except Exception as exc:
        logger.warning("Font registration failed: %s — Hindi sections will be skipped", exc)
        _hindi_ready = False

    return _hindi_ready


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _wrap(c: canvas.Canvas, text: str, max_w: float, font: str, size: float) -> list[str]:
    """Word-wrap *text* to fit within *max_w* points."""
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_w = 0.0
    for word in words:
        w   = c.stringWidth(word, font, size)
        gap = c.stringWidth(" ", font, size) if cur else 0.0
        if cur and cur_w + gap + w > max_w:
            lines.append(" ".join(cur))
            cur, cur_w = [word], w
        else:
            cur.append(word)
            cur_w += gap + w
    if cur:
        lines.append(" ".join(cur))
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
    color=colors.black,
) -> float:
    """Draw word-wrapped text; return y below last line."""
    c.setFillColor(color)
    c.setFont(font, size)
    for line in _wrap(c, text, max_w, font, size):
        c.drawString(x, y, line)
        y -= leading
    return y


def _section_divider(c: canvas.Canvas, label: str, y: float, font: str = "Helvetica-Bold") -> float:
    """Bold section label with a green underline. Returns y below divider."""
    c.setFillColor(C_DARK_GREEN)
    c.setFont(font, 12)
    c.drawString(MARGIN, y, label)
    y -= 7
    c.setStrokeColor(C_DARK_GREEN)
    c.setLineWidth(0.75)
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    return y - 13


def _make_qr(url: str) -> ImageReader:
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def _report_id(phone_hash: str) -> str:
    return f"CS-{phone_hash[:8].upper()}-{datetime.now().strftime('%Y%m%d')}"


# ── Page 1 — Patient report ───────────────────────────────────────────────────

def generate_report(data: dict) -> bytes:
    """Build a single-patient A4 screening report. Returns PDF bytes.

    Required keys in *data*:
        phone_hash, scan_date, scan_type, image_bytes (unused / optional),
        risk_level, confidence, hindi_message, english_message,
        action_urgency, centers (list of dicts), disclaimer_hindi, disclaimer_english
    """
    hindi_ok = _setup_hindi_font()

    phone_hash     = data.get("phone_hash", "unknown")
    scan_date      = data.get("scan_date", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    scan_type      = data.get("scan_type", "General")
    risk_level     = data.get("risk_level", "MEDIUM_RISK")
    confidence     = data.get("confidence", 0.0)
    hindi_msg      = data.get("hindi_message", "")
    eng_msg        = data.get("english_message", "")
    action_urgency = data.get("action_urgency", "monitor")
    centers        = data.get("centers", [])
    disc_hindi     = data.get("disclaimer_hindi",
                              "यह एक AI स्क्रीनिंग है, डॉक्टर से मिलें।")
    disc_eng       = data.get("disclaimer_english",
                              "This is an AI screening tool, not a medical diagnosis.")
    report_id      = _report_id(phone_hash)

    risk_color = RISK_BG.get(risk_level, C_ORANGE)
    risk_label = RISK_LABEL.get(risk_level, risk_level)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"CancerSetu Screening Report — {report_id}")

    # ── HEADER BAR ─────────────────────────────────────────────────────────────
    HDR_H = 64.0
    c.setFillColor(C_DARK_GREEN)
    c.rect(0, PAGE_H - HDR_H, PAGE_W, HDR_H, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(MARGIN, PAGE_H - 30, "CancerSetu")
    c.setFont("Helvetica", 10)
    c.drawString(MARGIN, PAGE_H - 48, "AI Cancer Screening  |  GSC 2025")

    c.setFont("Helvetica", 9)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 26, f"Report: {report_id}")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 40, f"Date: {scan_date}")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 54, f"Type: {scan_type.title()}")

    y = PAGE_H - HDR_H - 18

    # ── RISK BADGE ─────────────────────────────────────────────────────────────
    badge_w, badge_h = 280.0, 108.0
    badge_x = (PAGE_W - badge_w) / 2
    badge_y = y - badge_h

    c.setFillColor(risk_color)
    c.roundRect(badge_x, badge_y, badge_w, badge_h, 10, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica", 10)
    c.drawCentredString(PAGE_W / 2, badge_y + badge_h - 22, "RISK LEVEL")
    c.setFont("Helvetica-Bold", 32)
    c.drawCentredString(PAGE_W / 2, badge_y + badge_h - 62, risk_label)
    c.setFont("Helvetica", 13)
    c.drawCentredString(PAGE_W / 2, badge_y + 20, f"Confidence: {round(confidence * 100)}%")

    y = badge_y - 18

    # ── DISCLAIMER BOX ─────────────────────────────────────────────────────────
    PAD = 10
    eng_lines = _wrap(c, disc_eng, CONTENT_W - PAD * 2 - 16, "Helvetica", 10)
    hi_lines  = (_wrap(c, disc_hindi, CONTENT_W - PAD * 2 - 16, HINDI_FONT, 10)
                 if hindi_ok else [])
    disc_h = PAD * 2 + 20 + (len(eng_lines) + len(hi_lines)) * 14 + 6

    c.setFillColor(C_YELLOW_BG)
    c.setStrokeColor(C_YELLOW_BDR)
    c.setLineWidth(1.5)
    c.roundRect(MARGIN, y - disc_h, CONTENT_W, disc_h, 6, fill=1, stroke=1)

    ty = y - PAD - 16
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN + PAD, ty, "!  THIS IS NOT A DIAGNOSIS")
    ty -= 16

    c.setFont("Helvetica", 10)
    for line in eng_lines:
        c.drawString(MARGIN + PAD + 12, ty, line)
        ty -= 14

    if hindi_ok:
        c.setFont(HINDI_FONT, 10)
        for line in hi_lines:
            c.drawString(MARGIN + PAD + 12, ty, line)
            ty -= 14

    y = y - disc_h - 16

    # ── HINDI EXPLANATION ──────────────────────────────────────────────────────
    if hindi_msg:
        if hindi_ok:
            y = _section_divider(c, "रिपोर्ट सारांश", y, font=HINDI_FONT)
            y = _text_block(c, hindi_msg, MARGIN, y, CONTENT_W, HINDI_FONT, 12, leading=20)
        else:
            y = _section_divider(c, "Hindi Summary (font unavailable)", y)
            c.setFont("Helvetica-Oblique", 9)
            c.setFillColor(C_GRAY)
            c.drawString(MARGIN, y, "Please see the English summary below.")
            y -= 14
        y -= 14

    # ── ENGLISH EXPLANATION ────────────────────────────────────────────────────
    y = _section_divider(c, "Report Summary", y)
    y = _text_block(c, eng_msg, MARGIN, y, CONTENT_W, "Helvetica", 11, leading=16)
    y -= 16

    # ── ACTION REQUIRED ────────────────────────────────────────────────────────
    timeframe = ACTION_TIMEFRAME.get(action_urgency)
    if timeframe:
        act_h = 60.0
        c.setFillColor(C_ACTION_BG)
        c.setStrokeColor(C_ACTION_BDR)
        c.setLineWidth(1.5)
        c.roundRect(MARGIN, y - act_h, CONTENT_W, act_h, 6, fill=1, stroke=1)

        c.setFillColor(C_ACTION_BDR)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(MARGIN + 12, y - 18, "ACTION REQUIRED")
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.black)
        c.drawString(MARGIN + 12, y - 34, f"Please visit a doctor: {timeframe}")
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(C_DARK_GREEN)
        c.drawString(MARGIN + 12, y - 50, f"Helpline: {CANCER_HELPLINE}")

        y = y - act_h - 16

    # ── NEAREST CENTERS ────────────────────────────────────────────────────────
    if centers:
        y = _section_divider(c, "Nearest Screening Centers", y)

        for i, center in enumerate(centers[:3]):
            row_h = 64.0
            c.setFillColor(C_LIGHT_GRAY if i % 2 == 0 else colors.white)
            c.rect(MARGIN, y - row_h, CONTENT_W, row_h, fill=1, stroke=0)

            # Name + details
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.black)
            c.drawString(MARGIN + 8, y - 14, center.get("name", ""))
            c.setFont("Helvetica", 9)
            c.setFillColor(C_GRAY)
            if center.get("address"):
                c.drawString(MARGIN + 8, y - 27, center["address"])
            if center.get("distance"):
                c.drawString(MARGIN + 8, y - 40, f"Distance: {center['distance']}")

            # QR code
            maps_link = center.get("maps_link", "")
            if maps_link:
                try:
                    qr_sz = 54.0
                    qr_img = _make_qr(maps_link)
                    c.drawImage(qr_img, PAGE_W - MARGIN - qr_sz - 6, y - row_h + 4, qr_sz, qr_sz)
                    c.setFont("Helvetica", 7)
                    c.setFillColor(C_GRAY)
                    c.drawRightString(PAGE_W - MARGIN - 6 - qr_sz / 2 + qr_sz,
                                      y - row_h + 2, "Maps")
                except Exception as exc:
                    logger.warning("QR failed for center %d: %s", i, exc)

            y -= row_h + 4

        y -= 6

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    FOOTER_H = 50.0
    c.setFillColor(C_LIGHT_GRAY)
    c.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)
    c.setStrokeColor(C_DARK_GREEN)
    c.setLineWidth(0.5)
    c.line(0, FOOTER_H, PAGE_W, FOOTER_H)

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(C_DARK_GREEN)
    c.drawCentredString(PAGE_W / 2, FOOTER_H - 13,
                        "Generated by CancerSetu — Google Solution Challenge 2025")
    c.setFont("Helvetica", 8)
    c.setFillColor(C_GRAY)
    c.drawCentredString(PAGE_W / 2, FOOTER_H - 25, "Powered by Gemini AI + TensorFlow")
    c.drawCentredString(PAGE_W / 2, FOOTER_H - 37, "For queries: cancersetu@gmail.com")
    c.setFont("Helvetica", 7)
    c.drawRightString(PAGE_W - MARGIN, FOOTER_H - 37, "Page 1")

    c.save()
    return buf.getvalue()


# ── Batch report for ASHA workers ─────────────────────────────────────────────

def generate_asha_batch_report(scans: list) -> bytes:
    """Multi-patient summary PDF for ASHA workers. Returns PDF bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("CancerSetu — ASHA Worker Batch Report")

    # ── HEADER ─────────────────────────────────────────────────────────────────
    HDR_H = 62.0
    c.setFillColor(C_DARK_GREEN)
    c.rect(0, PAGE_H - HDR_H, PAGE_W, HDR_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN, PAGE_H - 28, "CancerSetu — ASHA Worker Report")
    c.setFont("Helvetica", 10)
    c.drawString(MARGIN, PAGE_H - 46, "AI Cancer Screening  |  GSC 2025")
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 34, datetime.now().strftime("%d %b %Y"))

    y = PAGE_H - HDR_H - 18

    # ── SUMMARY STATS ──────────────────────────────────────────────────────────
    total    = len(scans)
    high     = sum(1 for s in scans if s.get("risk_level") == "HIGH_RISK")
    referred = sum(1 for s in scans if s.get("action_urgency") not in ("monitor", None, ""))

    stat_w = CONTENT_W / 3
    stat_h = 54.0
    stat_y = y - stat_h

    for i, (label, val, col) in enumerate([
        ("Total Screened",     total,    C_DARK_GREEN),
        ("High Risk",          high,     C_RED),
        ("Referred to Doctor", referred, C_ORANGE),
    ]):
        sx = MARGIN + i * stat_w
        c.setFillColor(col)
        c.roundRect(sx + 4, stat_y, stat_w - 8, stat_h, 6, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(sx + stat_w / 2, stat_y + stat_h - 30, str(val))
        c.setFont("Helvetica", 8)
        c.drawCentredString(sx + stat_w / 2, stat_y + 9, label)

    y = stat_y - 20

    # ── TABLE HEADER ───────────────────────────────────────────────────────────
    cols = [("Patient ID", 70), ("Date", 65), ("Scan Type", 65),
            ("Risk Level", 75), ("Action", 70), ("Center", 120)]
    col_x = [MARGIN + sum(w for _, w in cols[:i]) for i in range(len(cols))]
    ROW_H = 20.0

    c.setFillColor(C_DARK_GREEN)
    c.rect(MARGIN, y - ROW_H, CONTENT_W, ROW_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8)
    for (label, _), cx in zip(cols, col_x):
        c.drawString(cx + 4, y - 14, label)
    y -= ROW_H

    # ── TABLE ROWS ─────────────────────────────────────────────────────────────
    ROW_BG = {
        "HIGH_RISK":   colors.HexColor("#FFEBEE"),
        "MEDIUM_RISK": colors.HexColor("#FFF3E0"),
        "LOW_RISK":    colors.HexColor("#E8F5E9"),
    }

    for scan in scans:
        if y < MARGIN + 60:
            c.showPage()
            y = PAGE_H - MARGIN

        risk = scan.get("risk_level", "MEDIUM_RISK")
        c.setFillColor(ROW_BG.get(risk, colors.white))
        c.rect(MARGIN, y - ROW_H, CONTENT_W, ROW_H, fill=1, stroke=0)

        # Cell dividers
        c.setStrokeColor(colors.HexColor("#E0E0E0"))
        c.setLineWidth(0.3)
        for cx in col_x[1:]:
            c.line(cx, y - ROW_H, cx, y)
        c.line(MARGIN, y - ROW_H, PAGE_W - MARGIN, y - ROW_H)

        row = [
            scan.get("phone_hash", "")[:8],
            scan.get("scan_date", "")[:10],
            scan.get("scan_type", "").title(),
            RISK_LABEL.get(risk, risk),
            "YES" if scan.get("action_urgency") not in ("monitor", None, "") else "No",
            ((scan.get("centers") or [{}])[0].get("name", "N/A"))[:20],
        ]
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.black)
        for val, cx, (_, cw) in zip(row, col_x, cols):
            c.drawString(cx + 4, y - 13, str(val)[:int(cw / 5)])
        y -= ROW_H

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    FOOTER_H = 44.0
    c.setFillColor(C_LIGHT_GRAY)
    c.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)
    c.setStrokeColor(C_DARK_GREEN)
    c.setLineWidth(0.5)
    c.line(0, FOOTER_H, PAGE_W, FOOTER_H)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(C_DARK_GREEN)
    c.drawCentredString(PAGE_W / 2, FOOTER_H - 14,
                        "CancerSetu — Google Solution Challenge 2025")
    c.setFont("Helvetica", 8)
    c.setFillColor(C_GRAY)
    c.drawCentredString(PAGE_W / 2, FOOTER_H - 26, "Powered by Gemini AI + TensorFlow")

    c.save()
    return buf.getvalue()
