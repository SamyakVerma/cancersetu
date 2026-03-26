import logging
import os

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore, storage

load_dotenv()

logger = logging.getLogger(__name__)

# ── Initialization (runs once at import) ──────────────────────────────────────

def _init_firebase() -> bool:
    """Initialize Firebase Admin SDK from service account key.
    Returns True on success, False if credentials are missing/invalid.
    """
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    if not cred_path:
        logger.warning("FIREBASE_CREDENTIALS_PATH not set — Firebase disabled")
        return False
    if not os.path.exists(cred_path):
        logger.warning("Firebase credentials file not found: %s — Firebase disabled", cred_path)
        return False

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
        })
        logger.info("Firebase Admin SDK initialized")
        return True
    except Exception as exc:
        logger.error("Firebase initialization failed: %s", exc)
        return False


_firebase_ready = _init_firebase()


def _db():
    return firestore.client() if _firebase_ready else None


def _bucket():
    return storage.bucket() if _firebase_ready else None


# ── Screening records ─────────────────────────────────────────────────────────

async def save_screening(
    phone: str,
    risk_level: str,
    confidence: float,
    hindi_message: str,
    english_message: str,
    pdf_url: str = "",
) -> str | None:
    """Save a screening result to Firestore. Returns the document ID or None."""
    db = _db()
    if db is None:
        logger.debug("Firebase not ready — screening not saved")
        return None

    try:
        doc_ref = db.collection("screenings").add({
            "phone": phone,
            "risk_level": risk_level,
            "confidence": round(confidence * 100, 1),
            "hindi_message": hindi_message,
            "english_message": english_message,
            "pdf_url": pdf_url,
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        doc_id = doc_ref[1].id
        logger.info("Screening saved: doc_id=%s phone=%s risk=%s", doc_id, phone, risk_level)
        return doc_id
    except Exception as exc:
        logger.error("Failed to save screening: %s", exc)
        return None


async def get_screening_history(phone: str, limit: int = 5) -> list[dict]:
    """Fetch last *limit* screenings for a phone number."""
    db = _db()
    if db is None:
        return []

    try:
        docs = (
            db.collection("screenings")
            .where("phone", "==", phone)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [{"id": d.id, **d.to_dict()} for d in docs]
    except Exception as exc:
        logger.error("Failed to fetch history for %s: %s", phone, exc)
        return []


# ── PDF storage ───────────────────────────────────────────────────────────────

async def upload_pdf(pdf_bytes: bytes, phone: str, doc_id: str) -> str:
    """Upload PDF report to Firebase Storage. Returns the gs:// URL or empty string."""
    bucket = _bucket()
    if bucket is None:
        return ""

    try:
        safe_phone = phone.replace("+", "").replace(" ", "")
        blob_path = f"reports/{safe_phone}/{doc_id}.pdf"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        url = f"gs://{bucket.name}/{blob_path}"
        logger.info("PDF uploaded: %s", url)
        return url
    except Exception as exc:
        logger.error("PDF upload failed: %s", exc)
        return ""


# ── Live stats for /stats endpoint ───────────────────────────────────────────

async def get_stats() -> dict:
    """Aggregate screening stats from Firestore for the /stats endpoint."""
    db = _db()
    if db is None:
        return {}

    try:
        docs = list(db.collection("screenings").stream())
        total = len(docs)
        positive = sum(
            1 for d in docs
            if d.to_dict().get("risk_level") in ("MEDIUM_RISK", "HIGH_RISK")
        )
        phones = {d.to_dict().get("phone") for d in docs}
        return {
            "total_screenings": total,
            "positive_detections": positive,
            "unique_patients": len(phones),
        }
    except Exception as exc:
        logger.error("Stats fetch failed: %s", exc)
        return {}
