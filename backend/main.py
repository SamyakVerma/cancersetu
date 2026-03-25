import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routers import whatsapp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="JanArogya API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} — {response.status_code} ({duration}ms)")
    return response


app.include_router(whatsapp.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "project": "JanArogya", "version": "1.0"}


@app.get("/stats")
def stats():
    return {
        "total_screenings": 0,
        "positive_detections": 0,
        "reports_generated": 0,
        "whatsapp_interactions": 0,
        "states_covered": [],
        "note": "Mock data — connect to Firestore for live stats",
    }
