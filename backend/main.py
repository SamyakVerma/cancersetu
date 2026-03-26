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
async def stats():
    from services.firebase_service import get_stats
    live = await get_stats()
    return {
        "total_screenings": live.get("total_screenings", 0),
        "positive_detections": live.get("positive_detections", 0),
        "unique_patients": live.get("unique_patients", 0),
        "reports_generated": live.get("total_screenings", 0),
        "whatsapp_interactions": live.get("total_screenings", 0),
    }
