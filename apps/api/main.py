from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Wildlife Protection AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVENTS: list[dict[str, Any]] = []
CAMERAS: list[dict[str, Any]] = []
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/v1/health")
def health():
    return {"status": "ok", "service": "api", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/events")
def events():
    return {"items": EVENTS, "count": len(EVENTS)}


@app.get("/v1/events/{event_id}")
def event(event_id: str):
    for item in EVENTS:
        if item["risk_event_id"] == event_id:
            return item
    raise HTTPException(404, "Event not found")


@app.get("/v1/cameras")
def cameras():
    return {"items": CAMERAS, "count": len(CAMERAS)}


@app.post("/v1/inference/video")
async def inference_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    media_id = str(uuid4())
    (UPLOAD_DIR / f"{media_id}{suffix}").write_bytes(await file.read())
    return {"job_id": media_id, "status": "queued", "filename": file.filename}


@app.post("/v1/events/{event_id}/review")
def review(event_id: str, decision: str):
    allowed = {"TRUE_POSITIVE", "FALSE_POSITIVE", "NEEDS_REVIEW", "REVIEWED"}
    if decision not in allowed:
        raise HTTPException(400, f"decision must be one of {sorted(allowed)}")
    for item in EVENTS:
        if item["risk_event_id"] == event_id:
            item["review"] = decision
            item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            return item
    raise HTTPException(404, "Event not found")
