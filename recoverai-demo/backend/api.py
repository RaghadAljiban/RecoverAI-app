import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.deps import get_classifier
from telerehab.config import SUPPORTED_EXERCISES
from telerehab.overlay import write_mp4

router = APIRouter()

OVERLAY_DIR = Path(tempfile.gettempdir()) / "telerehab_overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@router.get("/exercises")
def list_exercises() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "exercises": [
            {"id": ex_id, "name": name}
            for name, ex_id in SUPPORTED_EXERCISES.items()
        ]
    }


@router.post("/classify")
async def classify(
    video: UploadFile = File(...),
    exercise_name: str = Form(...),
) -> JSONResponse:
    if exercise_name not in SUPPORTED_EXERCISES:
        raise HTTPException(status_code=400, detail=f"Unsupported exercise: {exercise_name!r}")

    suffix = Path(video.filename or "upload").suffix.lower()
    if suffix and suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix}")
    if not suffix:
        suffix = ".mp4"

    tmp_video = OVERLAY_DIR / f"in_{uuid.uuid4().hex}{suffix}"
    try:
        with tmp_video.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        classifier = get_classifier()
        try:
            result = classifier.classify(str(tmp_video), exercise_name)
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        overlay_path = write_mp4(result.overlay_frames, result.fps)
        overlay_id = Path(overlay_path).name if overlay_path else None
        if overlay_path:
            final_path = OVERLAY_DIR / f"overlay_{uuid.uuid4().hex}.mp4"
            shutil.move(overlay_path, final_path)
            overlay_id = final_path.name

        return JSONResponse({
            "prediction_label": result.prediction_label,
            "prediction_idx": int(result.prediction_idx),
            "confidence": float(result.confidence),
            "prob_correct": float(result.prob_correct),
            "prob_incorrect": float(result.prob_incorrect),
            "logit": float(result.logit),
            "threshold": float(result.threshold),
            "exercise_id": int(result.exercise_id),
            "exercise_name": result.exercise_name,
            "predicted_exercise_id": int(result.predicted_exercise_id),
            "predicted_exercise_name": result.predicted_exercise_name,
            "exercise_match": bool(result.exercise_match),
            "exercise_confidence": float(result.exercise_confidence),
            "fps": float(result.fps),
            "info": result.info,
            "debug": result.debug,
            "overlay_url": f"/overlay/{overlay_id}" if overlay_id else None,
        })
    finally:
        try:
            tmp_video.unlink(missing_ok=True)
        except OSError:
            pass


def serve_overlay(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = OVERLAY_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path, media_type="video/mp4")
