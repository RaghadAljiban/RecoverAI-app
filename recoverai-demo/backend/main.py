import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")
for name in ("mediapipe", "tensorflow", "absl"):
    logging.getLogger(name).setLevel(logging.ERROR)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import router as api_router
from backend.api import serve_overlay
from backend.deps import get_classifier
from telerehab.config import SERVER_HOST, SERVER_PORT

FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="TeleRehab")

app.include_router(api_router, prefix="/api")
app.get("/overlay/{filename}")(serve_overlay)


@app.on_event("startup")
def warm_classifier() -> None:
    clf = get_classifier()
    print(f"[telerehab] device: {clf.device}")
    print(f"[telerehab] checkpoint loaded ({sum(p.numel() for p in clf.bundle.model.parameters()):,} params)")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def run() -> None:
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
