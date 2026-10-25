"""FastAPI service: upload a food photo, get the top-3 dishes plus approximate nutrition.

Run locally:
    ARTIFACTS_DIR=artifacts uvicorn foodvision.api:app --reload
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from . import __version__
from .nutrition import DISCLAIMER, lookup
from .predictor import Predictor

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.5
STATIC_DIR = Path(__file__).with_name("static")


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    artifacts_dir = Path(artifacts_dir or os.environ.get("ARTIFACTS_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not (artifacts_dir / "model.pt").exists():
            raise RuntimeError(
                f"model.pt not found in {artifacts_dir.resolve()}. "
                "Download the training artifacts and set ARTIFACTS_DIR.")
        app.state.predictor = Predictor(artifacts_dir)
        yield

    app = FastAPI(title="Food Vision API", version=__version__, lifespan=lifespan,
                  description="Classifies 20 Indian dishes from a photo.")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health(request: Request):
        p: Predictor = request.app.state.predictor
        return {"status": "ok", "version": __version__, "model": p.config["model_name"],
                "num_classes": len(p.class_names)}

    @app.get("/classes")
    def classes(request: Request):
        return {"classes": request.app.state.predictor.class_names}

    # Plain `def` (not async): FastAPI runs it in a thread pool, so CPU-bound
    # inference doesn't block the event loop.
    @app.post("/predict")
    def predict(request: Request, file: UploadFile = File(...), top_k: int = 3):
        if not 1 <= top_k <= 10:
            raise HTTPException(422, "top_k must be between 1 and 10")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(400, "Empty file")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image too large (max 8 MB)")

        predictor: Predictor = request.app.state.predictor
        t0 = time.perf_counter()
        try:
            preds = predictor.predict(data, top_k=top_k)
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "File is not a readable image")
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        for p in preds:
            info = lookup(p["dish"]) or {}
            p["name"] = info.get("name", p["dish"].replace("_", " ").title())
            p["nutrition"] = {k: info[k] for k in ("serving", "kcal", "protein_g", "carbs_g", "fat_g")
                              if k in info} or None

        confident = preds[0]["confidence"] >= CONFIDENCE_THRESHOLD
        return {
            "predictions": preds,
            "confident": confident,
            "message": None if confident else (
                "Low confidence. This model only knows 20 Indian dishes; "
                "the photo may show something else."),
            "nutrition_disclaimer": DISCLAIMER,
            "latency_ms": latency_ms,
        }

    return app


app = create_app()
