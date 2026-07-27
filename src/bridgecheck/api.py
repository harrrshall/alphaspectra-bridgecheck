"""FastAPI service and bundled browser application."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .artifact import BridgeArtifact
from .audit import PairedSpectrum, audit_paired_spectra
from .predict import ContractError, predict_spectrum
from .schema import AuditRequest, PredictRequest


@lru_cache(maxsize=1)
def default_artifact() -> BridgeArtifact:
    return BridgeArtifact.load()


def create_app(artifact: BridgeArtifact | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:  # pragma: no cover - exercised by installation smoke test
        raise RuntimeError("install the 'api' extra to run the service") from error

    model = artifact or default_artifact()
    app = FastAPI(
        title="AlphaSpectra BridgeCheck",
        version=model.manifest["version"],
        description="Physics-grounded, explicitly model-derived SWIR candidates and paired-data audits.",
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        size = request.headers.get("content-length")
        if size is not None:
            try:
                content_length = int(size)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})
            if content_length < 0:
                return JSONResponse(status_code=400, content={"detail": "invalid Content-Length"})
            if content_length > 8_000_000:
                return JSONResponse(status_code=413, content={"detail": "request body exceeds 8 MB"})
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "model_id": model.manifest["model_id"],
            "artifact_array_sha256": model.manifest["artifact"]["array_sha256"],
        }

    @app.get("/v1/model")
    def model_info():
        return model.public_info()

    @app.post("/v1/predict")
    def predict(request: PredictRequest):
        if request.reflectance_unit != "fraction":
            raise HTTPException(status_code=422, detail="reflectance_unit must be 'fraction'")
        try:
            result = predict_spectrum(
                model, request.wavelength_nm, request.reflectance, neighbors=request.neighbors
            )
        except ContractError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        response = result.to_dict()
        response["observed"] = {
            "origin": "measured",
            "wavelength_nm": request.wavelength_nm,
            "reflectance": request.reflectance,
            "observed_band_mask": [True] * len(request.wavelength_nm),
        }
        return response

    @app.post("/v1/audit")
    def audit(request: AuditRequest):
        try:
            samples = [
                PairedSpectrum(
                    sample_id=row.sample_id,
                    group_id=row.group_id,
                    context_wavelength_nm=np.asarray(row.context_wavelength_nm, dtype=np.float64),
                    context_reflectance=np.asarray(row.context_reflectance, dtype=np.float64),
                    target_wavelength_nm=np.asarray(row.target_wavelength_nm, dtype=np.float64),
                    target_reflectance=np.asarray(row.target_reflectance, dtype=np.float64),
                )
                for row in request.samples
            ]
            return audit_paired_spectra(
                model, samples, bootstrap_repeats=request.bootstrap_repeats
            )
        except ContractError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    model_dir = Path(__file__).resolve().parent / "model"
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/model", StaticFiles(directory=model_dir), name="model")
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
