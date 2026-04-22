"""
REST API Routes
===============
All HTTP endpoints for the SpecGenAI system.

Endpoints:
  POST   /upload                     Upload a spec or sample file
  POST   /train                      Parse + learn → create SchemaModel
  GET    /specs                      List all trained specifications
  GET    /specs/{spec_id}            Get full spec detail
  DELETE /specs/{spec_id}            Delete a specification
  POST   /generate                   Generate synthetic data
  GET    /download/{generation_id}   Download a generated file
  GET    /health                     Health check
  GET    /stats                      Storage statistics
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.models.requests import (
    GenerateRequest, GenerateResponse,
    HealthResponse, ListSpecsResponse,
    GetSpecResponse, TrainRequest, TrainResponse, UploadResponse,
)
from app.services.spec_service import SpecGenAIService
from app.core.config import settings
from app.core.exceptions import (
    ParseError, TrainingError, GenerationError,
    SpecNotFoundError, StorageError, ValidationError,
    UnsupportedSpecTypeError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency ────────────────────────────────────────────────────────────────

def get_service() -> SpecGenAIService:
    return SpecGenAIService()


# ── Error handler helper ──────────────────────────────────────────────────────

def _http(exc: Exception) -> HTTPException:
    """Map domain exceptions to HTTP status codes."""
    if isinstance(exc, (SpecNotFoundError,)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValidationError, ParseError, UnsupportedSpecTypeError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (TrainingError, GenerationError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, StorageError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Application health check."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/stats", tags=["System"])
async def storage_stats(svc: SpecGenAIService = Depends(get_service)):
    """Return storage statistics."""
    try:
        return await svc.get_storage_stats()
    except Exception as exc:
        raise _http(exc)


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    tags=["Upload"],
    summary="Upload a specification or sample data file",
)
async def upload_file(
    file: UploadFile = File(..., description="Spec file or sample data file"),
    spec_type: str = Form(
        ...,
        description="One of: vcf, ach, json, sample",
        pattern="^(vcf|ach|json|sample)$",
    ),
    svc: SpecGenAIService = Depends(get_service),
):
    """
    Upload a raw specification file or sample data file.

    Returns an **upload_id** which you pass to `/train`.

    Supported spec_type values:
    - **vcf**    – VISA Card File specification or native VCF data
    - **ach**    – ACH/NACHA file or JSON spec
    - **json**   – Custom JSON schema or SpecGenAI native JSON spec
    - **sample** – Any delimited/fixed-width sample data file
    """
    content = await file.read()
    try:
        return await svc.upload_file(
            filename=file.filename or "upload.bin",
            content=content,
            spec_type=spec_type,
        )
    except Exception as exc:
        raise _http(exc)


# ── Train ─────────────────────────────────────────────────────────────────────

@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=201,
    tags=["Training"],
    summary="Parse and learn specification from uploaded files",
)
async def train_spec(
    req: TrainRequest,
    svc: SpecGenAIService = Depends(get_service),
):
    """
    Parse, analyse and learn a SchemaModel from one or more uploaded files.

    **Flow**:
    1. Resolve each `upload_id` to its uploaded file path
    2. Parse spec files using the appropriate parser
    3. Parse any sample data files for statistical enrichment
    4. Run the SpecLearner pipeline (field inference, pattern detection)
    5. Persist the trained SchemaModel
    6. Return spec_id, records/fields learned, inference stats

    **Example body**:
    ```json
    {
      "upload_ids": ["abc123", "def456"],
      "spec_name": "MyACHSpec",
      "spec_type": "ach",
      "description": "Internal payroll ACH file"
    }
    ```
    """
    try:
        return await svc.train_spec(req)
    except Exception as exc:
        raise _http(exc)


# ── Specifications ────────────────────────────────────────────────────────────

@router.get(
    "/specs",
    response_model=ListSpecsResponse,
    tags=["Specifications"],
    summary="List all trained specifications",
)
async def list_specs(svc: SpecGenAIService = Depends(get_service)):
    """Return a summary list of all trained SchemaModels."""
    try:
        return await svc.list_specs()
    except Exception as exc:
        raise _http(exc)


@router.get(
    "/specs/{spec_id}",
    response_model=GetSpecResponse,
    tags=["Specifications"],
    summary="Get full detail of a trained specification",
)
async def get_spec(spec_id: str, svc: SpecGenAIService = Depends(get_service)):
    """
    Retrieve the full SchemaModel for the given **spec_id**, including
    all record definitions, field definitions and constraints.
    """
    try:
        return await svc.get_spec(spec_id)
    except Exception as exc:
        raise _http(exc)


@router.delete(
    "/specs/{spec_id}",
    status_code=204,
    tags=["Specifications"],
    summary="Delete a specification",
)
async def delete_spec(spec_id: str, svc: SpecGenAIService = Depends(get_service)):
    """Permanently delete a trained specification."""
    try:
        await svc.delete_spec(spec_id)
    except Exception as exc:
        raise _http(exc)


# ── Generate ──────────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=201,
    tags=["Generation"],
    summary="Generate a synthetic data file from a trained spec",
)
async def generate_file(
    req: GenerateRequest,
    svc: SpecGenAIService = Depends(get_service),
):
    """
    Generate synthetic test data conforming to the learned specification.

    **output_format** controls the response:
    - **file**    – Saves to disk, returns `output_path` (use `/download/{generation_id}` to fetch)
    - **preview** – Returns the first 20 lines in `preview_lines`
    - **json**    – Returns structured records in `payload`

    **Example body**:
    ```json
    {
      "spec_id": "your-spec-uuid",
      "record_count": 50,
      "seed": 42,
      "output_format": "file"
    }
    ```
    """
    try:
        return await svc.generate_file(req)
    except Exception as exc:
        raise _http(exc)


# ── Download ──────────────────────────────────────────────────────────────────

@router.get(
    "/download/{generation_id}",
    tags=["Generation"],
    summary="Download a previously generated file",
)
async def download_output(
    generation_id: str,
    svc: SpecGenAIService = Depends(get_service),
):
    """
    Download the raw generated file for a given **generation_id**.
    The Content-Disposition header is set for browser download.
    """
    try:
        path, media_type = await svc.get_output_content(generation_id)
        return FileResponse(
            path=str(path),
            media_type=media_type,
            filename=path.name,
            headers={"Content-Disposition": f"attachment; filename={path.name}"},
        )
    except Exception as exc:
        raise _http(exc)
