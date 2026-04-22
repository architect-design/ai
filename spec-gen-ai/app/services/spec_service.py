"""
SpecGenAI Service Layer
========================
High-level orchestration of all use cases:
  - upload_file        → store raw upload, return upload_id
  - train_spec         → parse + learn → save SchemaModel
  - list_specs         → enumerate all trained schemas
  - get_spec           → fetch single schema
  - generate_file      → produce synthetic data file
  - download_output    → stream a previously generated file

All methods raise typed exceptions from app.core.exceptions.
Controllers translate those to HTTP status codes.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.schema import SchemaModel, SpecType
from app.models.requests import (
    TrainRequest, TrainResponse,
    SpecSummary, ListSpecsResponse, GetSpecResponse,
    GenerateRequest, GenerateResponse,
    UploadResponse,
)
from app.parsers import get_parser
from app.learner.spec_learner import SpecLearner
from app.generator.base_generator import get_generator, GenerationResult
from app.validation.validator import ValidationEngine
from app.storage.storage_service import StorageService
from app.core.config import settings
from app.core.exceptions import (
    SpecNotFoundError, TrainingError, GenerationError, ValidationError
)

logger = logging.getLogger(__name__)


class SpecGenAIService:
    """
    Stateless service — all state lives in the storage layer.
    Instantiate once and reuse (e.g., as a FastAPI dependency).
    """

    def __init__(self):
        self._storage = StorageService()
        self._learner = SpecLearner()

    # ── Upload ────────────────────────────────────────────────────────────────

    async def upload_file(
        self, filename: str, content: bytes, spec_type: str
    ) -> UploadResponse:
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise ValidationError(
                f"File too large: {len(content)} bytes. Max: {max_bytes} bytes."
            )

        if spec_type not in settings.ALLOWED_SPEC_TYPES:
            raise ValidationError(
                f"Unsupported spec_type '{spec_type}'. "
                f"Allowed: {settings.ALLOWED_SPEC_TYPES}"
            )

        upload_id, path = self._storage.save_upload(filename, content)
        logger.info("Uploaded file '%s' as upload_id=%s", filename, upload_id)

        return UploadResponse(
            upload_id=upload_id,
            filename=filename,
            spec_type=spec_type,
            file_size_bytes=len(content),
            message=f"File uploaded successfully. Use upload_id='{upload_id}' in train request.",
        )

    # ── Training ──────────────────────────────────────────────────────────────

    async def train_spec(self, req: TrainRequest) -> TrainResponse:
        # Check for name collision
        existing = self._storage.find_spec_by_name(req.spec_name)
        if existing and not req.override_existing:
            raise TrainingError(
                f"A specification named '{req.spec_name}' already exists "
                f"(id={existing.spec_id}). Set override_existing=true to replace it."
            )

        # Resolve upload paths, separate spec files from sample files
        spec_files: list[tuple[Path, str]] = []
        sample_files: list[Path] = []

        for uid in req.upload_ids:
            path = self._storage.get_upload_path(uid)
            # Classify: if spec_type == sample or file ext suggests data
            if req.spec_type == "sample":
                sample_files.append(path)
            elif path.suffix.lower() in (".csv", ".tsv", ".dat"):
                sample_files.append(path)
            else:
                spec_files.append((path, req.spec_type))

        # If everything was classified as sample, promote first to spec
        if not spec_files and sample_files:
            spec_files.append((sample_files.pop(0), "sample"))

        # Train
        try:
            model = self._learner.train(
                spec_files=spec_files,
                sample_files=sample_files,
                spec_name=req.spec_name,
                description=req.description,
            )
        except Exception as exc:
            raise TrainingError(f"Training failed: {exc}") from exc

        # Delete old spec if overriding
        if existing and req.override_existing:
            self._storage.delete_spec(existing.spec_id)

        # Persist
        self._storage.save_spec(model)

        total_fields = sum(len(r.fields) for r in model.records)
        return TrainResponse(
            spec_id=model.spec_id,
            spec_name=model.spec_name,
            spec_type=model.spec_type.value,
            records_learned=len(model.records),
            fields_learned=total_fields,
            inference_stats=model.inference_stats,
            trained_at=model.trained_at or datetime.utcnow().isoformat(),
            message=(
                f"Training complete. Learned {len(model.records)} record types "
                f"with {total_fields} fields."
            ),
        )

    # ── Spec listing / retrieval ───────────────────────────────────────────────

    async def list_specs(self) -> ListSpecsResponse:
        models = self._storage.list_specs()
        summaries = [
            SpecSummary(
                spec_id=m.spec_id,
                spec_name=m.spec_name,
                spec_type=m.spec_type.value,
                version=m.version,
                description=m.description,
                records_count=len(m.records),
                is_trained=m.is_trained,
                created_at=m.created_at,
                trained_at=m.trained_at,
            )
            for m in models
        ]
        return ListSpecsResponse(total=len(summaries), specs=summaries)

    async def get_spec(self, spec_id: str) -> GetSpecResponse:
        model = self._storage.load_spec(spec_id)
        return GetSpecResponse(
            spec_id=model.spec_id,
            spec_name=model.spec_name,
            spec_type=model.spec_type.value,
            version=model.version,
            description=model.description,
            file_structure=model.file_structure.to_dict(),
            records=[r.to_dict() for r in model.records],
            global_rules=model.global_rules,
            inference_stats=model.inference_stats,
            is_trained=model.is_trained,
            created_at=model.created_at,
            trained_at=model.trained_at,
        )

    async def delete_spec(self, spec_id: str):
        if not self._storage.spec_exists(spec_id):
            raise SpecNotFoundError(f"Specification '{spec_id}' not found")
        self._storage.delete_spec(spec_id)

    # ── Generation ────────────────────────────────────────────────────────────

    async def generate_file(self, req: GenerateRequest) -> GenerateResponse:
        # Load schema
        model = self._storage.load_spec(req.spec_id)

        if not model.is_trained:
            raise GenerationError(
                f"Specification '{req.spec_id}' has not been trained yet. "
                "Please call /train first."
            )

        # Generate
        generator = get_generator(model, seed=req.seed)
        try:
            result: GenerationResult = generator.generate(
                record_count=req.record_count,
                overrides=req.overrides,
            )
        except Exception as exc:
            raise GenerationError(f"Generation error: {exc}") from exc

        # Validate
        validator = ValidationEngine(model)
        report = validator.validate(result.records)
        all_errors = result.validation_errors + report.global_errors

        generation_id = str(uuid.uuid4())
        output_path: str | None = None
        preview_lines: list[str] | None = None
        payload: list[dict] | None = None

        if req.output_format == "file":
            path = self._storage.save_output(
                generation_id,
                result.content,
                extension=result.file_extension,
                encoding=model.file_structure.encoding,
            )
            output_path = str(path)

        elif req.output_format == "preview":
            lines = result.content.splitlines()
            preview_lines = lines[:20]

        elif req.output_format == "json":
            # Return clean record dicts (no internal keys)
            payload = [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in result.records
                if row.get("_record_type") not in (
                    model.file_structure.header_records +
                    model.file_structure.trailer_records
                )
            ]

        return GenerateResponse(
            generation_id=generation_id,
            spec_id=req.spec_id,
            spec_name=model.spec_name,
            record_count=result.record_count,
            output_path=output_path,
            preview_lines=preview_lines,
            payload=payload,
            validation_passed=report.passed,
            validation_errors=all_errors[:20],
            generated_at=datetime.utcnow().isoformat(),
        )

    async def get_output_content(self, generation_id: str) -> tuple[Path, str]:
        """Returns (path, media_type) for a generated output file."""
        path = self._storage.get_output_path(generation_id)
        ext = path.suffix.lower()
        media_type = {
            ".json": "application/json",
            ".ach":  "text/plain",
            ".vcf":  "text/plain",
            ".txt":  "text/plain",
            ".csv":  "text/csv",
        }.get(ext, "application/octet-stream")
        return path, media_type

    async def get_storage_stats(self) -> dict:
        return self._storage.storage_stats()
