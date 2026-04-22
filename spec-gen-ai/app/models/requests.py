"""
Pydantic models for all REST API request and response payloads.
Kept separate from the internal SchemaModel domain objects.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, field_validator


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    spec_type: str
    file_size_bytes: int
    message: str


# ── Training ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    upload_ids: list[str] = Field(..., min_length=1)
    spec_name: str = Field(..., min_length=1, max_length=128)
    spec_type: str = Field(..., pattern="^(vcf|ach|json|sample)$")
    description: str = ""
    override_existing: bool = False

    @field_validator("spec_name")
    @classmethod
    def spec_name_must_be_alphanumeric(cls, v: str) -> str:
        if not all(c.isalnum() or c in (" ", "_", "-") for c in v):
            raise ValueError("spec_name may only contain letters, digits, spaces, _ and -")
        return v.strip()


class TrainResponse(BaseModel):
    spec_id: str
    spec_name: str
    spec_type: str
    records_learned: int
    fields_learned: int
    inference_stats: dict[str, Any]
    trained_at: str
    message: str


# ── Specification listing ──────────────────────────────────────────────────────

class SpecSummary(BaseModel):
    spec_id: str
    spec_name: str
    spec_type: str
    version: str
    description: str
    records_count: int
    is_trained: bool
    created_at: str
    trained_at: str | None


class ListSpecsResponse(BaseModel):
    total: int
    specs: list[SpecSummary]


class GetSpecResponse(BaseModel):
    spec_id: str
    spec_name: str
    spec_type: str
    version: str
    description: str
    file_structure: dict[str, Any]
    records: list[dict[str, Any]]
    global_rules: list[str]
    inference_stats: dict[str, Any]
    is_trained: bool
    created_at: str
    trained_at: str | None


# ── Generation ────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    spec_id: str
    record_count: int = Field(default=10, ge=1, le=10000)
    seed: int | None = None           # For reproducible generation
    overrides: dict[str, Any] = Field(default_factory=dict)
    output_format: str = Field(default="file", pattern="^(file|json|preview)$")


class GenerateResponse(BaseModel):
    generation_id: str
    spec_id: str
    spec_name: str
    record_count: int
    output_path: str | None = None
    preview_lines: list[str] | None = None
    payload: list[dict[str, Any]] | None = None
    validation_passed: bool
    validation_errors: list[str] = Field(default_factory=list)
    generated_at: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
