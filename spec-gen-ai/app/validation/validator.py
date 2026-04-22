"""
Validation Engine
=================
Validates a generated or uploaded data file against a SchemaModel.
Produces structured validation reports with field-level error detail.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

from app.models.schema import (
    SchemaModel, RecordDef, FieldDef, FieldType, FieldConstraints,
)
from app.rule_engine.rule_engine import RuleEngine, _luhn_check, _aba_check

logger = logging.getLogger(__name__)


@dataclass
class FieldError:
    record_type: str
    record_index: int
    field_name: str
    value: str
    error: str


@dataclass
class ValidationReport:
    passed: bool
    total_records: int
    records_with_errors: int
    total_errors: int
    field_errors: list[FieldError] = field(default_factory=list)
    global_errors: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_records": self.total_records,
            "records_with_errors": self.records_with_errors,
            "total_errors": self.total_errors,
            "field_errors": [
                {
                    "record_type": e.record_type,
                    "record_index": e.record_index,
                    "field_name": e.field_name,
                    "value": e.value[:50],
                    "error": e.error,
                }
                for e in self.field_errors[:100]  # cap at 100 in report
            ],
            "global_errors": self.global_errors,
            "summary": self.summary,
        }


class ValidationEngine:
    """
    Validates a list of record dicts against the rules derived from a SchemaModel.
    """

    MAX_ERRORS_PER_FIELD = 5  # Stop reporting after this many errors for one field

    def __init__(self, model: SchemaModel):
        self.model = model
        self._rule_engine = RuleEngine()
        self._rule_engine.load_from_model(model)
        self._rec_map: dict[str, RecordDef] = {
            r.record_type_id: r for r in model.records
        }

    # ── Main validate entry ────────────────────────────────────────────────────

    def validate(self, records: list[dict[str, Any]]) -> ValidationReport:
        field_errors: list[FieldError] = []
        error_counts: dict[str, int] = {}  # field_key → count
        records_with_errors: set[int] = set()

        for idx, row in enumerate(records):
            rt_id = row.get("_record_type", "UNKNOWN")
            rec_def = self._rec_map.get(rt_id)
            if not rec_def:
                continue

            # Field-level validation
            for fd in rec_def.fields:
                value = row.get(fd.name, "")
                field_key = f"{rt_id}.{fd.name}"
                if error_counts.get(field_key, 0) >= self.MAX_ERRORS_PER_FIELD:
                    continue

                errs = self._validate_field(rt_id, fd, value, row)
                for err in errs:
                    field_errors.append(FieldError(rt_id, idx, fd.name, str(value)[:40], err))
                    error_counts[field_key] = error_counts.get(field_key, 0) + 1
                    records_with_errors.add(idx)

            # Record-level validation
            rec_errs = self._rule_engine.validate_record(rt_id, row)
            for err in rec_errs:
                field_errors.append(FieldError(rt_id, idx, "_record_", "", err))
                records_with_errors.add(idx)

        # File-level validation
        global_errors = self._rule_engine.validate_file(records)

        total_errors = len(field_errors) + len(global_errors)
        passed = total_errors == 0

        summary = (
            f"Validation {'PASSED' if passed else 'FAILED'}. "
            f"Records: {len(records)}, "
            f"Errors: {total_errors}, "
            f"Affected records: {len(records_with_errors)}"
        )

        return ValidationReport(
            passed=passed,
            total_records=len(records),
            records_with_errors=len(records_with_errors),
            total_errors=total_errors,
            field_errors=field_errors,
            global_errors=global_errors,
            summary=summary,
        )

    # ── Field validation ──────────────────────────────────────────────────────

    def _validate_field(
        self,
        rt_id: str,
        fd: FieldDef,
        value: Any,
        row: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        c = fd.constraints
        val_str = str(value) if value is not None else ""

        # Required check
        if c.required and not val_str.strip():
            errors.append(f"Field '{fd.name}' is required but empty.")
            return errors  # No point checking further

        # Skip further validation for empty optional fields
        if not val_str.strip() and not c.required:
            return errors

        # Length checks
        if c.min_length is not None and len(val_str) < c.min_length:
            errors.append(
                f"'{fd.name}' length {len(val_str)} < min_length {c.min_length}"
            )
        if c.max_length is not None and len(val_str) > c.max_length:
            errors.append(
                f"'{fd.name}' length {len(val_str)} > max_length {c.max_length}"
            )

        # Allowed values
        if c.allowed_values and val_str not in c.allowed_values:
            errors.append(
                f"'{fd.name}' value '{val_str}' not in {c.allowed_values[:5]}…"
            )

        # Pattern
        if c.pattern:
            try:
                if not re.fullmatch(c.pattern, val_str):
                    errors.append(
                        f"'{fd.name}' value '{val_str[:20]}' does not match pattern {c.pattern}"
                    )
            except re.error:
                pass  # ignore bad patterns

        # Numeric range
        if fd.field_type in (FieldType.NUMERIC, FieldType.AMOUNT):
            try:
                num = float(val_str.replace(",", ""))
                if c.min_value is not None and num < c.min_value:
                    errors.append(f"'{fd.name}' value {num} < min {c.min_value}")
                if c.max_value is not None and num > c.max_value:
                    errors.append(f"'{fd.name}' value {num} > max {c.max_value}")
            except ValueError:
                errors.append(f"'{fd.name}' expected numeric, got '{val_str[:20]}'")

        # Checksum algorithms
        if c.checksum_algorithm == "luhn":
            if not _luhn_check(val_str.strip()):
                errors.append(f"'{fd.name}' PAN fails Luhn checksum: '{val_str[:20]}'")

        if c.checksum_algorithm in ("aba", "mod10") and fd.field_type == FieldType.ROUTING_NUMBER:
            if not _aba_check(val_str.strip()):
                errors.append(
                    f"'{fd.name}' routing number fails ABA check: '{val_str[:20]}'"
                )

        return errors
