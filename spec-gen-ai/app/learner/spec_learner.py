"""
Specification Learner / Trainer
================================
Orchestrates the full training pipeline:

  1. Parse uploaded files → raw SchemaModels
  2. Merge multiple SchemaModels (spec + sample data) into one
  3. Enrich fields using FieldInferrer (statistical ML)
  4. Detect structural patterns using PatternDetector
  5. Build dependency graph and infer constraints
  6. Finalise the SchemaModel and mark it as trained

This is the single entry point for the training service.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.schema import (
    SchemaModel, SpecType, RecordDef, FieldDef, FieldType,
    FieldConstraints, FieldDependency,
)
from app.parsers import get_parser
from app.learner.field_inferrer import FieldInferrer
from app.learner.pattern_detector import PatternDetector
from app.core.exceptions import TrainingError, InsufficientSampleDataError
from app.core.config import settings

logger = logging.getLogger(__name__)


class SpecLearner:
    """
    High-level trainer that produces a fully-trained SchemaModel.

    Usage:
        learner = SpecLearner()
        model = learner.train(
            spec_files=[(Path("myspec.json"), "json")],
            sample_files=[Path("sample_data.txt")],
            spec_name="MyPaymentSpec",
        )
    """

    def __init__(self):
        self._field_inferrer = FieldInferrer()
        self._pattern_detector = PatternDetector()

    # ── Main entry point ──────────────────────────────────────────────────────

    def train(
        self,
        spec_files: list[tuple[Path, str]],  # (path, spec_type)
        sample_files: list[Path],
        spec_name: str,
        description: str = "",
    ) -> SchemaModel:
        """
        Full training pipeline.  Returns a trained SchemaModel.
        """
        logger.info("Starting training for spec '%s'", spec_name)

        if not spec_files:
            raise TrainingError("At least one specification file must be provided.")

        # ── Step 1: Parse all spec files ──────────────────────────────────────
        spec_models: list[SchemaModel] = []
        for path, stype in spec_files:
            logger.info("  Parsing spec file: %s (%s)", path.name, stype)
            parser = get_parser(stype)
            try:
                model = parser.parse_file(path)
                spec_models.append(model)
            except Exception as exc:
                raise TrainingError(f"Failed to parse spec file '{path.name}': {exc}") from exc

        # ── Step 2: Parse sample data files ───────────────────────────────────
        sample_models: list[SchemaModel] = []
        for path in sample_files:
            logger.info("  Parsing sample file: %s", path.name)
            parser = get_parser("sample")
            try:
                model = parser.parse_file(path)
                sample_models.append(model)
            except (InsufficientSampleDataError, Exception) as exc:
                logger.warning("  Sample file '%s' skipped: %s", path.name, exc)

        # ── Step 3: Merge all models ──────────────────────────────────────────
        primary = spec_models[0]
        for additional in spec_models[1:]:
            primary = self._merge_models(primary, additional)
        for sample_model in sample_models:
            primary = self._enrich_from_sample(primary, sample_model)

        # ── Step 4: Enrich fields with statistical inference ──────────────────
        total_fields = 0
        enriched_fields = 0
        for rec in primary.records:
            for field in rec.fields:
                prev_conf = field.inferred_confidence
                self._field_inferrer.enrich(field)
                total_fields += 1
                if field.inferred_confidence > prev_conf:
                    enriched_fields += 1

        # ── Step 5: Detect structural patterns ───────────────────────────────
        pattern_result = None
        if sample_files:
            first_sample_content = sample_files[0].read_text(encoding="utf-8", errors="replace")
            try:
                pattern_result = self._pattern_detector.detect_from_file(
                    first_sample_content, primary
                )
                logger.info("  Pattern detection: ordering=%s", pattern_result.ordering_sequence)
            except Exception as exc:
                logger.warning("  Pattern detection failed: %s", exc)

        # ── Step 6: Build dependency graph ────────────────────────────────────
        self._build_dependencies(primary)

        # ── Step 7: Infer constant values ─────────────────────────────────────
        self._infer_constants(primary)

        # ── Step 8: Finalise ─────────────────────────────────────────────────
        primary.spec_name = spec_name
        primary.description = description or primary.description
        primary.trained_at = datetime.utcnow().isoformat()
        primary.is_trained = True
        primary.inference_stats.update(
            {
                "total_fields": total_fields,
                "enriched_fields": enriched_fields,
                "total_records": len(primary.records),
                "sample_files_used": len(sample_models),
                "pattern_ordering": pattern_result.ordering_sequence if pattern_result else [],
                "anomalies_found": pattern_result.anomalies if pattern_result else [],
                "training_completed_at": primary.trained_at,
            }
        )

        logger.info(
            "Training complete: %d records, %d fields, %d enriched",
            len(primary.records), total_fields, enriched_fields,
        )
        return primary

    # ── Merge logic ───────────────────────────────────────────────────────────

    def _merge_models(self, base: SchemaModel, other: SchemaModel) -> SchemaModel:
        """Merge 'other' into 'base', preferring base on conflict."""
        existing_ids = {r.record_type_id for r in base.records}
        for rec in other.records:
            if rec.record_type_id not in existing_ids:
                base.records.append(rec)
        base.source_files.extend(other.source_files)
        base.global_rules = list(set(base.global_rules + other.global_rules))
        return base

    def _enrich_from_sample(self, base: SchemaModel, sample: SchemaModel) -> SchemaModel:
        """
        Use field sample_values from a sample-data model to enrich
        fields in the base model by name matching.
        """
        # Build a flat name → values map from the sample model
        sample_field_values: dict[str, list[str]] = {}
        for rec in sample.records:
            for field in rec.fields:
                if field.sample_values:
                    sample_field_values[field.name] = field.sample_values

        # Enrich matching fields in base
        for rec in base.records:
            for field in rec.fields:
                if field.name in sample_field_values and not field.sample_values:
                    field.sample_values = sample_field_values[field.name]

        base.source_files.extend(sample.source_files)
        return base

    # ── Dependency inference ──────────────────────────────────────────────────

    def _build_dependencies(self, model: SchemaModel):
        """
        Infer common dependencies:
        - Checksum fields depend on data fields
        - Trailer count fields depend on detail record count
        - batch_number in entries matches batch header batch_number
        """
        for rec in model.records:
            field_map = {f.name: f for f in rec.fields}

            for field in rec.fields:
                # Checksum field depends on all numeric fields in the record
                if field.field_type == FieldType.CHECKSUM and not field.dependencies:
                    numeric_fields = [
                        f.name for f in rec.fields
                        if f.field_type in (FieldType.AMOUNT, FieldType.NUMERIC, FieldType.ROUTING_NUMBER)
                        and f.name != field.name
                    ]
                    for dep_name in numeric_fields[:3]:  # top 3
                        field.dependencies.append(FieldDependency(
                            depends_on=dep_name,
                            dependency_type="computed",
                            formula=f"checksum_of({dep_name})",
                        ))

                # Sequence fields that have "trace" or "number" in name
                if field.field_type == FieldType.SEQUENCE and not field.dependencies:
                    if "trace" in field.name or "batch_number" in field.name:
                        field.dependencies.append(FieldDependency(
                            depends_on="__record_index__",
                            dependency_type="computed",
                            formula="auto_increment(start=1)",
                        ))

    # ── Constant inference ────────────────────────────────────────────────────

    def _infer_constants(self, model: SchemaModel):
        """Set default_value for CONSTANT fields if not already set."""
        constant_values: dict[str, str] = {
            "record_type_code": "",  # set per record
            "record_size": "094",
            "blocking_factor": "10",
            "format_code": "1",
            "filler": " " * 20,
        }
        for rec in model.records:
            for field in rec.fields:
                if field.field_type == FieldType.CONSTANT and field.default_value is None:
                    if field.name in constant_values:
                        field.default_value = constant_values[field.name]
                    elif field.name == "record_type_code":
                        field.default_value = rec.record_type_id
