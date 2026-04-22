"""
JSON Schema / Custom Spec Parser
==================================
Accepts three kinds of JSON input:

1. **SpecGenAI native format** – the JSON structure we define for users
   to describe their custom file format (see sample_spec.json).

2. **JSON Schema draft-07 / draft-2020** – standard $schema documents.
   The parser extracts properties, types, required arrays, patterns, etc.

3. **Flat sample JSON array** – an array of objects from which we infer
   field names, types and constraints statistically (handled by the
   SampleDataParser; this parser delegates to it if appropriate).
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.models.schema import (
    SchemaModel, SpecType, RecordDef, FieldDef, FieldType,
    FieldConstraints, FieldDependency, FileStructure,
    RecordCategory, Justification,
)
from app.parsers.base_parser import BaseParser
from app.core.exceptions import ParseError


# Mapping from JSON Schema type strings to our FieldType
_JSONSCHEMA_TYPE_MAP: dict[str, FieldType] = {
    "string":  FieldType.STRING,
    "integer": FieldType.NUMERIC,
    "number":  FieldType.AMOUNT,
    "boolean": FieldType.BOOLEAN,
    "array":   FieldType.STRING,    # treat nested arrays as string for now
    "object":  FieldType.STRING,
    "null":    FieldType.STRING,
}

# SpecGenAI custom type names → FieldType
_CUSTOM_TYPE_MAP: dict[str, FieldType] = {
    "string":         FieldType.STRING,
    "numeric":        FieldType.NUMERIC,
    "alphanumeric":   FieldType.ALPHANUMERIC,
    "date":           FieldType.DATE,
    "datetime":       FieldType.DATETIME,
    "boolean":        FieldType.BOOLEAN,
    "amount":         FieldType.AMOUNT,
    "routing_number": FieldType.ROUTING_NUMBER,
    "account_number": FieldType.ACCOUNT_NUMBER,
    "pan":            FieldType.PAN,
    "cvv":            FieldType.CVV,
    "expiry":         FieldType.EXPIRY,
    "checksum":       FieldType.CHECKSUM,
    "enum":           FieldType.ENUM,
    "sequence":       FieldType.SEQUENCE,
    "constant":       FieldType.CONSTANT,
    "computed":       FieldType.COMPUTED,
}


class JSONSchemaParser(BaseParser):
    """Parses JSON-based specification documents."""

    spec_type = SpecType.JSON

    def _do_parse(self, content: str, source_name: str) -> SchemaModel:
        content = content.strip()
        if not content:
            raise ParseError("JSON content is empty")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON: {exc}") from exc

        if isinstance(raw, list):
            # Array of sample objects — delegate message
            self._warn("Received array input; treating as sample data. "
                       "Consider using spec_type=sample for richer inference.")
            return self._parse_sample_array(raw, source_name)

        if not isinstance(raw, dict):
            raise ParseError("JSON spec must be a JSON object or array")

        # Detect format
        if raw.get("$schema") or ("properties" in raw and "type" in raw):
            return self._parse_json_schema_draft(raw, source_name)
        elif "records" in raw or "spec_type" in raw:
            return self._parse_native_spec(raw, source_name)
        else:
            # Unknown structure — try as a single-record flat object
            return self._parse_flat_object(raw, source_name)

    # ── SpecGenAI native format ──────────────────────────────────────────────

    def _parse_native_spec(self, raw: dict, source_name: str) -> SchemaModel:
        """
        Example native spec format (see sample_spec.json):
        {
          "spec_name": "MyPaymentFile",
          "spec_type": "json",
          "description": "...",
          "file_structure": { "format": "delimited", "delimiter": "|", ... },
          "records": [
            {
              "record_type_id": "HDR",
              "name": "Header",
              "category": "header",
              "ordering": 0,
              "fields": [
                {
                  "name": "record_id",
                  "type": "constant",
                  "length": 3,
                  "default_value": "HDR",
                  "required": true
                }, ...
              ]
            }
          ],
          "global_rules": ["sum_check == total"]
        }
        """
        # File structure
        fs_raw = raw.get("file_structure", {})
        fs = FileStructure(
            format=fs_raw.get("format", "delimited"),
            line_ending=fs_raw.get("line_ending", "\n"),
            encoding=fs_raw.get("encoding", "utf-8"),
            file_extension=fs_raw.get("file_extension", ".txt"),
            header_records=fs_raw.get("header_records", []),
            detail_records=fs_raw.get("detail_records", []),
            trailer_records=fs_raw.get("trailer_records", []),
            batch_supported=fs_raw.get("batch_supported", False),
            batch_header_record=fs_raw.get("batch_header_record"),
            batch_trailer_record=fs_raw.get("batch_trailer_record"),
        )

        records: list[RecordDef] = []
        for i, rec_raw in enumerate(raw.get("records", [])):
            rec = RecordDef(
                record_type_id=str(rec_raw.get("record_type_id", f"REC{i}")),
                name=rec_raw.get("name", f"Record {i}"),
                category=RecordCategory(rec_raw.get("category", "unknown")),
                fixed_width=rec_raw.get("fixed_width", False),
                record_length=rec_raw.get("record_length"),
                delimiter=rec_raw.get("delimiter") or fs_raw.get("delimiter"),
                repeatable=rec_raw.get("repeatable", False),
                min_occurrences=rec_raw.get("min_occurrences", 0),
                max_occurrences=rec_raw.get("max_occurrences"),
                ordering=rec_raw.get("ordering", i * 10),
            )
            for j, f_raw in enumerate(rec_raw.get("fields", [])):
                rec.fields.append(self._parse_native_field(f_raw, j))
            records.append(rec)

        return SchemaModel(
            spec_name=raw.get("spec_name", source_name),
            spec_type=SpecType.JSON,
            version=raw.get("version", "1.0"),
            description=raw.get("description", ""),
            source_files=[source_name] if source_name else [],
            file_structure=fs,
            records=records,
            global_rules=raw.get("global_rules", []),
        )

    def _parse_native_field(self, f_raw: dict, idx: int) -> FieldDef:
        name = self._normalise_name(f_raw.get("name", f"field_{idx}"))
        raw_type = f_raw.get("type", "string").lower()
        ftype = _CUSTOM_TYPE_MAP.get(raw_type, FieldType.STRING)

        constraints = FieldConstraints(
            min_length=f_raw.get("min_length"),
            max_length=f_raw.get("max_length") or f_raw.get("length"),
            min_value=f_raw.get("min_value"),
            max_value=f_raw.get("max_value"),
            pattern=f_raw.get("pattern"),
            allowed_values=f_raw.get("allowed_values") or f_raw.get("enum", []),
            required=f_raw.get("required", True),
            nullable=f_raw.get("nullable", False),
        )

        deps: list[FieldDependency] = []
        for dep_raw in f_raw.get("dependencies", []):
            deps.append(FieldDependency(
                depends_on=dep_raw["depends_on"],
                dependency_type=dep_raw.get("type", "conditional"),
                condition=dep_raw.get("condition"),
                formula=dep_raw.get("formula"),
            ))

        return FieldDef(
            name=name,
            field_type=ftype,
            start_pos=f_raw.get("start_pos"),
            end_pos=f_raw.get("end_pos"),
            length=f_raw.get("length"),
            justification=Justification(f_raw.get("justification", "left")),
            pad_char=f_raw.get("pad_char", " "),
            format_string=f_raw.get("format_string"),
            default_value=f_raw.get("default_value"),
            description=f_raw.get("description", ""),
            constraints=constraints,
            dependencies=deps,
        )

    # ── JSON Schema draft mode ───────────────────────────────────────────────

    def _parse_json_schema_draft(self, raw: dict, source_name: str) -> SchemaModel:
        title = raw.get("title", source_name)
        description = raw.get("description", "")
        properties = raw.get("properties", {})
        required_fields = set(raw.get("required", []))

        rec = RecordDef(
            record_type_id="RECORD",
            name=title,
            category=RecordCategory.DETAIL,
            repeatable=True,
            ordering=10,
        )

        for prop_name, prop_schema in properties.items():
            fd = self._prop_to_field(prop_name, prop_schema, prop_name in required_fields)
            rec.fields.append(fd)

        fs = FileStructure(
            format="json",
            file_extension=".json",
            detail_records=["RECORD"],
        )

        return SchemaModel(
            spec_name=title,
            spec_type=SpecType.JSON,
            version=raw.get("$schema", "draft-07"),
            description=description,
            source_files=[source_name] if source_name else [],
            file_structure=fs,
            records=[rec],
        )

    def _prop_to_field(self, name: str, schema: dict, required: bool) -> FieldDef:
        raw_type = schema.get("type", "string")
        if isinstance(raw_type, list):
            raw_type = next((t for t in raw_type if t != "null"), "string")
        ftype = _JSONSCHEMA_TYPE_MAP.get(raw_type, FieldType.STRING)

        # Smarter type inference from format / pattern hints
        fmt = schema.get("format", "")
        if fmt in ("date", "date-time"):
            ftype = FieldType.DATE if fmt == "date" else FieldType.DATETIME
        if schema.get("pattern"):
            if re.search(r"\\d{13,19}", schema["pattern"]):
                ftype = FieldType.PAN

        constraints = FieldConstraints(
            min_length=schema.get("minLength"),
            max_length=schema.get("maxLength"),
            min_value=schema.get("minimum") or schema.get("exclusiveMinimum"),
            max_value=schema.get("maximum") or schema.get("exclusiveMaximum"),
            pattern=schema.get("pattern"),
            allowed_values=schema.get("enum") or [],
            required=required,
            nullable="null" in (schema.get("type") if isinstance(schema.get("type"), list) else []),
        )

        if constraints.allowed_values:
            ftype = FieldType.ENUM

        return FieldDef(
            name=self._normalise_name(name),
            field_type=ftype,
            description=schema.get("description", ""),
            constraints=constraints,
            format_string=fmt or None,
        )

    # ── Flat object / sample array ───────────────────────────────────────────

    def _parse_flat_object(self, raw: dict, source_name: str) -> SchemaModel:
        return self._parse_sample_array([raw], source_name)

    def _parse_sample_array(self, rows: list[dict], source_name: str) -> SchemaModel:
        """Build a minimal schema by inspecting the key set of the first row."""
        if not rows:
            raise ParseError("Empty sample array")
        keys = list(rows[0].keys())
        rec = RecordDef(
            record_type_id="RECORD",
            name="Inferred Record",
            category=RecordCategory.DETAIL,
            repeatable=True,
            ordering=10,
        )
        for key in keys:
            values = [str(row.get(key, "")) for row in rows[:50]]
            ftype = self._infer_type_from_values(values)
            rec.fields.append(FieldDef(
                name=self._normalise_name(key),
                field_type=ftype,
                description=f"Inferred from sample data key: {key}",
                inferred_confidence=0.6,
                sample_values=values[:10],
            ))

        return SchemaModel(
            spec_name=source_name,
            spec_type=SpecType.JSON,
            description="Inferred from sample JSON array",
            source_files=[source_name] if source_name else [],
            file_structure=FileStructure(format="json", file_extension=".json",
                                         detail_records=["RECORD"]),
            records=[rec],
        )

    @staticmethod
    def _infer_type_from_values(values: list[str]) -> FieldType:
        non_empty = [v for v in values if v]
        if not non_empty:
            return FieldType.STRING
        if all(re.fullmatch(r"-?\d+", v) for v in non_empty):
            return FieldType.NUMERIC
        if all(re.fullmatch(r"-?\d+(\.\d+)?", v) for v in non_empty):
            return FieldType.AMOUNT
        if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in non_empty):
            return FieldType.DATE
        if all(v.lower() in ("true", "false", "0", "1", "yes", "no") for v in non_empty):
            return FieldType.BOOLEAN
        unique_vals = set(non_empty)
        unique_ratio = len(unique_vals) / len(non_empty)
        # Use absolute cardinality <= 10 OR low ratio for larger samples
        if len(unique_vals) <= 10 or (unique_ratio < 0.4 and len(unique_vals) <= 20):
            return FieldType.ENUM
        return FieldType.STRING
