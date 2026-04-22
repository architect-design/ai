"""
Internal Schema Model
=====================
These dataclasses are the lingua franca of the entire system.
Parsers produce SchemaModel instances; generators consume them;
the rule engine annotates them with dependency and constraint info.

Design principle: every attribute has a sensible default so parsers
only need to fill in what they know — the learner fills in the rest.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


# ── Enumerations ─────────────────────────────────────────────────────────────

class SpecType(str, Enum):
    VCF = "vcf"         # VISA Card File
    ACH = "ach"         # ACH / NACHA
    JSON = "json"       # JSON Schema / custom spec
    SAMPLE = "sample"   # Learned purely from sample data


class FieldType(str, Enum):
    STRING = "string"
    NUMERIC = "numeric"
    ALPHANUMERIC = "alphanumeric"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    AMOUNT = "amount"           # Currency / decimal amount
    ROUTING_NUMBER = "routing_number"
    ACCOUNT_NUMBER = "account_number"
    PAN = "pan"                 # Primary Account Number (card)
    CVV = "cvv"
    EXPIRY = "expiry"
    CHECKSUM = "checksum"
    ENUM = "enum"               # Fixed set of allowed values
    SEQUENCE = "sequence"       # Auto-incrementing counter
    CONSTANT = "constant"       # Always the same value
    COMPUTED = "computed"       # Derived from other fields


class Justification(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class RecordCategory(str, Enum):
    HEADER = "header"
    DETAIL = "detail"
    TRAILER = "trailer"
    ADDENDA = "addenda"
    BATCH_HEADER = "batch_header"
    BATCH_TRAILER = "batch_trailer"
    UNKNOWN = "unknown"


# ── Field-level structures ────────────────────────────────────────────────────

@dataclass
class FieldConstraints:
    """All validation constraints for a single field."""
    min_length: int | None = None
    max_length: int | None = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None          # Regex pattern
    allowed_values: list[str] = field(default_factory=list)
    checksum_algorithm: str | None = None   # e.g. "luhn", "mod10"
    required: bool = True
    nullable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FieldConstraints:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FieldDependency:
    """Describes how one field value depends on another."""
    depends_on: str               # Field name this field depends on
    dependency_type: str          # "conditional", "computed", "ordered"
    condition: str | None = None  # Python expression: "depends_on == 'X'"
    formula: str | None = None    # Python expression for computed fields

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FieldDef:
    """Complete definition of one field within a record."""
    name: str
    field_type: FieldType = FieldType.STRING
    start_pos: int | None = None      # For fixed-width formats
    end_pos: int | None = None
    length: int | None = None
    justification: Justification = Justification.LEFT
    pad_char: str = " "
    format_string: str | None = None  # e.g. "YYYYMMDD" for dates
    default_value: Any = None
    description: str = ""
    constraints: FieldConstraints = field(default_factory=FieldConstraints)
    dependencies: list[FieldDependency] = field(default_factory=list)
    # Inferred statistics from sample data
    inferred_confidence: float = 0.0  # 0.0 – 1.0
    sample_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["field_type"] = self.field_type.value
        d["justification"] = self.justification.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FieldDef:
        d = dict(d)
        d["field_type"] = FieldType(d.get("field_type", "string"))
        d["justification"] = Justification(d.get("justification", "left"))
        if "constraints" in d and isinstance(d["constraints"], dict):
            d["constraints"] = FieldConstraints.from_dict(d["constraints"])
        if "dependencies" in d:
            d["dependencies"] = [FieldDependency(**dep) for dep in d["dependencies"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Record-level structures ───────────────────────────────────────────────────

@dataclass
class RecordDef:
    """Definition of one record / row type within the spec."""
    record_type_id: str               # e.g. "1", "5", "6", "9" in ACH
    name: str
    category: RecordCategory = RecordCategory.UNKNOWN
    fields: list[FieldDef] = field(default_factory=list)
    fixed_width: bool = False
    record_length: int | None = None  # For fixed-width formats
    delimiter: str | None = None      # For delimited formats
    repeatable: bool = False          # Can this record type appear many times?
    min_occurrences: int = 0
    max_occurrences: int | None = None
    ordering: int = 0                 # Sort key for file assembly

    def get_field(self, name: str) -> FieldDef | None:
        return next((f for f in self.fields if f.name == name), None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["fields"] = [f.to_dict() for f in self.fields]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RecordDef:
        d = dict(d)
        d["category"] = RecordCategory(d.get("category", "unknown"))
        d["fields"] = [FieldDef.from_dict(f) for f in d.get("fields", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Top-level schema model ────────────────────────────────────────────────────

@dataclass
class FileStructure:
    """Describes the overall structure / envelope of the file."""
    format: str = "fixed_width"       # "fixed_width", "delimited", "json", "tagged"
    line_ending: str = "\r\n"
    encoding: str = "ascii"
    file_extension: str = ".txt"
    header_records: list[str] = field(default_factory=list)   # record_type_ids
    detail_records: list[str] = field(default_factory=list)
    trailer_records: list[str] = field(default_factory=list)
    batch_supported: bool = False
    batch_header_record: str | None = None
    batch_trailer_record: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FileStructure:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SchemaModel:
    """
    The central artefact produced by the training pipeline.
    One SchemaModel exists per uploaded + trained specification.
    It is serialised to JSON in the specs/ directory.
    """
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spec_name: str = ""
    spec_type: SpecType = SpecType.JSON
    version: str = "1.0"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trained_at: str | None = None
    source_files: list[str] = field(default_factory=list)

    # Core structural data
    file_structure: FileStructure = field(default_factory=FileStructure)
    records: list[RecordDef] = field(default_factory=list)

    # Global rules expressed as Python-evaluable expressions
    global_rules: list[str] = field(default_factory=list)

    # Metadata about inference quality
    inference_stats: dict[str, Any] = field(default_factory=dict)
    is_trained: bool = False

    # ── Convenience helpers ─────────────────────────────────────────────────

    def get_record(self, type_id: str) -> RecordDef | None:
        return next((r for r in self.records if r.record_type_id == type_id), None)

    def ordered_records(self) -> list[RecordDef]:
        return sorted(self.records, key=lambda r: r.ordering)

    # ── Serialisation ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "spec_name": self.spec_name,
            "spec_type": self.spec_type.value,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "trained_at": self.trained_at,
            "source_files": self.source_files,
            "file_structure": self.file_structure.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "global_rules": self.global_rules,
            "inference_stats": self.inference_stats,
            "is_trained": self.is_trained,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> SchemaModel:
        model = cls(
            spec_id=d.get("spec_id", str(uuid.uuid4())),
            spec_name=d.get("spec_name", ""),
            spec_type=SpecType(d.get("spec_type", "json")),
            version=d.get("version", "1.0"),
            description=d.get("description", ""),
            created_at=d.get("created_at", datetime.utcnow().isoformat()),
            trained_at=d.get("trained_at"),
            source_files=d.get("source_files", []),
            global_rules=d.get("global_rules", []),
            inference_stats=d.get("inference_stats", {}),
            is_trained=d.get("is_trained", False),
        )
        if "file_structure" in d:
            model.file_structure = FileStructure.from_dict(d["file_structure"])
        model.records = [RecordDef.from_dict(r) for r in d.get("records", [])]
        return model

    @classmethod
    def from_json(cls, json_str: str) -> SchemaModel:
        return cls.from_dict(json.loads(json_str))
