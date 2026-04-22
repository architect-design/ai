"""
File Generators
===============
BaseGenerator + concrete generators for ACH, VCF and JSON formats.

Each generator:
  1. Receives a trained SchemaModel + generation parameters
  2. Uses DataSynthesizer to produce field values
  3. Applies the RuleEngine for validation and post-processing
  4. Assembles the raw file bytes
  5. Returns the complete file content as a string + validation results
"""
from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.models.schema import (
    SchemaModel, RecordDef, FieldDef, FieldType,
    RecordCategory, FileStructure,
)
from app.generator.data_synthesizer import DataSynthesizer
from app.rule_engine.rule_engine import RuleEngine
from app.core.exceptions import GenerationError

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    content: str
    records: list[dict[str, Any]]
    validation_errors: list[str] = field(default_factory=list)
    validation_passed: bool = True
    record_count: int = 0
    file_extension: str = ".txt"


# ── Base Generator ────────────────────────────────────────────────────────────

class BaseGenerator(abc.ABC):

    def __init__(self, model: SchemaModel, seed: int | None = None):
        self.model = model
        self.synth = DataSynthesizer(seed=seed)
        self.rule_engine = RuleEngine()
        self.rule_engine.load_from_model(model)

    def generate(
        self,
        record_count: int = 10,
        overrides: dict[str, Any] | None = None,
    ) -> GenerationResult:
        overrides = overrides or {}
        logger.info("[%s] Generating %d records", self.__class__.__name__, record_count)
        self.synth.reset_counters()
        try:
            result = self._do_generate(record_count, overrides)
        except Exception as exc:
            raise GenerationError(f"Generation failed: {exc}") from exc

        # File-level validation
        file_errors = self.rule_engine.validate_file(result.records)
        result.validation_errors.extend(file_errors)
        result.validation_passed = len(result.validation_errors) == 0
        result.file_extension = self.model.file_structure.file_extension
        return result

    @abc.abstractmethod
    def _do_generate(self, record_count: int, overrides: dict) -> GenerationResult:
        """Subclasses implement format-specific generation."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _gen_record(
        self,
        rec_def: RecordDef,
        context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Generate one record dict: {field_name: value}."""
        ctx = context or {}
        row: dict[str, str] = {}

        ordered_fields = self.rule_engine.resolve_field_order(rec_def)

        for fd in ordered_fields:
            val = self.synth.generate(fd, ctx)
            row[fd.name] = val
            ctx[fd.name] = val  # make available for dependent fields

        return row

    def _serialise_fixed_width(self, rec_def: RecordDef, row: dict[str, str]) -> str:
        """Render a row dict into a fixed-width string."""
        if not rec_def.record_length:
            return "|".join(row.values())

        line = list(" " * rec_def.record_length)
        for fd in rec_def.fields:
            if fd.start_pos and fd.end_pos:
                val = row.get(fd.name, "")
                # Slice is 0-based; spec positions are 1-based
                start = fd.start_pos - 1
                end = fd.end_pos
                segment = val[:end - start].ljust(end - start)
                line[start:end] = list(segment)
        return "".join(line)

    def _serialise_delimited(
        self,
        rec_def: RecordDef,
        row: dict[str, str],
        delimiter: str = "|",
    ) -> str:
        return delimiter.join(row.get(fd.name, "") for fd in rec_def.fields)


# ── ACH Generator ─────────────────────────────────────────────────────────────

class ACHGenerator(BaseGenerator):
    """Generates valid NACHA ACH files."""

    def _do_generate(self, record_count: int, overrides: dict) -> GenerationResult:
        lines: list[str] = []
        all_rows: list[dict[str, Any]] = []
        validation_errors: list[str] = []

        # Prepare counters
        batch_count = max(1, record_count // 5)
        entries_per_batch = max(1, record_count // batch_count)
        total_entries = 0
        total_debits = 0
        total_credits = 0
        file_entry_hash = 0

        # ── File Header (Record Type 1) ──────────────────────────────────────
        fhr = self.model.get_record("1")
        if fhr:
            row = self._gen_record(fhr)
            row["record_type_code"] = "1"
            row["file_creation_date"] = date.today().strftime("%y%m%d")
            row["file_creation_time"] = "1200"
            row["file_id_modifier"] = "A"
            row["record_size"] = "094"
            row["blocking_factor"] = "10"
            row["format_code"] = "1"
            line = self._serialise_fixed_width(fhr, row)
            lines.append(line[:94].ljust(94))
            all_rows.append({"_record_type": "1", **row})

        odfi_routing = (overrides.get("odfi_routing") or
                        self.synth._gen_routing_number())

        for batch_num in range(1, batch_count + 1):
            batch_debit = 0
            batch_credit = 0
            batch_entry_hash = 0
            batch_entries = entries_per_batch if batch_num < batch_count else (
                record_count - (batch_count - 1) * entries_per_batch
            )

            # ── Batch Header (Record Type 5) ─────────────────────────────────
            bhr = self.model.get_record("5")
            if bhr:
                row = self._gen_record(bhr)
                row["record_type_code"] = "5"
                row["service_class_code"] = "200"
                row["company_identification"] = "1234567890"
                row["standard_entry_class_code"] = "PPD"
                row["effective_entry_date"] = date.today().strftime("%y%m%d")
                row["originator_status_code"] = "1"
                row["odfi_routing_number"] = odfi_routing[:8]
                row["batch_number"] = str(batch_num).zfill(7)
                line = self._serialise_fixed_width(bhr, row)
                lines.append(line[:94].ljust(94))
                all_rows.append({"_record_type": "5", **row})

            # ── Entry Detail Records (Record Type 6) ─────────────────────────
            edr = self.model.get_record("6")
            for entry_seq in range(1, batch_entries + 1):
                if edr:
                    row = self._gen_record(edr)
                    row["record_type_code"] = "6"
                    row["transaction_code"] = self.synth._rng.choice(["22", "27"])
                    rdfi = self.synth._gen_routing_number()
                    row["rdfi_routing_transit_number"] = rdfi[:8]
                    row["check_digit"] = rdfi[8]
                    row["addenda_record_indicator"] = "0"
                    row["trace_number"] = self.synth.gen_trace_number(
                        odfi_routing, total_entries + entry_seq
                    )
                    amount_str = row.get("amount", "0000001000")
                    try:
                        amount_int = int(amount_str.strip())
                    except ValueError:
                        amount_int = 1000
                    if row["transaction_code"] in ("22", "32"):
                        batch_credit += amount_int
                    else:
                        batch_debit += amount_int

                    rdfi_num = int(rdfi[:8])
                    batch_entry_hash += rdfi_num

                    line = self._serialise_fixed_width(edr, row)
                    lines.append(line[:94].ljust(94))
                    all_rows.append({"_record_type": "6", **row})

            total_entries += batch_entries
            total_debits += batch_debit
            total_credits += batch_credit
            file_entry_hash += batch_entry_hash

            # ── Batch Control (Record Type 8) ─────────────────────────────────
            bcr = self.model.get_record("8")
            if bcr:
                row = self._gen_record(bcr)
                row["record_type_code"] = "8"
                row["service_class_code"] = "200"
                row["entry_addenda_count"] = str(batch_entries).zfill(6)
                row["entry_hash"] = str(batch_entry_hash % 10**10).zfill(10)
                row["total_debit_entry_dollar_amount"] = str(batch_debit).zfill(12)
                row["total_credit_entry_dollar_amount"] = str(batch_credit).zfill(12)
                row["odfi_routing_number"] = odfi_routing[:8]
                row["batch_number"] = str(batch_num).zfill(7)
                row["reserved"] = " " * 6
                line = self._serialise_fixed_width(bcr, row)
                lines.append(line[:94].ljust(94))
                all_rows.append({"_record_type": "8", **row})

        # ── File Control (Record Type 9) ──────────────────────────────────────
        fcr = self.model.get_record("9")
        if fcr:
            total_records = len(lines) + 1  # +1 for this record
            block_count = max(1, -(-total_records // 10))  # ceiling division
            padding_records = block_count * 10 - total_records
            for _ in range(padding_records):
                lines.append("9" * 94)

            row = self._gen_record(fcr)
            row["record_type_code"] = "9"
            row["batch_count"] = str(batch_count).zfill(6)
            row["block_count"] = str(block_count).zfill(6)
            row["entry_addenda_count"] = str(total_entries).zfill(8)
            row["entry_hash"] = str(file_entry_hash % 10**10).zfill(10)
            row["total_debit_entry_dollar_amount"] = str(total_debits).zfill(12)
            row["total_credit_entry_dollar_amount"] = str(total_credits).zfill(12)
            row["reserved"] = " " * 39
            line = self._serialise_fixed_width(fcr, row)
            lines.append(line[:94].ljust(94))
            all_rows.append({"_record_type": "9", **row})

        line_ending = self.model.file_structure.line_ending or "\r\n"
        content = line_ending.join(lines)

        return GenerationResult(
            content=content,
            records=all_rows,
            record_count=total_entries,
            validation_errors=validation_errors,
        )


# ── VCF Generator ─────────────────────────────────────────────────────────────

class VCFGenerator(BaseGenerator):
    """Generates VISA VCF card file records."""

    def _do_generate(self, record_count: int, overrides: dict) -> GenerationResult:
        lines: list[str] = []
        all_rows: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        total_amount = 0

        # ── File Header ────────────────────────────────────────────────────────
        hdr_def = self.model.get_record("HEADER")
        if hdr_def:
            row = self._gen_record(hdr_def)
            row["record_type"] = "H"
            row["file_creation_date"] = date.today().strftime("%y%m%d")
            row["file_sequence"] = "0001"
            lines.append(self._serialise_fixed_width(hdr_def, row))
            all_rows.append({"_record_type": "HEADER", **row})

        # ── Detail Records ─────────────────────────────────────────────────────
        detail_def = self.model.get_record("DETAIL")
        for i in range(record_count):
            if detail_def:
                row = self._gen_record(detail_def)
                # Override specific fields for realism
                row["record_type"] = self.synth._rng.choice(["05", "06"])
                row["transaction_code"] = self.synth._rng.choice(["00", "20"])
                row["network_id"] = "0001"
                row["cvv2_result_code"] = self.synth._rng.choice(["M", "N", " "])
                row["avs_result_code"] = self.synth._rng.choice(["Y", "N", "A", " "])
                row["mcc"] = self.synth.gen_mcc()
                row["merchant_name"] = self.synth._rng.choice(
                    self.synth._MERCHANT_NAMES
                )[:25].ljust(25)
                row["merchant_city"] = self.synth._rng.choice(
                    self.synth._CITIES
                )[:13].ljust(13)

                try:
                    total_amount += int(row.get("transaction_amount", "0").strip() or "0")
                except ValueError:
                    pass

                # Validate record-level rules
                rec_errors = self.rule_engine.validate_record("DETAIL", row)
                validation_errors.extend(rec_errors[:3])  # cap errors per record

                lines.append(self._serialise_fixed_width(detail_def, row))
                all_rows.append({"_record_type": "DETAIL", **row})

        # ── File Trailer ───────────────────────────────────────────────────────
        trl_def = self.model.get_record("TRAILER")
        if trl_def:
            row = self._gen_record(trl_def)
            row["record_type"] = "T"
            row["total_records"] = str(record_count).zfill(10)
            row["total_amount"] = str(total_amount).zfill(14)
            lines.append(self._serialise_fixed_width(trl_def, row))
            all_rows.append({"_record_type": "TRAILER", **row})

        line_ending = self.model.file_structure.line_ending or "\n"
        content = line_ending.join(lines)

        return GenerationResult(
            content=content,
            records=all_rows,
            record_count=record_count,
            validation_errors=validation_errors,
        )


# ── JSON / Generic Generator ──────────────────────────────────────────────────

class JSONGenerator(BaseGenerator):
    """Generates JSON or delimited files from JSON/custom spec schemas."""

    def _do_generate(self, record_count: int, overrides: dict) -> GenerationResult:
        is_json = self.model.file_structure.format == "json"
        delimiter = (
            self.model.file_structure.__dict__.get("delimiter") or "|"
        )

        # Find the primary detail record
        detail_recs = [
            r for r in self.model.ordered_records()
            if r.category in (RecordCategory.DETAIL, RecordCategory.UNKNOWN)
        ]
        header_recs = [r for r in self.model.ordered_records()
                       if r.category == RecordCategory.HEADER]
        trailer_recs = [r for r in self.model.ordered_records()
                        if r.category == RecordCategory.TRAILER]

        lines: list[str] = []
        all_rows: list[dict[str, Any]] = []
        json_objects: list[dict] = []
        validation_errors: list[str] = []

        # Header records
        for rec_def in header_recs:
            row = self._gen_record(rec_def)
            all_rows.append({"_record_type": rec_def.record_type_id, **row})
            if not is_json:
                lines.append(self._serialise_delimited(rec_def, row, delimiter))

        # Detail records
        primary_detail = detail_recs[0] if detail_recs else None
        if primary_detail:
            for _ in range(record_count):
                row = self._gen_record(primary_detail)
                rec_errors = self.rule_engine.validate_record(
                    primary_detail.record_type_id, row
                )
                validation_errors.extend(rec_errors[:2])
                all_rows.append({"_record_type": primary_detail.record_type_id, **row})
                if is_json:
                    json_objects.append({k: v for k, v in row.items()})
                else:
                    lines.append(self._serialise_delimited(primary_detail, row, delimiter))

        # Trailer records
        for rec_def in trailer_recs:
            row = self._gen_record(rec_def)
            all_rows.append({"_record_type": rec_def.record_type_id, **row})
            if not is_json:
                lines.append(self._serialise_delimited(rec_def, row, delimiter))

        if is_json:
            content = json.dumps(json_objects, indent=2)
        else:
            line_ending = self.model.file_structure.line_ending or "\n"
            content = line_ending.join(lines)

        return GenerationResult(
            content=content,
            records=all_rows,
            record_count=len(json_objects) if is_json else record_count,
            validation_errors=validation_errors,
        )


# ── Generator factory ─────────────────────────────────────────────────────────

def get_generator(model: SchemaModel, seed: int | None = None) -> BaseGenerator:
    from app.models.schema import SpecType
    mapping = {
        SpecType.ACH:    ACHGenerator,
        SpecType.VCF:    VCFGenerator,
        SpecType.JSON:   JSONGenerator,
        SpecType.SAMPLE: JSONGenerator,
    }
    cls = mapping.get(model.spec_type, JSONGenerator)
    return cls(model, seed=seed)
