"""
ACH / NACHA Parser
==================
Parses ACH/NACHA specification files.  Two input modes are supported:

1. **Native ACH file** – an actual .ach/.txt file with 94-character records.
   The parser inspects the record-type codes (column 1) and reverse-engineers
   the field layout from the NACHA standard.

2. **ACH spec JSON** – a JSON document describing the record layout
   (useful when users upload a custom NACHA addenda spec).

NACHA standard record types:
  1  – File Header
  5  – Batch Header
  6  – Entry Detail
  7  – Addenda
  8  – Batch Control
  9  – File Control
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.models.schema import (
    SchemaModel, SpecType, RecordDef, FieldDef, FieldType,
    FieldConstraints, FileStructure, RecordCategory, Justification,
)
from app.parsers.base_parser import BaseParser
from app.core.exceptions import ParseError


# ---------------------------------------------------------------------------
# NACHA standard field definitions
# Each tuple: (name, start, end, field_type, description)
# Positions are 1-based, inclusive (NACHA convention)
# ---------------------------------------------------------------------------
_NACHA_RECORDS: dict[str, dict[str, Any]] = {
    "1": {
        "name": "File Header",
        "category": RecordCategory.HEADER,
        "ordering": 0,
        "fields": [
            ("record_type_code", 1, 1, FieldType.CONSTANT, "Always '1'"),
            ("priority_code", 2, 3, FieldType.NUMERIC, "Priority code, usually '01'"),
            ("immediate_destination", 4, 13, FieldType.ROUTING_NUMBER, "Receiving DFI routing number"),
            ("immediate_origin", 14, 23, FieldType.ROUTING_NUMBER, "Company EIN or routing number"),
            ("file_creation_date", 24, 29, FieldType.DATE, "YYMMDD"),
            ("file_creation_time", 30, 33, FieldType.STRING, "HHMM"),
            ("file_id_modifier", 34, 34, FieldType.STRING, "A-Z or 0-9"),
            ("record_size", 35, 37, FieldType.CONSTANT, "Always '094'"),
            ("blocking_factor", 38, 39, FieldType.CONSTANT, "Always '10'"),
            ("format_code", 40, 40, FieldType.CONSTANT, "Always '1'"),
            ("immediate_destination_name", 41, 63, FieldType.STRING, "Bank name"),
            ("immediate_origin_name", 64, 86, FieldType.STRING, "Company name"),
            ("reference_code", 87, 94, FieldType.STRING, "Optional reference"),
        ],
    },
    "5": {
        "name": "Batch Header",
        "category": RecordCategory.BATCH_HEADER,
        "ordering": 10,
        "fields": [
            ("record_type_code", 1, 1, FieldType.CONSTANT, "Always '5'"),
            ("service_class_code", 2, 4, FieldType.ENUM, "200=mixed, 220=credits, 225=debits"),
            ("company_name", 5, 20, FieldType.STRING, ""),
            ("company_discretionary_data", 21, 40, FieldType.STRING, ""),
            ("company_identification", 41, 50, FieldType.STRING, "10-char company ID"),
            ("standard_entry_class_code", 51, 53, FieldType.ENUM, "PPD, CCD, CTX, WEB, TEL…"),
            ("company_entry_description", 54, 63, FieldType.STRING, ""),
            ("company_descriptive_date", 64, 69, FieldType.DATE, "YYMMDD"),
            ("effective_entry_date", 70, 75, FieldType.DATE, "YYMMDD"),
            ("settlement_date", 76, 78, FieldType.STRING, "Filled by bank"),
            ("originator_status_code", 79, 79, FieldType.CONSTANT, "'1'"),
            ("odfi_routing_number", 80, 87, FieldType.ROUTING_NUMBER, "First 8 digits"),
            ("batch_number", 88, 94, FieldType.SEQUENCE, "Sequential batch number"),
        ],
    },
    "6": {
        "name": "Entry Detail",
        "category": RecordCategory.DETAIL,
        "ordering": 20,
        "fields": [
            ("record_type_code", 1, 1, FieldType.CONSTANT, "Always '6'"),
            ("transaction_code", 2, 3, FieldType.ENUM, "22=checking credit, 27=checking debit…"),
            ("rdfi_routing_transit_number", 4, 11, FieldType.ROUTING_NUMBER, "RDFI routing (8 digits)"),
            ("check_digit", 12, 12, FieldType.CHECKSUM, "Mod-10 check digit"),
            ("rdfi_account_number", 13, 29, FieldType.ACCOUNT_NUMBER, ""),
            ("amount", 30, 39, FieldType.AMOUNT, "In cents, no decimal"),
            ("individual_identification_number", 40, 54, FieldType.STRING, ""),
            ("individual_name", 55, 76, FieldType.STRING, ""),
            ("discretionary_data", 77, 78, FieldType.STRING, ""),
            ("addenda_record_indicator", 79, 79, FieldType.ENUM, "0=no addenda, 1=addenda follows"),
            ("trace_number", 80, 94, FieldType.SEQUENCE, "ODFI routing + seq number"),
        ],
    },
    "7": {
        "name": "Addenda Record",
        "category": RecordCategory.ADDENDA,
        "ordering": 25,
        "fields": [
            ("record_type_code", 1, 1, FieldType.CONSTANT, "Always '7'"),
            ("addenda_type_code", 2, 3, FieldType.ENUM, "05=general addenda"),
            ("payment_related_information", 4, 83, FieldType.STRING, ""),
            ("sequence_number", 84, 87, FieldType.SEQUENCE, ""),
            ("entry_detail_sequence_number", 88, 94, FieldType.SEQUENCE, "Last 7 of trace number"),
        ],
    },
    "8": {
        "name": "Batch Control",
        "category": RecordCategory.BATCH_TRAILER,
        "ordering": 90,
        "fields": [
            ("record_type_code", 1, 1, FieldType.CONSTANT, "Always '8'"),
            ("service_class_code", 2, 4, FieldType.ENUM, "Matches batch header"),
            ("entry_addenda_count", 5, 10, FieldType.NUMERIC, "Count of 6 and 7 records"),
            ("entry_hash", 11, 20, FieldType.CHECKSUM, "Sum of RDFI routing numbers mod 10^10"),
            ("total_debit_entry_dollar_amount", 21, 32, FieldType.AMOUNT, ""),
            ("total_credit_entry_dollar_amount", 33, 44, FieldType.AMOUNT, ""),
            ("company_identification", 45, 54, FieldType.STRING, ""),
            ("message_authentication_code", 55, 73, FieldType.STRING, ""),
            ("reserved", 74, 79, FieldType.CONSTANT, "Blanks"),
            ("odfi_routing_number", 80, 87, FieldType.ROUTING_NUMBER, ""),
            ("batch_number", 88, 94, FieldType.SEQUENCE, ""),
        ],
    },
    "9": {
        "name": "File Control",
        "category": RecordCategory.TRAILER,
        "ordering": 99,
        "fields": [
            ("record_type_code", 1, 1, FieldType.CONSTANT, "Always '9'"),
            ("batch_count", 2, 7, FieldType.NUMERIC, ""),
            ("block_count", 8, 13, FieldType.NUMERIC, "Total 10-record blocks"),
            ("entry_addenda_count", 14, 21, FieldType.NUMERIC, ""),
            ("entry_hash", 22, 31, FieldType.CHECKSUM, "Sum of all RDFI routing numbers"),
            ("total_debit_entry_dollar_amount", 32, 43, FieldType.AMOUNT, ""),
            ("total_credit_entry_dollar_amount", 44, 55, FieldType.AMOUNT, ""),
            ("reserved", 56, 94, FieldType.CONSTANT, "Blanks"),
        ],
    },
}

_ACH_SERVICE_CLASS_CODES = ["200", "220", "225"]
_ACH_TRANSACTION_CODES = ["22", "23", "24", "27", "28", "29", "32", "33", "34", "37", "38", "39"]
_ACH_SEC_CODES = ["PPD", "CCD", "CTX", "WEB", "TEL", "ARC", "BOC", "POP", "RCK"]


class ACHParser(BaseParser):
    """
    Parses native ACH files or ACH spec JSON documents.
    Produces a fully-structured SchemaModel for ACH/NACHA.
    """

    spec_type = SpecType.ACH

    def _do_parse(self, content: str, source_name: str) -> SchemaModel:
        content = content.strip()
        if not content:
            raise ParseError("ACH content is empty")

        # Determine if this is a JSON spec or a native ACH file
        if content.lstrip().startswith("{") or content.lstrip().startswith("["):
            return self._parse_json_spec(content, source_name)
        else:
            return self._parse_native_ach(content, source_name)

    # ── Native ACH file mode ─────────────────────────────────────────────────

    def _parse_native_ach(self, content: str, source_name: str) -> SchemaModel:
        """
        Read an actual ACH file, verify record-type codes are valid,
        and build a schema from the NACHA standard definitions.
        We also infer which record types actually appear in the file.
        """
        lines = [ln.rstrip("\n\r") for ln in content.splitlines()]
        found_types: set[str] = set()
        line_errors = 0

        for i, line in enumerate(lines, 1):
            if not line:
                continue
            if len(line) != 94:
                self._warn(f"Line {i} has length {len(line)}, expected 94. Padding/truncating.")
                line_errors += 1
            rt = line[0] if line else ""
            if rt in _NACHA_RECORDS:
                found_types.add(rt)
            else:
                self._warn(f"Unknown record type '{rt}' on line {i}")

        model = self._build_standard_model(source_name)
        model.inference_stats.update(
            {
                "total_lines": len(lines),
                "found_record_types": sorted(found_types),
                "malformed_lines": line_errors,
                "source": "native_ach",
            }
        )
        return model

    # ── JSON spec mode ───────────────────────────────────────────────────────

    def _parse_json_spec(self, content: str, source_name: str) -> SchemaModel:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid JSON in ACH spec: {exc}") from exc

        # If the JSON is a wrapper with 'records' key, parse it directly
        if isinstance(raw, dict) and "records" in raw:
            return self._parse_spec_dict(raw, source_name)

        # Otherwise treat it as a list of custom record definitions
        return self._build_standard_model(source_name)

    def _parse_spec_dict(self, raw: dict, source_name: str) -> SchemaModel:
        model = self._build_standard_model(source_name)
        # Allow the JSON to override specific field lists
        for rec_def_raw in raw.get("records", []):
            rt_id = str(rec_def_raw.get("record_type_id", ""))
            existing = model.get_record(rt_id)
            if existing and "fields" in rec_def_raw:
                self._warn(f"Custom fields override for record type {rt_id}")
        return model

    # ── Shared model builder ─────────────────────────────────────────────────

    def _build_standard_model(self, source_name: str) -> SchemaModel:
        fs = FileStructure(
            format="fixed_width",
            line_ending="\r\n",
            encoding="ascii",
            file_extension=".ach",
            header_records=["1"],
            detail_records=["5", "6", "7"],
            trailer_records=["8", "9"],
            batch_supported=True,
            batch_header_record="5",
            batch_trailer_record="8",
        )

        records: list[RecordDef] = []
        for rt_id, meta in _NACHA_RECORDS.items():
            rec = RecordDef(
                record_type_id=rt_id,
                name=meta["name"],
                category=meta["category"],
                fixed_width=True,
                record_length=94,
                repeatable=(rt_id in ("6", "7")),
                min_occurrences=1 if rt_id in ("1", "9") else 0,
                max_occurrences=None if rt_id in ("6", "7") else 1,
                ordering=meta["ordering"],
            )

            for fname, start, end, ftype, desc in meta["fields"]:
                length = end - start + 1
                constraints = self._build_constraints(fname, ftype, length)
                fd = FieldDef(
                    name=fname,
                    field_type=ftype,
                    start_pos=start,
                    end_pos=end,
                    length=length,
                    justification=(
                        Justification.RIGHT if ftype in (FieldType.NUMERIC, FieldType.AMOUNT, FieldType.SEQUENCE)
                        else Justification.LEFT
                    ),
                    pad_char="0" if ftype in (FieldType.NUMERIC, FieldType.AMOUNT, FieldType.SEQUENCE) else " ",
                    description=desc,
                    constraints=constraints,
                )
                rec.fields.append(fd)

            records.append(rec)

        return SchemaModel(
            spec_name=f"ACH_{source_name}",
            spec_type=SpecType.ACH,
            version="1.0",
            description="ACH/NACHA standard file specification",
            source_files=[source_name] if source_name else [],
            file_structure=fs,
            records=records,
            global_rules=[
                "entry_count_in_batch_control == len(entry_detail_records)",
                "batch_count_in_file_control == total_batches",
                "file_padded_to_multiple_of_10_records",
            ],
        )

    @staticmethod
    def _build_constraints(fname: str, ftype: FieldType, length: int) -> FieldConstraints:
        c = FieldConstraints(min_length=length, max_length=length)
        if ftype == FieldType.ENUM:
            if "service_class_code" in fname:
                c.allowed_values = _ACH_SERVICE_CLASS_CODES
            elif "transaction_code" in fname:
                c.allowed_values = _ACH_TRANSACTION_CODES
            elif "standard_entry_class" in fname:
                c.allowed_values = _ACH_SEC_CODES
            elif "addenda_record_indicator" in fname:
                c.allowed_values = ["0", "1"]
            elif "addenda_type_code" in fname:
                c.allowed_values = ["05"]
        elif ftype == FieldType.CONSTANT:
            pass  # default_value set at generation time
        elif ftype == FieldType.AMOUNT:
            c.min_value = 0
            c.max_value = 9_999_999_999
        elif ftype == FieldType.CHECKSUM:
            c.checksum_algorithm = "mod10"
        return c
