"""
VISA VCF (Visa Card File) Parser
=================================
Handles two common representations of VCF specifications:

1. **Fixed-position spec** – a text/JSON file listing fields, their
   start/end positions and types (similar to NACHA).

2. **Tagged / key=value format** – each record is a series of
   TAG=VALUE pairs separated by a delimiter (common in acquirer feeds).

3. **Native VCF sample** – lines of tagged data the parser can reverse-
   engineer the field set from.

Because Visa's actual VCF specification is proprietary, the built-in
field set covers the most common fields found in publicly documented
Visa Base II clearing records.
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
# Built-in VCF field catalogue
# ---------------------------------------------------------------------------
_VCF_FIELDS: list[tuple[str, int, int, FieldType, str]] = [
    # (name, start, end, type, description)  — 1-based, inclusive
    ("record_type",            1,   2,  FieldType.ENUM,           "05=auth, 06=clearing…"),
    ("transaction_code",       3,   4,  FieldType.ENUM,           "Purchase, refund, etc."),
    ("primary_account_number", 5,  20,  FieldType.PAN,            "16-digit PAN, left-justified"),
    ("processing_code",       21,  26,  FieldType.ENUM,           "ISO 8583 processing code"),
    ("transaction_amount",    27,  36,  FieldType.AMOUNT,         "10-digit, no decimal"),
    ("transaction_currency",  37,  39,  FieldType.ENUM,           "ISO 4217 numeric"),
    ("transaction_date",      40,  45,  FieldType.DATE,           "YYMMDD"),
    ("transaction_time",      46,  51,  FieldType.STRING,         "HHMMSS"),
    ("expiry_date",           52,  55,  FieldType.EXPIRY,         "YYMM"),
    ("pos_entry_mode",        56,  58,  FieldType.ENUM,           "010=manual, 020=mag, 051=chip"),
    ("pos_condition_code",    59,  60,  FieldType.ENUM,           "00=normal"),
    ("acquirer_institution_id", 61, 71, FieldType.NUMERIC,        "11-digit BIN"),
    ("retrieval_reference_number", 72, 83, FieldType.STRING,      "12-char unique ref"),
    ("authorization_code",    84,  89,  FieldType.STRING,         "6-char auth code"),
    ("response_code",         90,  91,  FieldType.ENUM,           "00=approved"),
    ("terminal_id",           92,  99,  FieldType.STRING,         "8-char terminal ID"),
    ("merchant_id",          100, 115,  FieldType.STRING,         "15-char merchant ID"),
    ("merchant_name",        116, 140,  FieldType.STRING,         "25-char merchant name"),
    ("merchant_city",        141, 153,  FieldType.STRING,         "13-char city"),
    ("merchant_country",     154, 156,  FieldType.ENUM,           "ISO 3166 numeric"),
    ("mcc",                  157, 160,  FieldType.ENUM,           "Merchant category code"),
    ("network_id",           161, 164,  FieldType.ENUM,           "0001=Visa"),
    ("cvv2_result_code",     165, 165,  FieldType.ENUM,           "M=match, N=no match"),
    ("avs_result_code",      166, 166,  FieldType.ENUM,           "Y/N/A/Z…"),
    ("settle_date",          167, 172,  FieldType.DATE,           "YYMMDD"),
    ("interchange_fee",      173, 180,  FieldType.AMOUNT,         "Signed 8-digit"),
    ("cardholder_name",      181, 206,  FieldType.STRING,         "26-char name"),
    ("filler",               207, 220,  FieldType.CONSTANT,       "Reserved spaces"),
]

_VCF_RECORD_TYPES = {
    "05": "Authorization",
    "06": "Financial / Clearing",
    "07": "Reversal",
    "08": "Chargeback",
}

_VCF_TRANSACTION_CODES = {
    "00": "Purchase",
    "01": "Cash Advance",
    "20": "Refund",
    "28": "Chargeback",
    "54": "Balance Inquiry",
}

_VCF_CURRENCY_CODES = ["840", "978", "826", "036", "124", "356", "392"]
_VCF_POS_ENTRY_MODES = ["010", "020", "051", "071", "081", "091"]
_RESPONSE_CODES = ["00", "01", "05", "12", "14", "51", "54", "57", "61", "91"]


class VCFParser(BaseParser):
    """
    Parses VISA VCF specification files or sample VCF data.
    Produces a SchemaModel pre-loaded with the Visa Base II field set.
    """

    spec_type = SpecType.VCF

    def _do_parse(self, content: str, source_name: str) -> SchemaModel:
        content = content.strip()
        if not content:
            raise ParseError("VCF content is empty")

        if content.lstrip().startswith("{"):
            return self._parse_json_spec(json.loads(content), source_name)
        elif self._looks_like_tagged(content):
            return self._parse_tagged(content, source_name)
        else:
            return self._build_standard_model(source_name)

    # ── JSON spec mode ───────────────────────────────────────────────────────

    def _parse_json_spec(self, raw: dict, source_name: str) -> SchemaModel:
        model = self._build_standard_model(source_name)
        # Allow spec JSON to add extra fields or override descriptions
        extra_fields = raw.get("extra_fields", [])
        detail_rec = model.get_record("DETAIL")
        if detail_rec:
            for ef in extra_fields:
                fd = FieldDef(
                    name=self._normalise_name(ef.get("name", "unknown")),
                    field_type=FieldType(ef.get("type", "string")),
                    start_pos=ef.get("start"),
                    end_pos=ef.get("end"),
                    length=ef.get("length"),
                    description=ef.get("description", ""),
                )
                detail_rec.fields.append(fd)
        return model

    # ── Tagged format mode ───────────────────────────────────────────────────

    def _looks_like_tagged(self, content: str) -> bool:
        first_line = content.splitlines()[0] if content else ""
        return bool(re.search(r"[A-Z_]{2,20}=", first_line))

    def _parse_tagged(self, content: str, source_name: str) -> SchemaModel:
        model = self._build_standard_model(source_name)
        # Collect all tag names that actually appear
        all_tags: set[str] = set()
        for line in content.splitlines()[:200]:  # sample first 200 lines
            pairs = re.findall(r"([A-Z_]{2,30})=([^|,\n]*)", line)
            all_tags.update(tag for tag, _ in pairs)

        detail_rec = model.get_record("DETAIL")
        if detail_rec:
            existing = {f.name for f in detail_rec.fields}
            for tag in sorted(all_tags - existing):
                detail_rec.fields.append(FieldDef(
                    name=self._normalise_name(tag),
                    field_type=FieldType.STRING,
                    description=f"Inferred from tagged field: {tag}",
                    inferred_confidence=0.5,
                ))

        model.file_structure.format = "tagged"
        model.inference_stats["tagged_fields_found"] = sorted(all_tags)
        return model

    # ── Standard model builder ───────────────────────────────────────────────

    def _build_standard_model(self, source_name: str) -> SchemaModel:
        fs = FileStructure(
            format="fixed_width",
            line_ending="\n",
            encoding="ascii",
            file_extension=".vcf",
            header_records=["HEADER"],
            detail_records=["DETAIL"],
            trailer_records=["TRAILER"],
            batch_supported=False,
        )

        header_rec = RecordDef(
            record_type_id="HEADER",
            name="File Header",
            category=RecordCategory.HEADER,
            fixed_width=True,
            record_length=220,
            repeatable=False,
            min_occurrences=1,
            max_occurrences=1,
            ordering=0,
            fields=[
                FieldDef("record_type", FieldType.CONSTANT, 1, 1, 1,
                         default_value="H", description="Header sentinel"),
                FieldDef("file_creation_date", FieldType.DATE, 2, 7, 6,
                         format_string="YYMMDD"),
                FieldDef("institution_id", FieldType.NUMERIC, 8, 18, 11),
                FieldDef("file_sequence", FieldType.SEQUENCE, 19, 22, 4,
                         justification=Justification.RIGHT, pad_char="0"),
            ],
        )

        detail_rec = RecordDef(
            record_type_id="DETAIL",
            name="Transaction Detail",
            category=RecordCategory.DETAIL,
            fixed_width=True,
            record_length=220,
            repeatable=True,
            min_occurrences=0,
            max_occurrences=None,
            ordering=10,
        )
        for fname, start, end, ftype, desc in _VCF_FIELDS:
            length = end - start + 1
            just = (
                Justification.RIGHT
                if ftype in (FieldType.AMOUNT, FieldType.NUMERIC, FieldType.SEQUENCE)
                else Justification.LEFT
            )
            pad = "0" if just == Justification.RIGHT else " "
            constraints = self._field_constraints(fname, ftype, length)
            detail_rec.fields.append(FieldDef(
                name=fname,
                field_type=ftype,
                start_pos=start,
                end_pos=end,
                length=length,
                justification=just,
                pad_char=pad,
                description=desc,
                constraints=constraints,
            ))

        trailer_rec = RecordDef(
            record_type_id="TRAILER",
            name="File Trailer",
            category=RecordCategory.TRAILER,
            fixed_width=True,
            record_length=220,
            repeatable=False,
            min_occurrences=1,
            max_occurrences=1,
            ordering=99,
            fields=[
                FieldDef("record_type", FieldType.CONSTANT, 1, 1, 1,
                         default_value="T"),
                FieldDef("total_records", FieldType.NUMERIC, 2, 11, 10,
                         justification=Justification.RIGHT, pad_char="0",
                         description="Count of detail records"),
                FieldDef("total_amount", FieldType.AMOUNT, 12, 25, 14,
                         justification=Justification.RIGHT, pad_char="0"),
            ],
        )

        return SchemaModel(
            spec_name=f"VCF_{source_name}",
            spec_type=SpecType.VCF,
            version="1.0",
            description="VISA Card File (Base II clearing) specification",
            source_files=[source_name] if source_name else [],
            file_structure=fs,
            records=[header_rec, detail_rec, trailer_rec],
            global_rules=[
                "trailer.total_records == len(detail_records)",
                "trailer.total_amount == sum(detail.transaction_amount)",
                "pan must pass Luhn check",
            ],
        )

    @staticmethod
    def _field_constraints(fname: str, ftype: FieldType, length: int) -> FieldConstraints:
        c = FieldConstraints(min_length=length, max_length=length)
        if ftype == FieldType.PAN:
            c.pattern = r"^\d{13,19}\s*$"
            c.checksum_algorithm = "luhn"
        elif ftype == FieldType.EXPIRY:
            c.pattern = r"^\d{4}$"
        elif ftype == FieldType.AMOUNT:
            c.min_value = 0
            c.pattern = r"^\d+$"
        elif ftype == FieldType.ENUM:
            if "record_type" in fname:
                c.allowed_values = list(_VCF_RECORD_TYPES)
            elif "transaction_code" in fname:
                c.allowed_values = list(_VCF_TRANSACTION_CODES)
            elif "transaction_currency" in fname:
                c.allowed_values = _VCF_CURRENCY_CODES
            elif "pos_entry_mode" in fname:
                c.allowed_values = _VCF_POS_ENTRY_MODES
            elif "response_code" in fname:
                c.allowed_values = _RESPONSE_CODES
            elif fname == "cvv2_result_code":
                c.allowed_values = ["M", "N", "P", "U", "S", " "]
            elif fname == "avs_result_code":
                c.allowed_values = ["Y", "N", "A", "Z", "W", "U", " "]
            elif fname == "network_id":
                c.allowed_values = ["0001"]
        return c
