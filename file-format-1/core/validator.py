"""
Validation Engine
==================
Validates uploaded data files against a stored spec.
Produces a structured report with field-level errors.
"""

import re
import os
import json
from datetime import datetime
from typing import Any


class ValidationResult:
    def __init__(self):
        self.errors:    list[dict] = []
        self.warnings:  list[dict] = []
        self.passed:    int = 0
        self.failed:    int = 0
        self.total_records: int = 0

    def add_error(self, record: int, field: str, message: str, value: Any = None):
        self.errors.append({
            "record": record, "field": field,
            "message": message, "value": str(value)[:80],
        })
        self.failed += 1

    def add_warning(self, record: int, field: str, message: str, value: Any = None):
        self.warnings.append({
            "record": record, "field": field,
            "message": message, "value": str(value)[:80],
        })

    def add_pass(self):
        self.passed += 1

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def score(self) -> float:
        total = self.passed + self.failed
        return (self.passed / total * 100) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "is_valid":      self.is_valid,
            "score":         round(self.score, 2),
            "passed":        self.passed,
            "failed":        self.failed,
            "total_records": self.total_records,
            "errors":        self.errors[:200],   # cap for UI
            "warnings":      self.warnings[:100],
        }


class FieldValidator:
    """Single-field rule-based validator."""

    def validate(self, value: str, field: dict, result: ValidationResult, rec_no: int):
        name   = field.get('name', 'UNKNOWN')
        dtype  = field.get('data_type', 'alphanumeric')
        length = field.get('length')
        req    = field.get('required', True)
        rules  = field.get('validation', {})

        # ── Presence ────────────────────────────────────────────────
        if req and (value is None or str(value).strip() == ''):
            result.add_error(rec_no, name, "Required field is empty", value)
            return
        if not req and (value is None or str(value).strip() == ''):
            result.add_pass()
            return

        sval = str(value).strip()

        # ── Length ──────────────────────────────────────────────────
        if length:
            if len(sval) > length:
                result.add_error(rec_no, name,
                    f"Value exceeds max length {length} (got {len(sval)})", sval)
            else:
                result.add_pass()

        # ── Data type ───────────────────────────────────────────────
        if dtype == 'numeric':
            if not re.fullmatch(r'\d+', sval):
                result.add_error(rec_no, name, "Must be numeric digits only", sval)
            else:
                result.add_pass()

        elif dtype == 'alpha':
            if not re.fullmatch(r'[A-Za-z ]+', sval):
                result.add_error(rec_no, name, "Must be alphabetic characters only", sval)
            else:
                result.add_pass()

        elif dtype == 'amount':
            if not re.fullmatch(r'\d+(\.\d{0,2})?', sval):
                result.add_error(rec_no, name,
                    "Amount must be numeric with up to 2 decimal places", sval)
            else:
                result.add_pass()

        elif dtype == 'date':
            fmt = field.get('format', 'YYMMDD')
            if not self._validate_date(sval, fmt):
                result.add_error(rec_no, name,
                    f"Invalid date format (expected {fmt})", sval)
            else:
                result.add_pass()

        elif dtype == 'routing_number':
            if not self._validate_routing(sval):
                result.add_error(rec_no, name,
                    "Invalid ABA routing number (checksum failed)", sval)
            else:
                result.add_pass()

        else:
            result.add_pass()

        # ── Domain rules ────────────────────────────────────────────
        allowed = rules.get('allowed')
        if allowed and sval not in allowed:
            result.add_error(rec_no, name,
                f"Value '{sval}' not in allowed set {allowed}", sval)

        min_val = rules.get('min')
        max_val = rules.get('max')
        if min_val is not None and max_val is not None:
            try:
                num = float(sval)
                if not (min_val <= num <= max_val):
                    result.add_error(rec_no, name,
                        f"Value {num} out of range [{min_val}, {max_val}]", sval)
                else:
                    result.add_pass()
            except ValueError:
                pass

    def _validate_date(self, value: str, fmt: str) -> bool:
        fmt_map = {
            'YYMMDD':     (r'^\d{6}$',   '%y%m%d'),
            'YYYYMMDD':   (r'^\d{8}$',   '%Y%m%d'),
            'MMDD':       (r'^\d{4}$',   '%m%d'),
            'DD-MON-YYYY':(r'^\d{2}-[A-Z]{3}-\d{4}$', '%d-%b-%Y'),
        }
        pattern, py_fmt = fmt_map.get(fmt, (r'^\d{6,8}$', '%y%m%d'))
        if not re.match(pattern, value, re.IGNORECASE):
            return False
        try:
            datetime.strptime(value.upper(), py_fmt)
            return True
        except ValueError:
            return False

    def _validate_routing(self, value: str) -> bool:
        """ABA routing number Luhn-style checksum."""
        v = re.sub(r'\D', '', value)
        if len(v) != 9:
            return False
        w = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(v[i]) * w[i] for i in range(9))
        return total % 10 == 0


class Validator:
    """Main validation orchestrator for all supported formats."""

    def __init__(self, kb):
        self.kb  = kb
        self.fv  = FieldValidator()

    def validate(self, spec_name: str, file_content: str,
                 delimiter: str | None = None) -> ValidationResult:
        spec = self.kb.load(spec_name)
        if not spec:
            r = ValidationResult()
            r.add_error(0, 'SPEC', f"Unknown spec: {spec_name}")
            return r

        fmt = spec.get('format_type', 'custom')

        if fmt == 'nacha':
            return self._validate_nacha(spec, file_content)
        elif fmt == 'oracle_gl':
            return self._validate_delimited(spec, file_content,
                                            delimiter or spec.get('delimiter', '|'))
        elif fmt in ('visa_vcf', 'custom'):
            return self._validate_fixed_or_delimited(spec, file_content, delimiter)
        else:
            return self._validate_generic(spec, file_content, delimiter)

    # ── NACHA fixed-width ─────────────────────────────────────────────
    def _validate_nacha(self, spec: dict, content: str) -> ValidationResult:
        result = ValidationResult()
        lines  = [l for l in content.splitlines() if l.strip()]
        result.total_records = len(lines)
        records_by_type = spec.get('records', {})

        for i, line in enumerate(lines, 1):
            if len(line) != 94:
                result.add_warning(i, 'RECORD_LENGTH',
                    f"Expected 94 chars, got {len(line)}", line)
            rt = line[0] if line else '?'
            rec_def = records_by_type.get(rt)
            if not rec_def:
                result.add_warning(i, 'RECORD_TYPE',
                    f"Unknown record type '{rt}'", rt)
                continue
            for field in rec_def.get('fields', []):
                s, e = field.get('start', 1) - 1, field.get('end', 1)
                val  = line[s:e] if e <= len(line) else ''
                # Fixed value check
                if 'value' in field and val.strip() != field['value']:
                    result.add_error(i, field['name'],
                        f"Fixed value must be '{field['value']}'", val.strip())
                else:
                    self.fv.validate(val.strip(), field, result, i)
        return result

    # ── Delimited (Oracle GL etc.) ─────────────────────────────────────
    def _validate_delimited(self, spec: dict, content: str,
                             delimiter: str) -> ValidationResult:
        result = ValidationResult()
        lines  = [l for l in content.splitlines() if l.strip()]
        fields = spec.get('fields', [])
        has_hdr = spec.get('has_header', True)

        start = 1 if has_hdr else 0
        result.total_records = max(0, len(lines) - start)

        for i, line in enumerate(lines[start:], 1):
            parts = line.split(delimiter)
            for j, field in enumerate(fields):
                val = parts[j] if j < len(parts) else ''
                self.fv.validate(val.strip(), field, result, i)
        return result

    # ── Auto-detect fixed vs delimited ────────────────────────────────
    def _validate_fixed_or_delimited(self, spec: dict, content: str,
                                      delimiter: str | None) -> ValidationResult:
        sample = content[:500]
        if delimiter and delimiter in sample:
            return self._validate_delimited(spec, content, delimiter)
        return self._validate_nacha(spec, content)   # try fixed-width

    # ── Generic line-by-line ───────────────────────────────────────────
    def _validate_generic(self, spec: dict, content: str,
                           delimiter: str | None) -> ValidationResult:
        delim  = delimiter or ','
        result = ValidationResult()
        lines  = [l for l in content.splitlines() if l.strip()]
        fields = spec.get('fields', [])
        result.total_records = len(lines)

        for i, line in enumerate(lines, 1):
            parts = line.split(delim)
            for j, field in enumerate(fields):
                val = parts[j] if j < len(parts) else ''
                self.fv.validate(val.strip(), field, result, i)
        return result
