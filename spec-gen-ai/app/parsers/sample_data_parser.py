"""
Sample Data Parser
==================
Infers a SchemaModel by statistically analysing rows of actual sample
data (CSV, fixed-width text, pipe-delimited, etc.).

Custom ML/inference logic (no external models):
  - Column boundary detection via column-position entropy
  - Field type inference via regex voting + statistical checks
  - Constraint derivation (min/max, allowed values, length distribution)
  - Pattern repetition detection for segmented files
"""
from __future__ import annotations

import csv
import io
import re
import statistics
from collections import Counter
from typing import Any

from app.models.schema import (
    SchemaModel, SpecType, RecordDef, FieldDef, FieldType,
    FieldConstraints, FileStructure, RecordCategory, Justification,
)
from app.parsers.base_parser import BaseParser
from app.core.exceptions import ParseError, InsufficientSampleDataError


# ---------------------------------------------------------------------------
# Heuristic patterns for type inference (vote-based)
# ---------------------------------------------------------------------------
_TYPE_PATTERNS: list[tuple[FieldType, str, float]] = [
    # (type, regex, weight)
    (FieldType.DATE,           r"^\d{8}$",                              1.5),  # YYYYMMDD / MMDDYYYY
    (FieldType.DATE,           r"^\d{4}-\d{2}-\d{2}$",                 2.0),  # ISO date
    (FieldType.DATETIME,       r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",   2.0),
    (FieldType.PAN,            r"^\d{13,19}$",                          1.2),  # Candidate PAN
    (FieldType.EXPIRY,         r"^\d{4}$",                              0.8),  # YYMM (weak)
    (FieldType.ROUTING_NUMBER, r"^\d{9}$",                              1.3),
    (FieldType.ACCOUNT_NUMBER, r"^\d{6,17}$",                           0.9),
    (FieldType.AMOUNT,         r"^\d+\.\d{2}$",                         2.0),  # 123.45
    (FieldType.AMOUNT,         r"^\d{5,12}$",                           0.7),  # Cents format
    (FieldType.NUMERIC,        r"^-?\d+$",                              1.0),
    (FieldType.ALPHANUMERIC,   r"^[A-Za-z0-9 ]+$",                      0.5),
    (FieldType.STRING,         r".*",                                   0.1),  # fallback
]


class SampleDataParser(BaseParser):
    """
    Analyses raw sample data files and infers a full SchemaModel.
    Supports: CSV, TSV, pipe-delimited, fixed-width, and JSON arrays.
    """

    spec_type = SpecType.SAMPLE

    def _do_parse(self, content: str, source_name: str) -> SchemaModel:
        content = content.strip()
        if not content:
            raise ParseError("Sample file is empty")

        lines = content.splitlines()
        if len(lines) < 2:
            raise InsufficientSampleDataError(
                f"Sample file has only {len(lines)} line(s); "
                "need at least 2 rows for inference."
            )

        # Detect format
        delimiter, is_fixed = self._detect_format(lines[:20])

        if is_fixed:
            rows, col_positions = self._parse_fixed_width(lines)
        else:
            rows, col_positions = self._parse_delimited(content, delimiter)

        if not rows:
            raise ParseError("Could not extract any rows from the sample file")

        # Extract headers (first row) and data rows
        if rows and all(re.search(r"[A-Za-z_]", v) for v in rows[0].values()):
            headers = list(rows[0].values())
            data_rows = rows[1:]
        else:
            n_cols = len(rows[0])
            headers = [f"field_{i+1:03d}" for i in range(n_cols)]
            data_rows = rows

        if len(data_rows) < 1:
            raise InsufficientSampleDataError("No data rows after header extraction")

        # Transpose: list of column vectors
        columns: dict[str, list[str]] = {h: [] for h in headers}
        for row in data_rows:
            vals = list(row.values())
            for i, h in enumerate(headers):
                columns[h].append(vals[i] if i < len(vals) else "")

        # Build field definitions
        fields: list[FieldDef] = []
        for col_idx, header in enumerate(headers):
            col_vals = [v.strip() for v in columns[header]]
            fd = self._infer_field(header, col_vals, col_idx, col_positions)
            fields.append(fd)

        rec = RecordDef(
            record_type_id="RECORD",
            name="Inferred Record",
            category=RecordCategory.DETAIL,
            fixed_width=is_fixed,
            record_length=max(len(ln) for ln in lines) if is_fixed else None,
            delimiter=None if is_fixed else delimiter,
            repeatable=True,
            min_occurrences=0,
            max_occurrences=None,
            ordering=10,
            fields=fields,
        )

        # Check for header/trailer pattern (first/last rows look different)
        records, fs = self._detect_header_trailer(lines, rec, delimiter, is_fixed)

        model = SchemaModel(
            spec_name=source_name,
            spec_type=SpecType.SAMPLE,
            description="Schema inferred from sample data file",
            source_files=[source_name] if source_name else [],
            file_structure=fs,
            records=records,
            inference_stats={
                "total_sample_rows": len(data_rows),
                "columns_detected": len(headers),
                "delimiter": repr(delimiter) if not is_fixed else "fixed-width",
                "fixed_width": is_fixed,
            },
        )
        return model

    # ── Format detection ─────────────────────────────────────────────────────

    def _detect_format(self, lines: list[str]) -> tuple[str, bool]:
        """Returns (delimiter, is_fixed_width)."""
        # Count delimiter candidates in first lines
        candidates = ["|", "\t", ",", ";", "~"]
        counts: dict[str, list[int]] = {d: [] for d in candidates}
        for line in lines:
            for d in candidates:
                counts[d].append(line.count(d))

        # A consistent non-zero count strongly suggests that delimiter
        for d in candidates:
            vals = counts[d]
            if vals and all(v == vals[0] and v > 0 for v in vals):
                return d, False

        # Check for CSV
        csv_counts = [line.count(",") for line in lines]
        if csv_counts and statistics.stdev(csv_counts) < 2 and csv_counts[0] > 0:
            return ",", False

        # Assume fixed-width if all lines have the same length
        lengths = [len(ln) for ln in lines if ln]
        if lengths and max(lengths) - min(lengths) <= 2:
            return "", True

        # Fallback to comma
        return ",", False

    # ── Delimited parsing ────────────────────────────────────────────────────

    def _parse_delimited(
        self, content: str, delimiter: str
    ) -> tuple[list[dict[int, str]], dict[int, tuple[int, int]]]:
        reader = csv.reader(io.StringIO(content), delimiter=delimiter or ",")
        rows = []
        for row in reader:
            if any(cell.strip() for cell in row):  # skip blank rows
                rows.append({i: cell for i, cell in enumerate(row)})
        col_positions: dict[int, tuple[int, int]] = {}  # not meaningful for delimited
        return rows, col_positions

    # ── Fixed-width parsing ──────────────────────────────────────────────────

    def _parse_fixed_width(
        self, lines: list[str]
    ) -> tuple[list[dict[int, str]], dict[int, tuple[int, int]]]:
        """
        Detect column boundaries using character-position entropy.
        Positions where spaces are common across all rows are likely boundaries.
        """
        if not lines:
            return [], {}

        max_len = max(len(ln) for ln in lines)
        padded = [ln.ljust(max_len) for ln in lines]

        # Space frequency per position
        space_freq = [
            sum(1 for ln in padded if ln[i] == " ") / len(padded)
            for i in range(max_len)
        ]

        # Find boundaries: positions where space_freq > 0.8
        boundaries: list[int] = [0]
        prev_was_boundary = False
        for i, freq in enumerate(space_freq):
            if freq > 0.8 and not prev_was_boundary:
                boundaries.append(i)
                prev_was_boundary = True
            else:
                prev_was_boundary = False
        boundaries.append(max_len)

        # Convert boundaries to column ranges
        col_positions: dict[int, tuple[int, int]] = {}
        ranges: list[tuple[int, int]] = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start > 1:  # ignore single-char gaps
                ranges.append((start, end))
                col_positions[len(ranges) - 1] = (start, end)

        rows: list[dict[int, str]] = []
        for ln in lines:
            row = {}
            for col_idx, (start, end) in enumerate(ranges):
                row[col_idx] = ln[start:end].strip()
            rows.append(row)

        return rows, col_positions

    # ── Field inference ───────────────────────────────────────────────────────

    def _infer_field(
        self,
        header: str,
        values: list[str],
        col_idx: int,
        col_positions: dict[int, tuple[int, int]],
    ) -> FieldDef:
        non_empty = [v for v in values if v]
        name = self._normalise_name(header)

        # Length statistics
        lengths = [len(v) for v in non_empty] if non_empty else [0]
        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        try:
            mean_len = statistics.mean(lengths) if lengths else 0
            std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
        except statistics.StatisticsError:
            mean_len, std_len = 0, 0

        # Fixed-width position
        start_pos, end_pos = col_positions.get(col_idx, (None, None))

        # Type inference via voting
        ftype, confidence = self._vote_for_type(non_empty, name)

        # Constraint derivation
        constraints = self._derive_constraints(ftype, non_empty, min_len, max_len)

        # Justification heuristic
        just = Justification.RIGHT if ftype in (
            FieldType.NUMERIC, FieldType.AMOUNT, FieldType.SEQUENCE
        ) else Justification.LEFT

        return FieldDef(
            name=name,
            field_type=ftype,
            start_pos=start_pos,
            end_pos=end_pos,
            length=max_len or None,
            justification=just,
            pad_char="0" if just == Justification.RIGHT else " ",
            inferred_confidence=confidence,
            sample_values=non_empty[:10],
            constraints=constraints,
            description=f"Inferred from sample (col {col_idx}, μlen={mean_len:.1f}±{std_len:.1f})",
        )

    def _vote_for_type(self, values: list[str], name: str) -> tuple[FieldType, float]:
        """
        Each regex pattern in _TYPE_PATTERNS casts a weighted vote.
        The winning type and its confidence score are returned.
        Name heuristics get bonus weight.
        """
        if not values:
            return FieldType.STRING, 0.0

        votes: dict[FieldType, float] = {}
        n = len(values)

        for ftype, pattern, weight in _TYPE_PATTERNS:
            matches = sum(1 for v in values if re.fullmatch(pattern, v))
            match_rate = matches / n
            if match_rate > 0:
                votes[ftype] = votes.get(ftype, 0.0) + match_rate * weight

        # Name-based bonuses
        name_lower = name.lower()
        bonuses: list[tuple[FieldType, float, str]] = [
            (FieldType.PAN,            1.5, "pan|card_number|account_num"),
            (FieldType.EXPIRY,         1.2, "expiry|exp_date|expiration"),
            (FieldType.CVV,            1.5, "cvv|cvc|cvv2|security_code"),
            (FieldType.ROUTING_NUMBER, 1.3, "routing|aba|transit"),
            (FieldType.ACCOUNT_NUMBER, 1.0, "account|acct"),
            (FieldType.AMOUNT,         1.2, "amount|amt|total|price|fee"),
            (FieldType.DATE,           1.3, "date|dob|expiry|settle"),
            (FieldType.SEQUENCE,       1.2, "seq|sequence|counter|number|num"),
        ]
        for ftype, bonus, keywords in bonuses:
            if any(kw in name_lower for kw in keywords.split("|")):
                votes[ftype] = votes.get(ftype, 0.0) + bonus

        # PAN: Luhn check bonus
        if FieldType.PAN in votes:
            luhn_pass = sum(1 for v in values if self._luhn_check(v))
            if luhn_pass / n > 0.8:
                votes[FieldType.PAN] += 2.0

        # ENUM: if cardinality is very low
        unique = set(values)
        if 2 <= len(unique) <= 20 and len(unique) / n < 0.3:
            votes[FieldType.ENUM] = votes.get(FieldType.ENUM, 0.0) + 1.5

        if not votes:
            return FieldType.STRING, 0.0

        best_type = max(votes, key=lambda t: votes[t])
        total_weight = sum(votes.values())
        confidence = votes[best_type] / total_weight if total_weight > 0 else 0.0
        return best_type, min(confidence, 1.0)

    def _derive_constraints(
        self, ftype: FieldType, values: list[str], min_len: int, max_len: int
    ) -> FieldConstraints:
        c = FieldConstraints(min_length=min_len or None, max_length=max_len or None)

        if ftype == FieldType.ENUM:
            c.allowed_values = list({v for v in values if v})
        elif ftype in (FieldType.NUMERIC, FieldType.AMOUNT):
            nums = []
            for v in values:
                try:
                    nums.append(float(v))
                except ValueError:
                    pass
            if nums:
                c.min_value = min(nums)
                c.max_value = max(nums)
        elif ftype == FieldType.PAN:
            c.checksum_algorithm = "luhn"
        elif ftype == FieldType.ROUTING_NUMBER:
            c.checksum_algorithm = "mod10"

        return c

    @staticmethod
    def _luhn_check(pan: str) -> bool:
        """Standard Luhn algorithm check."""
        digits = [int(d) for d in pan if d.isdigit()]
        if len(digits) < 13:
            return False
        digits.reverse()
        total = 0
        for i, d in enumerate(digits):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0

    # ── Header / trailer detection ────────────────────────────────────────────

    def _detect_header_trailer(
        self,
        lines: list[str],
        detail_rec: RecordDef,
        delimiter: str,
        is_fixed: bool,
    ) -> tuple[list[RecordDef], FileStructure]:
        """
        Heuristic: if the first and/or last line differs significantly from
        the median line length, treat it as a header/trailer record.
        """
        lengths = [len(ln) for ln in lines if ln]
        if not lengths:
            return [detail_rec], FileStructure(format="delimited" if not is_fixed else "fixed_width",
                                               detail_records=["RECORD"])
        median_len = statistics.median(lengths)
        records: list[RecordDef] = []
        header_ids: list[str] = []
        trailer_ids: list[str] = []

        first_len = len(lines[0]) if lines else 0
        last_len = len(lines[-1]) if lines else 0

        if first_len and abs(first_len - median_len) / (median_len + 1) > 0.25:
            hdr = RecordDef(
                record_type_id="HEADER", name="File Header",
                category=RecordCategory.HEADER, ordering=0,
                repeatable=False, min_occurrences=1, max_occurrences=1,
            )
            records.append(hdr)
            header_ids.append("HEADER")

        records.append(detail_rec)

        if len(lines) > 2 and last_len and abs(last_len - median_len) / (median_len + 1) > 0.3:
            trl = RecordDef(
                record_type_id="TRAILER", name="File Trailer",
                category=RecordCategory.TRAILER, ordering=99,
                repeatable=False, min_occurrences=1, max_occurrences=1,
            )
            records.append(trl)
            trailer_ids.append("TRAILER")

        fs = FileStructure(
            format="fixed_width" if is_fixed else "delimited",
            file_extension=".txt",
            header_records=header_ids,
            detail_records=["RECORD"],
            trailer_records=trailer_ids,
        )
        return records, fs
