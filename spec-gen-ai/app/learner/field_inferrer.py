"""
Field Type Inferrer
====================
Standalone module that enriches a *parsed* SchemaModel by running
statistical inference over the schema's sample_values.

This is the "learning" phase for fields whose types were only weakly
inferred by the parser.  Uses no external ML libraries — all logic
is pure Python with optional numpy for statistical functions.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from app.models.schema import FieldDef, FieldType, FieldConstraints
from app.core.config import settings


class FieldInferrer:
    """
    Enriches FieldDef instances using sample_values.
    Called by the Learner after initial parsing.
    """

    CONFIDENCE_THRESHOLD: float = 0.45  # multi-strategy voting needs lower threshold

    # ── Public API ───────────────────────────────────────────────────────────

    def enrich(self, field: FieldDef) -> FieldDef:
        """
        In-place enrichment of a FieldDef.
        Returns the same object (modified) for fluent chaining.
        """
        if not field.sample_values:
            return field

        values = [v.strip() for v in field.sample_values if v and v.strip()]
        if not values:
            return field

        # Run all inference passes
        inferred_type, confidence = self._infer_type(values, field.name)

        # Update if inference produces any result better than default STRING
        if inferred_type != FieldType.STRING and confidence >= self.CONFIDENCE_THRESHOLD:
            field.field_type = inferred_type
            field.inferred_confidence = confidence
        elif confidence > field.inferred_confidence:
            field.field_type = inferred_type
            field.inferred_confidence = confidence

        # Always enrich constraints from sample
        self._enrich_constraints(field, values)
        self._infer_format_string(field, values)
        self._infer_default(field, values)

        return field

    # ── Core type inference ───────────────────────────────────────────────────

    def _infer_type(self, values: list[str], name: str) -> tuple[FieldType, float]:
        """Multi-strategy voting system. Returns (type, confidence in 0–1)."""
        n = len(values)
        votes: dict[FieldType, float] = {}

        # Strategy 1: Regex matching
        regex_votes = self._regex_vote(values)
        for ft, score in regex_votes.items():
            votes[ft] = votes.get(ft, 0.0) + score * 1.0

        # Strategy 2: Statistical distribution
        stat_votes = self._statistical_vote(values)
        for ft, score in stat_votes.items():
            votes[ft] = votes.get(ft, 0.0) + score * 0.8

        # Strategy 3: Name heuristics
        name_votes = self._name_vote(name)
        for ft, score in name_votes.items():
            votes[ft] = votes.get(ft, 0.0) + score * 0.6

        # Strategy 4: Checksum validation
        checksum_votes = self._checksum_vote(values)
        for ft, score in checksum_votes.items():
            votes[ft] = votes.get(ft, 0.0) + score * 1.5  # high weight for checksums

        if not votes:
            return FieldType.STRING, 0.0

        best = max(votes, key=lambda t: votes[t])
        total = sum(votes.values())
        confidence = votes[best] / total if total > 0 else 0.0
        return best, min(confidence, 1.0)

    def _regex_vote(self, values: list[str]) -> dict[FieldType, float]:
        patterns = [
            (FieldType.DATE,           r"^\d{8}$",                          3.0),  # YYYYMMDD strong
            (FieldType.DATE,           r"^\d{4}[-/]\d{2}[-/]\d{2}$",        3.5),  # ISO date
            (FieldType.DATE,           r"^\d{2}[-/]\d{2}[-/]\d{4}$",        2.5),
            (FieldType.DATETIME,       r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", 3.0),
            (FieldType.PAN,            r"^\d{13,19}$",                      1.2),
            (FieldType.EXPIRY,         r"^\d{2}/\d{2}$",                    2.5),
            (FieldType.EXPIRY,         r"^\d{4}$",                          0.4),  # very weak — 4-digit
            (FieldType.ROUTING_NUMBER, r"^\d{9}$",                          2.5),  # exactly 9 digits
            (FieldType.ACCOUNT_NUMBER, r"^\d{6,17}$",                       0.5),
            (FieldType.AMOUNT,         r"^\$?-?\d{1,12}[.,]\d{2}$",         3.5),  # decimal amount
            (FieldType.AMOUNT,         r"^\d{4,10}$",                       0.4),  # cents (very weak)
            (FieldType.NUMERIC,        r"^-?\d{1,5}$",                      0.8),  # short ints only
            (FieldType.BOOLEAN,        r"^(true|false|yes|no|0|1|y|n)$",    3.0),
            (FieldType.ALPHANUMERIC,   r"^[A-Za-z0-9 ._-]+$",               0.2),
        ]
        n = len(values)
        votes: dict[FieldType, float] = {}
        for ftype, pattern, weight in patterns:
            matches = sum(1 for v in values if re.fullmatch(pattern, v, re.IGNORECASE))
            rate = matches / n
            if rate > 0.5:
                votes[ftype] = votes.get(ftype, 0.0) + rate * weight

        # Exclusion rules: if a dominant specific type wins, suppress overlapping generics
        if FieldType.DATE in votes and votes[FieldType.DATE] >= 2.5:
            votes.pop(FieldType.NUMERIC, None)
            votes.pop(FieldType.AMOUNT, None)
        if FieldType.ROUTING_NUMBER in votes and votes[FieldType.ROUTING_NUMBER] >= 2.0:
            votes.pop(FieldType.ACCOUNT_NUMBER, None)
            votes.pop(FieldType.NUMERIC, None)
        if FieldType.PAN in votes and votes[FieldType.PAN] >= 1.0:
            votes.pop(FieldType.ACCOUNT_NUMBER, None)
            votes.pop(FieldType.NUMERIC, None)

        return votes

    def _statistical_vote(self, values: list[str]) -> dict[FieldType, float]:
        votes: dict[FieldType, float] = {}
        n = len(values)

        # Numeric statistics
        numerics = []
        for v in values:
            try:
                numerics.append(float(v.replace(",", "").replace("$", "")))
            except ValueError:
                pass
        numeric_rate = len(numerics) / n

        if numeric_rate > 0.9:
            if all(v.replace(",", "").replace(".", "").replace("-", "").isdigit() for v in values if v):
                votes[FieldType.NUMERIC] = votes.get(FieldType.NUMERIC, 0.0) + numeric_rate

        # Cardinality-based enum detection
        unique_vals = set(values)
        cardinality = len(unique_vals)
        if 2 <= cardinality <= 25 and cardinality / n < 0.25:
            votes[FieldType.ENUM] = votes.get(FieldType.ENUM, 0.0) + 1.2

        # Length consistency → fixed-width field
        lengths = [len(v) for v in values]
        if lengths and len(set(lengths)) == 1:
            # All same length — could be a fixed-width field of special type
            fixed_len = lengths[0]
            if fixed_len == 9:
                votes[FieldType.ROUTING_NUMBER] = votes.get(FieldType.ROUTING_NUMBER, 0.0) + 0.5
            elif fixed_len in (13, 14, 15, 16, 17, 18, 19):
                votes[FieldType.PAN] = votes.get(FieldType.PAN, 0.0) + 0.5
            elif fixed_len == 4:
                votes[FieldType.EXPIRY] = votes.get(FieldType.EXPIRY, 0.0) + 0.3

        return votes

    def _name_vote(self, name: str) -> dict[FieldType, float]:
        name_lower = name.lower()
        rules: list[tuple[FieldType, float, list[str]]] = [
            (FieldType.PAN,            2.0, ["pan", "card_number", "card_num", "primary_account"]),
            (FieldType.EXPIRY,         2.0, ["expiry", "expiration", "exp_date", "exp_month", "exp_year"]),
            (FieldType.CVV,            3.0, ["cvv", "cvc", "cvv2", "security_code", "card_code"]),
            (FieldType.ROUTING_NUMBER, 2.0, ["routing", "aba", "transit_num", "routing_number"]),
            (FieldType.ACCOUNT_NUMBER, 1.5, ["account_number", "acct_num", "bank_account"]),
            (FieldType.AMOUNT,         1.5, ["amount", "amt", "total", "balance", "price", "fee", "charge"]),
            (FieldType.DATE,           1.5, ["date", "dob", "birth_date", "settle_date", "created"]),
            (FieldType.DATETIME,       1.8, ["timestamp", "datetime", "created_at", "updated_at"]),
            (FieldType.SEQUENCE,       1.5, ["seq", "sequence", "record_num", "line_number", "id", "trace"]),
            (FieldType.BOOLEAN,        1.5, ["is_", "has_", "flag", "indicator", "enabled", "active"]),
        ]
        votes: dict[FieldType, float] = {}
        for ftype, weight, keywords in rules:
            if any(kw in name_lower for kw in keywords):
                votes[ftype] = votes.get(ftype, 0.0) + weight
        return votes

    def _checksum_vote(self, values: list[str]) -> dict[FieldType, float]:
        votes: dict[FieldType, float] = {}
        if not values:
            return votes

        # Test Luhn
        all_digits = all(v.isdigit() for v in values)
        if all_digits and 13 <= len(values[0]) <= 19:
            luhn_pass = sum(1 for v in values if _luhn(v))
            rate = luhn_pass / len(values)
            if rate > 0.8:
                votes[FieldType.PAN] = votes.get(FieldType.PAN, 0.0) + rate * 2.0

        # Routing number check (ABA algorithm)
        if all_digits and len(values[0]) == 9:
            aba_pass = sum(1 for v in values if _aba_check(v))
            rate = aba_pass / len(values)
            if rate > 0.8:
                votes[FieldType.ROUTING_NUMBER] = votes.get(FieldType.ROUTING_NUMBER, 0.0) + rate * 2.0

        return votes

    # ── Constraint enrichment ─────────────────────────────────────────────────

    def _enrich_constraints(self, field: FieldDef, values: list[str]):
        c = field.constraints
        lengths = [len(v) for v in values]
        if lengths:
            c.min_length = min(lengths)
            c.max_length = max(lengths)

        if field.field_type in (FieldType.NUMERIC, FieldType.AMOUNT):
            nums = []
            for v in values:
                try:
                    nums.append(float(v.replace(",", "").replace("$", "")))
                except ValueError:
                    pass
            if nums:
                c.min_value = min(nums)
                c.max_value = max(nums)

        if field.field_type == FieldType.ENUM:
            c.allowed_values = sorted({v for v in values if v})

        if field.field_type == FieldType.PAN:
            c.checksum_algorithm = "luhn"
        elif field.field_type == FieldType.ROUTING_NUMBER:
            c.checksum_algorithm = "aba"

    # ── Format string inference ───────────────────────────────────────────────

    def _infer_format_string(self, field: FieldDef, values: list[str]):
        if field.field_type not in (FieldType.DATE, FieldType.DATETIME):
            return
        date_formats = [
            (r"^\d{4}-\d{2}-\d{2}$",              "YYYY-MM-DD"),
            (r"^\d{8}$",                           "YYYYMMDD"),
            (r"^\d{2}/\d{2}/\d{4}$",               "MM/DD/YYYY"),
            (r"^\d{2}-\d{2}-\d{4}$",               "MM-DD-YYYY"),
            (r"^\d{6}$",                           "YYMMDD"),
            (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}",   "YYYY-MM-DDTHH:MM"),
        ]
        for pattern, fmt in date_formats:
            if all(re.match(pattern, v) for v in values[:20] if v):
                field.format_string = fmt
                break

    # ── Default value inference ───────────────────────────────────────────────

    def _infer_default(self, field: FieldDef, values: list[str]):
        if field.field_type == FieldType.CONSTANT:
            # All values must be identical
            unique = set(values)
            if len(unique) == 1:
                field.default_value = unique.pop()
        elif field.field_type == FieldType.ENUM and field.constraints.allowed_values:
            # Most common value becomes the default
            counter = Counter(values)
            field.default_value = counter.most_common(1)[0][0]


# ── Standalone checksum helpers ───────────────────────────────────────────────

def _luhn(pan: str) -> bool:
    digits = [int(d) for d in pan if d.isdigit()]
    if len(digits) < 13:
        return False
    digits.reverse()
    total = sum(
        d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
        for i, d in enumerate(digits)
    )
    return total % 10 == 0


def _aba_check(routing: str) -> bool:
    """ABA routing number checksum (weighted sum mod 10)."""
    if len(routing) != 9 or not routing.isdigit():
        return False
    d = [int(c) for c in routing]
    total = (3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8]))
    return total % 10 == 0
