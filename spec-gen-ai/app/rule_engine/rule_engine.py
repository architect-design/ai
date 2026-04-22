"""
Rule Engine
===========
Manages, validates, and applies business rules for data generation.

Rules are stored as:
  - FieldRule: applies to a single field value
  - RecordRule: applies to a complete record dict
  - FileRule:   applies to the complete generated file

The engine is also responsible for resolving the field generation order
based on the dependency graph.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.schema import (
    SchemaModel, RecordDef, FieldDef, FieldType, FieldDependency,
)
from app.core.exceptions import RuleViolationError, CircularDependencyError

logger = logging.getLogger(__name__)


# ── Rule dataclasses ──────────────────────────────────────────────────────────

@dataclass
class FieldRule:
    """A validation/transformation rule for a single field."""
    name: str
    description: str
    field_name: str
    record_type_id: str
    validate: Callable[[Any, dict[str, Any]], bool] | None = None
    transform: Callable[[Any, dict[str, Any]], Any] | None = None

    def apply_validate(self, value: Any, context: dict[str, Any]) -> bool:
        if self.validate is None:
            return True
        try:
            return self.validate(value, context)
        except Exception as exc:
            logger.warning("Rule '%s' validation error: %s", self.name, exc)
            return False

    def apply_transform(self, value: Any, context: dict[str, Any]) -> Any:
        if self.transform is None:
            return value
        try:
            return self.transform(value, context)
        except Exception as exc:
            logger.warning("Rule '%s' transform error: %s", self.name, exc)
            return value


@dataclass
class RecordRule:
    """A cross-field rule applied to a complete record."""
    name: str
    description: str
    record_type_id: str
    check: Callable[[dict[str, Any]], tuple[bool, str]] = field(
        default=lambda _: (True, "")
    )


@dataclass
class FileRule:
    """An aggregate rule applied to the full generated file."""
    name: str
    description: str
    check: Callable[[list[dict[str, Any]]], tuple[bool, str]] = field(
        default=lambda _: (True, "")
    )


# ── Rule Engine ───────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Loads rules from a SchemaModel and applies them during generation
    and validation.
    """

    def __init__(self):
        self._field_rules: dict[str, list[FieldRule]] = defaultdict(list)
        self._record_rules: dict[str, list[RecordRule]] = defaultdict(list)
        self._file_rules: list[FileRule] = []

    # ── Bootstrap from schema model ───────────────────────────────────────────

    def load_from_model(self, model: SchemaModel):
        """Register built-in and schema-derived rules."""
        self._field_rules.clear()
        self._record_rules.clear()
        self._file_rules.clear()

        for rec in model.records:
            for field_def in rec.fields:
                key = f"{rec.record_type_id}.{field_def.name}"
                self._register_constraint_rules(key, rec.record_type_id, field_def)

            self._register_record_rules(rec)

        self._register_file_rules(model)
        logger.info(
            "RuleEngine loaded: %d field keys, %d file rules",
            len(self._field_rules),
            len(self._file_rules),
        )

    def _register_constraint_rules(
        self, key: str, rt_id: str, fd: FieldDef
    ):
        c = fd.constraints

        if c.min_length is not None:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.min_length",
                description=f"Min length {c.min_length}",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx, ml=c.min_length: len(str(v)) >= ml,
            ))

        if c.max_length is not None:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.max_length",
                description=f"Max length {c.max_length}",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx, ml=c.max_length: len(str(v)) <= ml,
            ))

        if c.allowed_values:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.allowed_values",
                description=f"Allowed: {c.allowed_values}",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx, av=c.allowed_values: str(v) in av or not av,
            ))

        if c.pattern:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.pattern",
                description=f"Pattern: {c.pattern}",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx, p=c.pattern: bool(re.match(p, str(v))),
            ))

        if c.checksum_algorithm == "luhn":
            self._field_rules[key].append(FieldRule(
                name=f"{key}.luhn",
                description="Luhn checksum",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx: _luhn_check(str(v)),
            ))

        if c.checksum_algorithm in ("aba", "mod10") and fd.field_type == FieldType.ROUTING_NUMBER:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.aba_check",
                description="ABA routing check",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx: _aba_check(str(v)),
            ))

        if c.min_value is not None:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.min_value",
                description=f"Min value {c.min_value}",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx, mv=c.min_value: (
                    float(str(v).replace(",", "")) >= mv
                    if _is_numeric(v) else True
                ),
            ))

        if c.max_value is not None:
            self._field_rules[key].append(FieldRule(
                name=f"{key}.max_value",
                description=f"Max value {c.max_value}",
                field_name=fd.name,
                record_type_id=rt_id,
                validate=lambda v, _ctx, mv=c.max_value: (
                    float(str(v).replace(",", "")) <= mv
                    if _is_numeric(v) else True
                ),
            ))

    def _register_record_rules(self, rec: RecordDef):
        """Register cross-field rules for a record."""
        # Fixed-width length check
        if rec.fixed_width and rec.record_length:
            expected_len = rec.record_length
            self._record_rules[rec.record_type_id].append(RecordRule(
                name=f"{rec.record_type_id}.fixed_width_length",
                description=f"Record must be exactly {expected_len} chars",
                record_type_id=rec.record_type_id,
                check=lambda row, el=expected_len: (
                    (True, "") if row.get("__raw_length__", el) == el
                    else (False, f"Expected {el} chars, got {row.get('__raw_length__', '?')}")
                ),
            ))

    def _register_file_rules(self, model: SchemaModel):
        """Register aggregate file-level rules."""
        for rule_expr in model.global_rules:
            self._file_rules.append(FileRule(
                name=f"global.{rule_expr[:40]}",
                description=rule_expr,
                check=lambda rows: (True, ""),  # evaluated separately
            ))

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_field(self, rt_id: str, field_name: str, value: Any, context: dict) -> list[str]:
        """Returns list of error messages (empty if valid)."""
        key = f"{rt_id}.{field_name}"
        errors: list[str] = []
        for rule in self._field_rules.get(key, []):
            if not rule.apply_validate(value, context):
                errors.append(f"[{rule.name}] {rule.description} — got: {repr(value)}")
        return errors

    def validate_record(self, rt_id: str, row: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for rule in self._record_rules.get(rt_id, []):
            ok, msg = rule.check(row)
            if not ok:
                errors.append(f"[{rule.name}] {msg}")
        return errors

    def validate_file(self, rows: list[dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        for rule in self._file_rules:
            ok, msg = rule.check(rows)
            if not ok:
                errors.append(f"[{rule.name}] {msg}")
        return errors

    # ── Dependency resolution (topological sort) ──────────────────────────────

    def resolve_field_order(self, rec: RecordDef) -> list[FieldDef]:
        """
        Return fields in dependency-respecting generation order.
        Uses Kahn's algorithm for topological sort.
        """
        name_to_field = {f.name: f for f in rec.fields}
        in_degree: dict[str, int] = {f.name: 0 for f in rec.fields}
        adjacency: dict[str, list[str]] = defaultdict(list)

        for f in rec.fields:
            for dep in f.dependencies:
                dep_name = dep.depends_on
                if dep_name in name_to_field:
                    in_degree[f.name] += 1
                    adjacency[dep_name].append(f.name)

        # Fields with no dependencies come first
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[FieldDef] = []
        visited: set[str] = set()

        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            if name in name_to_field:
                order.append(name_to_field[name])
            for successor in adjacency.get(name, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(order) != len(rec.fields):
            # Cycle detected — fall back to original order
            remaining = [f for f in rec.fields if f.name not in visited]
            order.extend(remaining)

        return order


# ── Standalone checksum helpers ───────────────────────────────────────────────

def _luhn_check(pan: str) -> bool:
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
    if len(routing) != 9 or not routing.isdigit():
        return False
    d = [int(c) for c in routing]
    total = 3*(d[0]+d[3]+d[6]) + 7*(d[1]+d[4]+d[7]) + (d[2]+d[5]+d[8])
    return total % 10 == 0


def _is_numeric(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False
