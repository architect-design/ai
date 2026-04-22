"""
Pattern Detector
================
Discovers repeating structural patterns within a file:
  - Batch/segment boundaries
  - Record ordering sequences
  - Nested repetition (e.g., entry groups within batches)
  - Field co-occurrence and mutual exclusion patterns

All logic is pure Python (no external ML).
"""
from __future__ import annotations

import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any

from app.models.schema import SchemaModel, RecordDef, RecordCategory


@dataclass
class PatternResult:
    """Summary of patterns found in a file."""
    ordering_sequence: list[str]          # Detected record type ordering
    batch_pattern: list[str] | None       # e.g. ["5", "6", "6", "8"]
    repeating_types: set[str]             # Record types that repeat
    transition_matrix: dict[str, dict[str, int]]  # type→type transition counts
    anomalies: list[str]                  # Unexpected patterns


class PatternDetector:
    """
    Analyses a sequence of record-type identifiers (e.g., extracted from
    the first character of each line in a native ACH file) and discovers
    structural patterns.
    """

    def detect_from_file(self, content: str, model: SchemaModel) -> PatternResult:
        """Extract record-type sequence from file content and run detection."""
        sequence = self._extract_sequence(content, model)
        return self._analyse_sequence(sequence, model)

    def detect_from_sequence(self, sequence: list[str], model: SchemaModel) -> PatternResult:
        return self._analyse_sequence(sequence, model)

    # ── Sequence extraction ───────────────────────────────────────────────────

    def _extract_sequence(self, content: str, model: SchemaModel) -> list[str]:
        """
        For fixed-width formats: use the first character/characters of each line.
        For tagged formats: use field values that look like record type codes.
        For JSON: not applicable.
        """
        known_types = {r.record_type_id for r in model.records}
        lines = [ln for ln in content.splitlines() if ln.strip()]
        sequence: list[str] = []

        for line in lines:
            # Try single-char record type (ACH-style)
            if line[0] in known_types:
                sequence.append(line[0])
                continue
            # Try two-char code
            if len(line) >= 2 and line[:2] in known_types:
                sequence.append(line[:2])
                continue
            # Try common prefixes like "HDR", "DTL", "TRL"
            for t in known_types:
                if line.upper().startswith(t):
                    sequence.append(t)
                    break

        return sequence

    # ── Sequence analysis ─────────────────────────────────────────────────────

    def _analyse_sequence(self, sequence: list[str], model: SchemaModel) -> PatternResult:
        if not sequence:
            return PatternResult([], None, set(), {}, ["Empty sequence"])

        # Count type frequencies
        freq = Counter(sequence)
        repeating = {t for t, c in freq.items() if c > 1}

        # Build transition matrix
        transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(len(sequence) - 1):
            transitions[sequence[i]][sequence[i + 1]] += 1

        # Infer ordering from most-common transition path
        ordering_sequence = self._infer_ordering(sequence, transitions)

        # Detect batch pattern
        batch_pattern = self._detect_batch_pattern(sequence, model)

        # Anomalies
        anomalies = self._detect_anomalies(sequence, model, ordering_sequence)

        # Update model ordering based on detected sequence
        self._update_model_ordering(model, ordering_sequence)

        return PatternResult(
            ordering_sequence=ordering_sequence,
            batch_pattern=batch_pattern,
            repeating_types=repeating,
            transition_matrix={k: dict(v) for k, v in transitions.items()},
            anomalies=anomalies,
        )

    def _infer_ordering(
        self,
        sequence: list[str],
        transitions: dict[str, dict[str, int]],
    ) -> list[str]:
        """
        Perform a greedy walk through the transition matrix starting from
        the most common first-position type, following highest-probability edges.
        """
        if not sequence:
            return []

        # Find likely starting nodes (appear at position 0 or near the start)
        start_counts = Counter(sequence[:max(1, len(sequence)//10)])
        start_node = start_counts.most_common(1)[0][0]

        visited: set[str] = set()
        path: list[str] = []
        node = start_node

        while node and node not in visited:
            path.append(node)
            visited.add(node)
            next_nodes = transitions.get(node, {})
            if not next_nodes:
                break
            # Choose the most frequent unvisited successor
            candidates = [(t, c) for t, c in next_nodes.items() if t not in visited]
            if not candidates:
                break
            node = max(candidates, key=lambda x: x[1])[0]

        # Append any remaining types not in the path (sorted by frequency)
        all_types = set(sequence)
        remaining = sorted(all_types - set(path), key=lambda t: -sequence.count(t))
        path.extend(remaining)

        return path

    def _detect_batch_pattern(
        self, sequence: list[str], model: SchemaModel
    ) -> list[str] | None:
        """
        Look for batch header + detail(s) + batch trailer sub-sequences.
        """
        bh = model.file_structure.batch_header_record
        bt = model.file_structure.batch_trailer_record
        if not bh or not bt:
            return None
        if bh not in sequence or bt not in sequence:
            return None

        # Find the first occurrence of bh and the next bt
        try:
            bh_idx = sequence.index(bh)
            bt_idx = sequence.index(bt, bh_idx)
            return list(dict.fromkeys(sequence[bh_idx:bt_idx + 1]))
        except ValueError:
            return None

    def _detect_anomalies(
        self,
        sequence: list[str],
        model: SchemaModel,
        expected_ordering: list[str],
    ) -> list[str]:
        anomalies: list[str] = []

        # Check for records that appear out of the expected order
        pos_map: dict[str, int] = {t: i for i, t in enumerate(expected_ordering)}
        prev_pos = -1
        for rt in sequence:
            cur_pos = pos_map.get(rt, -1)
            if cur_pos != -1 and cur_pos < prev_pos:
                anomalies.append(f"Record type '{rt}' appears out of order in the file.")
            if cur_pos != -1:
                prev_pos = cur_pos

        # Check for header appearing more than once
        hdr_ids = set(model.file_structure.header_records)
        for rt in hdr_ids:
            count = sequence.count(rt)
            if count > 1:
                anomalies.append(f"Header record '{rt}' appears {count} times (expected 1).")

        return anomalies

    def _update_model_ordering(self, model: SchemaModel, ordering: list[str]):
        """Back-propagate detected ordering into model.records[*].ordering."""
        for rec in model.records:
            if rec.record_type_id in ordering:
                rec.ordering = ordering.index(rec.record_type_id) * 10
