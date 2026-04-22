"""
Audit Log
==========
Thread-safe, append-only audit trail for all spec training,
validation, and generation events. Stored as newline-delimited JSON (NDJSON).

Usage:
    from core.audit_log import AuditLog
    log = AuditLog()
    log.record_training("NACHA", "user@corp.com", field_count=94)
    log.record_validation("NACHA", "payroll.ach", is_valid=True, score=99.2)
    log.record_generation("NACHA", rows=10)
    entries = log.query(action="VALIDATE", limit=50)
"""

import os
import json
import threading
from datetime import datetime, timezone
from typing import Literal

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH  = os.path.join(BASE_DIR, "models", "audit.ndjson")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

Action = Literal["TRAIN", "VALIDATE", "GENERATE", "DB_CONNECT", "DB_QUERY", "DELETE"]


class AuditLog:
    """Append-only NDJSON audit log with thread-safe writes."""

    _lock = threading.Lock()

    def __init__(self, path: str = LOG_PATH):
        self.path = path

    # ── Writers ──────────────────────────────────────────────────────
    def record_training(
        self,
        spec_name: str,
        user: str = "system",
        field_count: int = 0,
        format_type: str = "",
        source_file: str = "",
    ) -> dict:
        return self._write({
            "action":       "TRAIN",
            "spec":         spec_name,
            "user":         user,
            "field_count":  field_count,
            "format_type":  format_type,
            "source_file":  source_file,
        })

    def record_validation(
        self,
        spec_name: str,
        filename: str,
        is_valid: bool,
        score: float,
        records: int = 0,
        errors: int = 0,
        user: str = "system",
    ) -> dict:
        return self._write({
            "action":    "VALIDATE",
            "spec":      spec_name,
            "filename":  filename,
            "is_valid":  is_valid,
            "score":     round(score, 2),
            "records":   records,
            "errors":    errors,
            "user":      user,
        })

    def record_generation(
        self,
        spec_name: str,
        rows: int,
        seed: int | None = None,
        from_db: bool = False,
        user: str = "system",
    ) -> dict:
        return self._write({
            "action":   "GENERATE",
            "spec":     spec_name,
            "rows":     rows,
            "seed":     seed,
            "from_db":  from_db,
            "user":     user,
        })

    def record_db_connect(
        self,
        db_type: str,
        masked_conn: str,
        success: bool,
        user: str = "system",
    ) -> dict:
        return self._write({
            "action":   "DB_CONNECT",
            "db_type":  db_type,
            "conn":     masked_conn,
            "success":  success,
            "user":     user,
        })

    def record_db_query(
        self,
        query_preview: str,
        rows_returned: int,
        user: str = "system",
    ) -> dict:
        return self._write({
            "action":         "DB_QUERY",
            "query_preview":  query_preview[:120],
            "rows_returned":  rows_returned,
            "user":           user,
        })

    def record_delete(self, spec_name: str, user: str = "system") -> dict:
        return self._write({
            "action": "DELETE",
            "spec":   spec_name,
            "user":   user,
        })

    # ── Reader ───────────────────────────────────────────────────────
    def query(
        self,
        action: str | None = None,
        spec: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return most-recent N entries, optionally filtered."""
        entries = []
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if action and entry.get("action") != action:
                    continue
                if spec and entry.get("spec") != spec:
                    continue
                entries.append(entry)
        return entries[-limit:][::-1]   # newest first

    def stats(self) -> dict:
        """Aggregate stats across all log entries."""
        all_entries = self.query(limit=100_000)
        by_action: dict[str, int] = {}
        for e in all_entries:
            a = e.get("action", "UNKNOWN")
            by_action[a] = by_action.get(a, 0) + 1

        val_entries = [e for e in all_entries if e.get("action") == "VALIDATE"]
        avg_score   = (
            sum(e.get("score", 0) for e in val_entries) / len(val_entries)
            if val_entries else 0
        )

        return {
            "total_events":     len(all_entries),
            "by_action":        by_action,
            "validations_run":  by_action.get("VALIDATE", 0),
            "files_generated":  by_action.get("GENERATE", 0),
            "specs_trained":    by_action.get("TRAIN", 0),
            "avg_valid_score":  round(avg_score, 2),
        }

    def clear(self) -> int:
        """Clear log. Returns number of entries deleted."""
        count = len(self.query(limit=1_000_000))
        with self._lock:
            with open(self.path, "w") as fh:
                fh.write("")
        return count

    # ── Internal ─────────────────────────────────────────────────────
    def _write(self, payload: dict) -> dict:
        entry = {
            "ts":   datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
        return entry
