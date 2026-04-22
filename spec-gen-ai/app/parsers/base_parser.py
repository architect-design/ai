"""
Base Parser
===========
All format-specific parsers inherit from BaseParser.
A parser converts raw file bytes into a *partial* SchemaModel —
only the structural information the parser can extract directly.
The Learner/Trainer layer fills in inferred statistics afterward.
"""
from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

from app.models.schema import SchemaModel, SpecType

logger = logging.getLogger(__name__)


class BaseParser(abc.ABC):
    """Abstract base class for all specification parsers."""

    spec_type: SpecType  # subclasses must set this

    def __init__(self):
        self._errors: list[str] = []
        self._warnings: list[str] = []

    # ── Public interface ─────────────────────────────────────────────────────

    def parse_file(self, path: Path) -> SchemaModel:
        """Parse a file on disk and return a partial SchemaModel."""
        logger.info("[%s] Parsing file: %s", self.__class__.__name__, path)
        content = self._read_file(path)
        return self.parse_content(content, source_name=path.name)

    def parse_content(self, content: str, source_name: str = "") -> SchemaModel:
        """Parse raw string content and return a partial SchemaModel."""
        self._errors.clear()
        self._warnings.clear()
        model = self._do_parse(content, source_name)
        model.spec_type = self.spec_type
        if source_name:
            model.source_files.append(source_name)
        if self._warnings:
            model.inference_stats["parser_warnings"] = self._warnings
        logger.info(
            "[%s] Parsed %d record types, %d errors, %d warnings",
            self.__class__.__name__,
            len(model.records),
            len(self._errors),
            len(self._warnings),
        )
        return model

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    # ── Subclass contract ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _do_parse(self, content: str, source_name: str) -> SchemaModel:
        """Subclasses implement the actual parsing logic here."""

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IOError(f"Cannot read file {path}: {exc}") from exc

    def _warn(self, msg: str):
        self._warnings.append(msg)
        logger.warning("[%s] %s", self.__class__.__name__, msg)

    def _error(self, msg: str):
        self._errors.append(msg)
        logger.error("[%s] %s", self.__class__.__name__, msg)

    @staticmethod
    def _normalise_name(raw: str) -> str:
        """Convert arbitrary column headers to snake_case field names."""
        import re
        raw = raw.strip().lower()
        raw = re.sub(r"[\s\-/]+", "_", raw)
        raw = re.sub(r"[^a-z0-9_]", "", raw)
        return raw or "unknown_field"
