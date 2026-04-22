"""
Storage Service
===============
Handles persistence of:
  - Uploaded raw files  (uploads/ directory)
  - Trained SchemaModels (specs/ directory as JSON)
  - Generated output files (outputs/ directory)
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.schema import SchemaModel
from app.core.config import settings
from app.core.exceptions import SpecNotFoundError, StorageError

logger = logging.getLogger(__name__)


class StorageService:

    def __init__(self):
        self._upload_dir = settings.UPLOAD_DIR
        self._spec_dir = settings.SPEC_DIR
        self._output_dir = settings.OUTPUT_DIR

    # ── Upload management ─────────────────────────────────────────────────────

    def save_upload(self, filename: str, content: bytes) -> tuple[str, Path]:
        """
        Persist an uploaded file.
        Returns (upload_id, path).
        """
        upload_id = str(uuid.uuid4())
        safe_name = self._sanitise_filename(filename)
        dest = self._upload_dir / upload_id / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(content)
        except OSError as exc:
            raise StorageError(f"Cannot write upload: {exc}") from exc
        logger.info("Saved upload %s → %s", upload_id, dest)
        return upload_id, dest

    def get_upload_path(self, upload_id: str) -> Path:
        folder = self._upload_dir / upload_id
        if not folder.exists():
            raise SpecNotFoundError(f"Upload '{upload_id}' not found")
        files = list(folder.iterdir())
        if not files:
            raise StorageError(f"Upload folder '{upload_id}' is empty")
        return files[0]

    def delete_upload(self, upload_id: str):
        folder = self._upload_dir / upload_id
        if folder.exists():
            shutil.rmtree(folder)

    # ── SchemaModel persistence ───────────────────────────────────────────────

    def save_spec(self, model: SchemaModel) -> Path:
        """Serialise and persist a SchemaModel. Returns the saved path."""
        path = self._spec_dir / f"{model.spec_id}.json"
        try:
            path.write_text(model.to_json(), encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"Cannot save spec '{model.spec_id}': {exc}") from exc
        logger.info("Saved spec %s (%s)", model.spec_id, model.spec_name)
        return path

    def load_spec(self, spec_id: str) -> SchemaModel:
        path = self._spec_dir / f"{spec_id}.json"
        if not path.exists():
            raise SpecNotFoundError(f"Specification '{spec_id}' not found")
        try:
            raw = path.read_text(encoding="utf-8")
            return SchemaModel.from_json(raw)
        except (json.JSONDecodeError, KeyError) as exc:
            raise StorageError(f"Corrupt spec file '{spec_id}': {exc}") from exc

    def list_specs(self) -> list[SchemaModel]:
        models: list[SchemaModel] = []
        for path in sorted(self._spec_dir.glob("*.json")):
            try:
                models.append(SchemaModel.from_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Skipping corrupt spec file %s: %s", path.name, exc)
        return models

    def delete_spec(self, spec_id: str):
        path = self._spec_dir / f"{spec_id}.json"
        if path.exists():
            path.unlink()
            logger.info("Deleted spec %s", spec_id)

    def spec_exists(self, spec_id: str) -> bool:
        return (self._spec_dir / f"{spec_id}.json").exists()

    def find_spec_by_name(self, name: str) -> SchemaModel | None:
        for model in self.list_specs():
            if model.spec_name.lower() == name.lower():
                return model
        return None

    # ── Output file management ────────────────────────────────────────────────

    def save_output(
        self,
        generation_id: str,
        content: str,
        extension: str = ".txt",
        encoding: str = "ascii",
    ) -> Path:
        dest = self._output_dir / f"{generation_id}{extension}"
        try:
            dest.write_text(content, encoding=encoding, errors="replace")
        except OSError as exc:
            raise StorageError(f"Cannot save output '{generation_id}': {exc}") from exc
        logger.info("Saved output %s (%d bytes)", dest.name, dest.stat().st_size)
        return dest

    def get_output_path(self, generation_id: str) -> Path:
        candidates = list(self._output_dir.glob(f"{generation_id}*"))
        if not candidates:
            raise SpecNotFoundError(f"Output '{generation_id}' not found")
        return candidates[0]

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitise_filename(filename: str) -> str:
        """Strip path components and dangerous characters."""
        name = Path(filename).name
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", "."))
        return safe or "upload.bin"

    def storage_stats(self) -> dict[str, Any]:
        def dir_size(p: Path) -> int:
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

        return {
            "uploads_count": sum(1 for _ in self._upload_dir.glob("*") if _.is_dir()),
            "specs_count": sum(1 for _ in self._spec_dir.glob("*.json")),
            "outputs_count": sum(1 for _ in self._output_dir.glob("*") if _.is_file()),
            "uploads_bytes": dir_size(self._upload_dir),
            "specs_bytes": dir_size(self._spec_dir),
            "outputs_bytes": dir_size(self._output_dir),
        }
