"""
Application configuration using pydantic-settings.
All settings can be overridden via environment variables.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "SpecGenAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # File storage paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    SPEC_DIR: Path = BASE_DIR / "specs"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    # Upload constraints
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_SPEC_TYPES: list[str] = ["vcf", "ach", "json", "sample"]

    # Generation defaults
    DEFAULT_RECORD_COUNT: int = 10
    MAX_RECORD_COUNT: int = 10000

    # Learner settings
    MIN_SAMPLE_ROWS_FOR_INFERENCE: int = 3
    FIELD_INFERENCE_CONFIDENCE_THRESHOLD: float = 0.70

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        for d in [self.UPLOAD_DIR, self.SPEC_DIR, self.OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
