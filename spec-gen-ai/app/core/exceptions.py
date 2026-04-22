"""
Custom exception hierarchy.
All domain exceptions derive from SpecGenAIError so callers can
catch either the base or specific subtypes.
"""


class SpecGenAIError(Exception):
    """Base class for all application exceptions."""
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


# ── Parser layer ────────────────────────────────────────────────────────────

class ParseError(SpecGenAIError):
    """Raised when a specification file cannot be parsed."""


class UnsupportedSpecTypeError(SpecGenAIError):
    """Raised when the spec type is not supported."""


# ── Learner / Trainer layer ──────────────────────────────────────────────────

class TrainingError(SpecGenAIError):
    """Raised when the learner cannot build a valid schema model."""


class InsufficientSampleDataError(TrainingError):
    """Raised when sample data is too sparse for inference."""


# ── Rule Engine ──────────────────────────────────────────────────────────────

class RuleViolationError(SpecGenAIError):
    """Raised when a generated record violates a business rule."""


class CircularDependencyError(SpecGenAIError):
    """Raised when field dependencies form a cycle."""


# ── Validation ───────────────────────────────────────────────────────────────

class ValidationError(SpecGenAIError):
    """Raised when generated or uploaded data fails schema validation."""
    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors: list[str] = errors or []


# ── Generator ────────────────────────────────────────────────────────────────

class GenerationError(SpecGenAIError):
    """Raised when the generator cannot produce a valid file."""


# ── Storage ──────────────────────────────────────────────────────────────────

class SpecNotFoundError(SpecGenAIError):
    """Raised when a requested specification model does not exist."""


class StorageError(SpecGenAIError):
    """Raised for file-system or serialisation failures."""
