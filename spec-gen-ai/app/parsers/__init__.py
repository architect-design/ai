"""Parser package — exports the parser factory."""
from app.parsers.base_parser import BaseParser
from app.parsers.ach_parser import ACHParser
from app.parsers.vcf_parser import VCFParser
from app.parsers.json_schema_parser import JSONSchemaParser
from app.parsers.sample_data_parser import SampleDataParser
from app.core.exceptions import UnsupportedSpecTypeError


_PARSER_REGISTRY: dict[str, type[BaseParser]] = {
    "ach":    ACHParser,
    "vcf":    VCFParser,
    "json":   JSONSchemaParser,
    "sample": SampleDataParser,
}


def get_parser(spec_type: str) -> BaseParser:
    """Factory function — returns the appropriate parser for a spec type."""
    cls = _PARSER_REGISTRY.get(spec_type.lower())
    if cls is None:
        raise UnsupportedSpecTypeError(
            f"No parser registered for spec_type='{spec_type}'. "
            f"Supported: {list(_PARSER_REGISTRY)}"
        )
    return cls()
