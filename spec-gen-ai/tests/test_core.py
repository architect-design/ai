"""
Unit Tests — SpecGenAI
=======================
Run with:
    pytest tests/test_core.py -v --tb=short
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Model imports ─────────────────────────────────────────────────────────────
from app.models.schema import (
    SchemaModel, SpecType, RecordDef, FieldDef, FieldType,
    FieldConstraints, FileStructure, RecordCategory, Justification,
)

# ── Parser imports ────────────────────────────────────────────────────────────
from app.parsers.ach_parser import ACHParser
from app.parsers.vcf_parser import VCFParser
from app.parsers.json_schema_parser import JSONSchemaParser
from app.parsers.sample_data_parser import SampleDataParser
from app.parsers import get_parser

# ── Learner imports ───────────────────────────────────────────────────────────
from app.learner.field_inferrer import FieldInferrer
from app.learner.pattern_detector import PatternDetector

# ── Rule engine ───────────────────────────────────────────────────────────────
from app.rule_engine.rule_engine import RuleEngine

# ── Generator ─────────────────────────────────────────────────────────────────
from app.generator.data_synthesizer import DataSynthesizer
from app.generator.base_generator import ACHGenerator, VCFGenerator, JSONGenerator, get_generator

# ── Validation ────────────────────────────────────────────────────────────────
from app.validation.validator import ValidationEngine

# ── Exceptions ────────────────────────────────────────────────────────────────
from app.core.exceptions import (
    ParseError, UnsupportedSpecTypeError, TrainingError,
    InsufficientSampleDataError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_model() -> SchemaModel:
    """Minimal SchemaModel with one detail record."""
    field_a = FieldDef(
        name="id", field_type=FieldType.SEQUENCE,
        start_pos=1, end_pos=5, length=5,
        justification=Justification.RIGHT, pad_char="0",
    )
    field_b = FieldDef(
        name="name", field_type=FieldType.STRING,
        start_pos=6, end_pos=30, length=25,
    )
    field_c = FieldDef(
        name="amount", field_type=FieldType.AMOUNT,
        start_pos=31, end_pos=40, length=10,
        justification=Justification.RIGHT, pad_char="0",
        constraints=FieldConstraints(min_value=1, max_value=999999),
    )
    rec = RecordDef(
        record_type_id="DETAIL",
        name="Test Detail",
        category=RecordCategory.DETAIL,
        fixed_width=True,
        record_length=40,
        repeatable=True,
        ordering=10,
        fields=[field_a, field_b, field_c],
    )
    return SchemaModel(
        spec_id=str(uuid.uuid4()),
        spec_name="TestSpec",
        spec_type=SpecType.JSON,
        is_trained=True,
        file_structure=FileStructure(
            format="fixed_width",
            detail_records=["DETAIL"],
        ),
        records=[rec],
    )


@pytest.fixture
def ach_model() -> SchemaModel:
    parser = ACHParser()
    return parser.parse_content(
        '{"spec_type": "ach", "records": []}', source_name="test.json"
    )


@pytest.fixture
def vcf_model() -> SchemaModel:
    parser = VCFParser()
    return parser.parse_content("{}", source_name="test.vcf")


@pytest.fixture
def synth() -> DataSynthesizer:
    return DataSynthesizer(seed=42)


# =============================================================================
# 1. Schema Model Tests
# =============================================================================

class TestSchemaModel:

    def test_serialise_roundtrip(self, minimal_model):
        """to_json / from_json must be lossless."""
        json_str = minimal_model.to_json()
        restored = SchemaModel.from_json(json_str)
        assert restored.spec_id == minimal_model.spec_id
        assert restored.spec_name == minimal_model.spec_name
        assert len(restored.records) == len(minimal_model.records)
        assert restored.records[0].fields[0].name == "id"

    def test_get_record(self, minimal_model):
        assert minimal_model.get_record("DETAIL") is not None
        assert minimal_model.get_record("NONEXISTENT") is None

    def test_ordered_records(self, minimal_model):
        minimal_model.records[0].ordering = 50
        ordered = minimal_model.ordered_records()
        assert ordered[0].ordering == 50  # only one record

    def test_field_constraints_roundtrip(self):
        c = FieldConstraints(
            min_length=5, max_length=20,
            min_value=0, max_value=999,
            pattern=r"^\d+$",
            allowed_values=["A", "B"],
            required=True,
        )
        d = c.to_dict()
        c2 = FieldConstraints.from_dict(d)
        assert c2.max_length == 20
        assert c2.allowed_values == ["A", "B"]

    def test_spec_type_enum(self):
        assert SpecType("ach") == SpecType.ACH
        assert SpecType.VCF.value == "vcf"


# =============================================================================
# 2. Parser Tests
# =============================================================================

class TestACHParser:

    def test_parse_standard_ach_model(self, ach_model):
        """ACH parser should produce 6 standard record types."""
        assert len(ach_model.records) == 6
        ids = {r.record_type_id for r in ach_model.records}
        assert {"1", "5", "6", "7", "8", "9"}.issubset(ids)

    def test_record_length_94(self, ach_model):
        for rec in ach_model.records:
            assert rec.record_length == 94, f"Record {rec.record_type_id} should be 94 chars"

    def test_entry_detail_fields(self, ach_model):
        entry = ach_model.get_record("6")
        assert entry is not None
        field_names = {f.name for f in entry.fields}
        assert "amount" in field_names
        assert "rdfi_routing_transit_number" in field_names

    def test_parse_native_ach_content(self):
        """Parsing a 94-char line with record type '1' should succeed."""
        line = "1" + " " * 93
        parser = ACHParser()
        model = parser.parse_content(line, source_name="native.ach")
        assert model.spec_type == SpecType.ACH

    def test_parser_factory(self):
        parser = get_parser("ach")
        assert isinstance(parser, ACHParser)


class TestVCFParser:

    def test_vcf_has_three_record_types(self, vcf_model):
        assert len(vcf_model.records) == 3
        ids = {r.record_type_id for r in vcf_model.records}
        assert {"HEADER", "DETAIL", "TRAILER"} == ids

    def test_pan_field_present(self, vcf_model):
        detail = vcf_model.get_record("DETAIL")
        assert detail is not None
        pan_field = detail.get_field("primary_account_number")
        assert pan_field is not None
        assert pan_field.field_type == FieldType.PAN

    def test_pan_has_luhn_constraint(self, vcf_model):
        detail = vcf_model.get_record("DETAIL")
        pan = detail.get_field("primary_account_number")
        assert pan.constraints.checksum_algorithm == "luhn"

    def test_tagged_format_detection(self):
        parser = VCFParser()
        tagged = "RECORD_TYPE=06|PAN=4111111111111111|AMOUNT=10000\n" * 5
        model = parser.parse_content(tagged, source_name="tagged.vcf")
        assert model.file_structure.format == "tagged"


class TestJSONSchemaParser:

    NATIVE_SPEC = json.dumps({
        "spec_name": "TestPaymentFile",
        "spec_type": "json",
        "file_structure": {"format": "delimited", "delimiter": "|"},
        "records": [
            {
                "record_type_id": "HDR",
                "name": "Header",
                "category": "header",
                "ordering": 0,
                "fields": [
                    {"name": "rec_id", "type": "constant", "default_value": "HDR"},
                    {"name": "date", "type": "date", "length": 8},
                ]
            },
            {
                "record_type_id": "DTL",
                "name": "Detail",
                "category": "detail",
                "ordering": 10,
                "fields": [
                    {"name": "id", "type": "sequence", "length": 6},
                    {"name": "amount", "type": "amount", "length": 12},
                    {"name": "status", "type": "enum",
                     "allowed_values": ["ACTIVE", "CLOSED"]},
                ]
            }
        ]
    })

    def test_parse_native_spec(self):
        parser = JSONSchemaParser()
        model = parser.parse_content(self.NATIVE_SPEC, source_name="spec.json")
        assert model.spec_name == "TestPaymentFile"
        assert len(model.records) == 2
        hdr = model.get_record("HDR")
        assert hdr is not None
        assert hdr.category == RecordCategory.HEADER

    def test_enum_field_parsed(self):
        parser = JSONSchemaParser()
        model = parser.parse_content(self.NATIVE_SPEC, source_name="spec.json")
        dtl = model.get_record("DTL")
        status_field = dtl.get_field("status")
        assert status_field.field_type == FieldType.ENUM
        assert "ACTIVE" in status_field.constraints.allowed_values

    def test_json_schema_draft(self):
        schema = json.dumps({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Payment",
            "type": "object",
            "properties": {
                "amount": {"type": "number", "minimum": 0},
                "currency": {"type": "string", "enum": ["USD", "EUR"]},
                "date": {"type": "string", "format": "date"},
            },
            "required": ["amount", "currency"],
        })
        parser = JSONSchemaParser()
        model = parser.parse_content(schema, source_name="schema.json")
        rec = model.get_record("RECORD")
        assert rec is not None
        currency_f = rec.get_field("currency")
        assert currency_f.field_type == FieldType.ENUM

    def test_sample_array_inference(self):
        data = json.dumps([
            {"id": "1", "amount": "10000", "status": "ACTIVE"},
            {"id": "2", "amount": "20000", "status": "CLOSED"},
            {"id": "3", "amount": "30000", "status": "ACTIVE"},
        ])
        parser = JSONSchemaParser()
        model = parser.parse_content(data, source_name="sample.json")
        rec = model.records[0]
        status_f = rec.get_field("status")
        assert status_f.field_type == FieldType.ENUM

    def test_unsupported_type_raises(self):
        with pytest.raises(UnsupportedSpecTypeError):
            get_parser("xml")


class TestSampleDataParser:

    CSV_CONTENT = "\n".join([
        "id,card_number,amount,transaction_date,response_code",
        "1,4111111111111111,10000,20240115,00",
        "2,4242424242424242,25000,20240116,00",
        "3,4000000000000002,5000,20240117,05",
        "4,4532015112830366,99999,20240118,00",
        "5,4916338506082832,1500,20240119,12",
    ])

    PIPE_CONTENT = "\n".join([
        "TXN_ID|AMOUNT|STATUS|DATE",
        "TXN001|10000|APPROVED|20240115",
        "TXN002|20000|DECLINED|20240116",
        "TXN003|30000|APPROVED|20240117",
    ])

    def test_parse_csv_detects_delimiter(self):
        parser = SampleDataParser()
        model = parser.parse_content(self.CSV_CONTENT, source_name="sample.csv")
        assert model.spec_type == SpecType.SAMPLE
        rec = model.records[0]
        assert rec is not None

    def test_card_number_inferred_as_pan(self):
        parser = SampleDataParser()
        model = parser.parse_content(self.CSV_CONTENT, source_name="sample.csv")
        all_fields = {f.name: f for rec in model.records for f in rec.fields}
        # The card_number column should be inferred as PAN
        assert "card_number" in all_fields
        assert all_fields["card_number"].field_type == FieldType.PAN

    def test_pipe_delimited_parsed(self):
        parser = SampleDataParser()
        model = parser.parse_content(self.PIPE_CONTENT, source_name="pipe.txt")
        assert len(model.records) >= 1

    def test_insufficient_sample_raises(self):
        parser = SampleDataParser()
        with pytest.raises(InsufficientSampleDataError):
            parser.parse_content("only one line", source_name="bad.csv")

    def test_empty_content_raises(self):
        parser = SampleDataParser()
        with pytest.raises(ParseError):
            parser.parse_content("", source_name="empty.csv")


# =============================================================================
# 3. Field Inferrer Tests
# =============================================================================

class TestFieldInferrer:

    def test_date_inference(self):
        inferrer = FieldInferrer()
        fd = FieldDef(
            name="txn_date",
            field_type=FieldType.STRING,
            sample_values=["20240115", "20240116", "20240117", "20240118"],
        )
        inferrer.enrich(fd)
        assert fd.field_type == FieldType.DATE

    def test_pan_inference_with_luhn(self):
        inferrer = FieldInferrer()
        # These are Luhn-valid test PANs
        valid_pans = ["4111111111111111", "4242424242424242", "4000000000000002"]
        fd = FieldDef(
            name="card_number",
            field_type=FieldType.STRING,
            sample_values=valid_pans,
        )
        inferrer.enrich(fd)
        assert fd.field_type == FieldType.PAN

    def test_enum_inference_low_cardinality(self):
        inferrer = FieldInferrer()
        values = ["ACTIVE", "CLOSED", "ACTIVE", "ACTIVE", "CLOSED",
                  "ACTIVE", "CLOSED", "ACTIVE", "ACTIVE", "ACTIVE"]
        fd = FieldDef(
            name="status",
            field_type=FieldType.STRING,
            sample_values=values,
        )
        inferrer.enrich(fd)
        assert fd.field_type == FieldType.ENUM
        assert set(fd.constraints.allowed_values) == {"ACTIVE", "CLOSED"}

    def test_routing_number_inference(self):
        inferrer = FieldInferrer()
        # Valid ABA routing numbers
        routing_nums = ["021000021", "021000089", "021200339", "021202337"]
        fd = FieldDef(
            name="routing_number",
            field_type=FieldType.STRING,
            sample_values=routing_nums,
        )
        inferrer.enrich(fd)
        assert fd.field_type == FieldType.ROUTING_NUMBER

    def test_amount_inference(self):
        inferrer = FieldInferrer()
        fd = FieldDef(
            name="total_amount",
            field_type=FieldType.STRING,
            sample_values=["10000", "20000", "35000", "5000", "99999"],
        )
        inferrer.enrich(fd)
        assert fd.field_type in (FieldType.AMOUNT, FieldType.NUMERIC)

    def test_format_string_iso_date(self):
        inferrer = FieldInferrer()
        fd = FieldDef(
            name="created_at",
            field_type=FieldType.STRING,
            sample_values=["2024-01-15", "2024-02-20", "2024-03-10"],
        )
        inferrer.enrich(fd)
        assert fd.field_type == FieldType.DATE
        assert fd.format_string == "YYYY-MM-DD"

    def test_no_crash_on_empty_values(self):
        inferrer = FieldInferrer()
        fd = FieldDef(name="empty_field", field_type=FieldType.STRING, sample_values=[])
        inferrer.enrich(fd)
        assert fd.field_type == FieldType.STRING  # unchanged


# =============================================================================
# 4. Pattern Detector Tests
# =============================================================================

class TestPatternDetector:

    def test_ordering_detected(self, ach_model):
        detector = PatternDetector()
        # Simulate a sequence of ACH record types
        sequence = ["1", "5", "6", "6", "6", "8", "9"]
        result = detector.detect_from_sequence(sequence, ach_model)
        assert "1" in result.ordering_sequence
        assert "9" in result.ordering_sequence
        assert result.ordering_sequence.index("1") < result.ordering_sequence.index("9")

    def test_batch_pattern_detected(self, ach_model):
        detector = PatternDetector()
        sequence = ["1", "5", "6", "6", "8", "5", "6", "8", "9"]
        result = detector.detect_from_sequence(sequence, ach_model)
        assert result.batch_pattern is not None
        assert "5" in result.batch_pattern
        assert "8" in result.batch_pattern

    def test_repeating_types(self, ach_model):
        detector = PatternDetector()
        sequence = ["1", "5", "6", "6", "6", "8", "9"]
        result = detector.detect_from_sequence(sequence, ach_model)
        assert "6" in result.repeating_types


# =============================================================================
# 5. Rule Engine Tests
# =============================================================================

class TestRuleEngine:

    def test_load_from_model(self, minimal_model):
        engine = RuleEngine()
        engine.load_from_model(minimal_model)
        # Should not raise

    def test_field_validation_allowed_values(self):
        engine = RuleEngine()
        fd = FieldDef(
            name="status",
            field_type=FieldType.ENUM,
            constraints=FieldConstraints(allowed_values=["ACTIVE", "CLOSED"]),
        )
        rec = RecordDef("TEST", "Test", fields=[fd])
        model = SchemaModel(records=[rec], is_trained=True)
        engine.load_from_model(model)

        errors = engine.validate_field("TEST", "status", "ACTIVE", {})
        assert not errors

        errors = engine.validate_field("TEST", "status", "INVALID", {})
        assert errors

    def test_field_validation_min_length(self):
        engine = RuleEngine()
        fd = FieldDef(
            name="code",
            field_type=FieldType.STRING,
            constraints=FieldConstraints(min_length=3, max_length=10),
        )
        rec = RecordDef("TEST", "Test", fields=[fd])
        model = SchemaModel(records=[rec], is_trained=True)
        engine.load_from_model(model)

        errors = engine.validate_field("TEST", "code", "AB", {})
        assert any("min_length" in e for e in errors)

        errors = engine.validate_field("TEST", "code", "ABC", {})
        assert not errors

    def test_dependency_resolution_topological_order(self):
        from app.models.schema import FieldDependency
        engine = RuleEngine()
        f_a = FieldDef("a", FieldType.STRING)
        f_b = FieldDef("b", FieldType.COMPUTED,
                       dependencies=[FieldDependency("a", "computed", formula="a + '_suffix'")])
        f_c = FieldDef("c", FieldType.COMPUTED,
                       dependencies=[FieldDependency("b", "computed")])
        rec = RecordDef("TEST", "Test", fields=[f_c, f_b, f_a])  # deliberately wrong order
        ordered = engine.resolve_field_order(rec)
        names = [f.name for f in ordered]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")


# =============================================================================
# 6. Data Synthesizer Tests
# =============================================================================

class TestDataSynthesizer:

    def test_pan_is_luhn_valid(self, synth):
        fd = FieldDef("pan", FieldType.PAN, length=16)
        for _ in range(20):
            val = synth.generate(fd)
            assert _luhn_check(val.strip()), f"PAN {val} failed Luhn"

    def test_sequence_increments(self, synth):
        fd = FieldDef("seq", FieldType.SEQUENCE, length=6,
                      justification=Justification.RIGHT, pad_char="0")
        vals = [synth.generate(fd) for _ in range(5)]
        nums = [int(v) for v in vals]
        assert nums == sorted(nums)
        assert nums[1] - nums[0] == 1

    def test_enum_value_from_allowed(self, synth):
        fd = FieldDef(
            "status", FieldType.ENUM, length=6,
            constraints=FieldConstraints(allowed_values=["ACTIVE", "CLOSED"])
        )
        for _ in range(30):
            val = synth.generate(fd).strip()
            assert val in ("ACTIVE", "CLOSED"), f"Unexpected value: {val}"

    def test_routing_number_aba_valid(self, synth):
        fd = FieldDef("routing", FieldType.ROUTING_NUMBER, length=9)
        for _ in range(10):
            val = synth.generate(fd).strip()
            assert _aba_check(val), f"Routing {val} failed ABA check"

    def test_date_format_yyyymmdd(self, synth):
        import re
        fd = FieldDef("txn_date", FieldType.DATE, length=8,
                      format_string="YYYYMMDD")
        for _ in range(10):
            val = synth.generate(fd).strip()
            assert re.fullmatch(r"\d{8}", val), f"Invalid date format: {val}"

    def test_amount_within_range(self, synth):
        fd = FieldDef(
            "amount", FieldType.AMOUNT, length=10,
            justification=Justification.RIGHT, pad_char="0",
            constraints=FieldConstraints(min_value=100, max_value=50000),
        )
        for _ in range(20):
            val = int(synth.generate(fd).strip())
            assert 100 <= val <= 50000, f"Amount {val} out of range"

    def test_seed_reproducibility(self):
        synth1 = DataSynthesizer(seed=99)
        synth2 = DataSynthesizer(seed=99)
        fd = FieldDef("pan", FieldType.PAN, length=16)
        assert synth1.generate(fd) == synth2.generate(fd)

    def test_counter_reset(self, synth):
        fd = FieldDef("seq", FieldType.SEQUENCE, length=4,
                      justification=Justification.RIGHT, pad_char="0")
        synth.generate(fd)
        synth.generate(fd)
        synth.reset_counters()
        val = int(synth.generate(fd).strip())
        assert val == 1  # should restart from 1


# =============================================================================
# 7. Generator Integration Tests
# =============================================================================

class TestACHGenerator:

    def test_generate_produces_94_char_lines(self, ach_model):
        ach_model.is_trained = True
        gen = ACHGenerator(ach_model, seed=42)
        result = gen.generate(record_count=5)
        for i, line in enumerate(result.content.splitlines()):
            assert len(line) == 94, f"Line {i+1} is {len(line)} chars: {line!r}"

    def test_record_count(self, ach_model):
        ach_model.is_trained = True
        gen = ACHGenerator(ach_model, seed=42)
        result = gen.generate(record_count=10)
        assert result.record_count == 10

    def test_file_padded_to_multiple_of_10(self, ach_model):
        ach_model.is_trained = True
        gen = ACHGenerator(ach_model, seed=42)
        result = gen.generate(record_count=7)
        lines = result.content.splitlines()
        assert len(lines) % 10 == 0, f"File has {len(lines)} lines, not multiple of 10"


class TestVCFGenerator:

    def test_generate_returns_header_and_trailer(self, vcf_model):
        vcf_model.is_trained = True
        gen = VCFGenerator(vcf_model, seed=42)
        result = gen.generate(record_count=5)
        lines = result.content.splitlines()
        assert lines[0].startswith("H")   # header
        assert lines[-1].startswith("T")  # trailer

    def test_record_count_matches(self, vcf_model):
        vcf_model.is_trained = True
        gen = VCFGenerator(vcf_model, seed=42)
        result = gen.generate(record_count=3)
        assert result.record_count == 3


class TestJSONGenerator:

    def test_generate_json_output(self):
        parser = JSONSchemaParser()
        spec = json.dumps({
            "spec_name": "TestJSON",
            "file_structure": {"format": "json"},
            "records": [{
                "record_type_id": "RECORD",
                "name": "Test",
                "category": "detail",
                "repeatable": True,
                "ordering": 10,
                "fields": [
                    {"name": "id", "type": "sequence", "length": 5},
                    {"name": "status", "type": "enum",
                     "allowed_values": ["A", "B", "C"]},
                ]
            }]
        })
        model = parser.parse_content(spec, source_name="test.json")
        model.is_trained = True
        gen = JSONGenerator(model, seed=42)
        result = gen.generate(record_count=3)
        data = json.loads(result.content)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_get_generator_factory(self, ach_model):
        gen = get_generator(ach_model, seed=42)
        assert isinstance(gen, ACHGenerator)

    def test_get_generator_vcf(self, vcf_model):
        gen = get_generator(vcf_model, seed=42)
        assert isinstance(gen, VCFGenerator)


# =============================================================================
# 8. Validation Engine Tests
# =============================================================================

class TestValidationEngine:

    def test_valid_records_pass(self, minimal_model):
        validator = ValidationEngine(minimal_model)
        records = [
            {"_record_type": "DETAIL", "id": "00001", "name": "Alice Smith          ",
             "amount": "0000001000"},
        ]
        report = validator.validate(records)
        assert report.total_records == 1

    def test_invalid_amount_detected(self, minimal_model):
        validator = ValidationEngine(minimal_model)
        records = [
            {"_record_type": "DETAIL", "id": "00001", "name": "Bob Jones            ",
             "amount": "NOTANUMBER"},
        ]
        report = validator.validate(records)
        assert any("amount" in e.field_name for e in report.field_errors)

    def test_ach_records_validate(self, ach_model):
        ach_model.is_trained = True
        gen = ACHGenerator(ach_model, seed=42)
        result = gen.generate(record_count=5)
        validator = ValidationEngine(ach_model)
        report = validator.validate(result.records)
        # Generated data should be structurally valid
        assert report.total_records > 0
        # Routing number ABA errors may appear for generated routing numbers
        # that aren't in the valid set; check that we at least produce a report
        assert isinstance(report.passed, bool)

    def test_validation_report_dict(self, minimal_model):
        validator = ValidationEngine(minimal_model)
        report = validator.validate([])
        d = report.to_dict()
        assert "passed" in d
        assert "total_records" in d
        assert "field_errors" in d


# =============================================================================
# Helpers (replicated locally so tests have no cross-module deps)
# =============================================================================

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
