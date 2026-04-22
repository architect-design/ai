"""
Test Suite for Financial LLM Studio
=====================================
Tests for spec learning, validation, and generation engines.
Run:
    cd financial_llm && pytest tests/ -v
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spec_engine   import SpecEngine, FinancialTokenizer, PatternExtractor
from core.validator     import Validator, FieldValidator, ValidationResult
from core.generator     import Generator, FieldGenerator
from core.db_connector  import DBConnector
from formats.builtin_formats import NACHA_SPEC, VISA_VCF_SPEC, ORACLE_GL_SPEC, seed_knowledge_base


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    """Fresh SpecEngine with a temp spec store."""
    import core.spec_engine as se_mod
    tmp = tmp_path_factory.mktemp("specs")
    se_mod.SPEC_STORE = str(tmp)
    eng = SpecEngine()
    seed_knowledge_base(eng.kb)
    return eng

@pytest.fixture(scope="module")
def validator(engine):
    return Validator(engine.kb)

@pytest.fixture(scope="module")
def generator(engine):
    return Generator(engine.kb)


# ════════════════════════════════════════════════════════════════════════
# Tokenizer
# ════════════════════════════════════════════════════════════════════════
class TestTokenizer:
    def test_basic(self):
        t = FinancialTokenizer()
        tokens = t.tokenize("RECORD_TYPE  1  1  1  N\nAMOUNT  2  11  10  AN")
        assert len(tokens) == 2

    def test_field_detection(self):
        t = FinancialTokenizer()
        tokens = t.tokenize("AMOUNT  2  11  10  N")
        assert tokens[0]['is_field'] is True

    def test_comment_skipped(self):
        t = FinancialTokenizer()
        tokens = t.tokenize("# This is a comment\nFIELD_NAME  1  5  5  AN")
        assert len(tokens) == 1

    def test_data_type_inference(self):
        t = FinancialTokenizer()
        tokens = t.tokenize("AMOUNT field 10 digits")
        assert tokens[0]['data_type'] == 'amount'


# ════════════════════════════════════════════════════════════════════════
# PatternExtractor
# ════════════════════════════════════════════════════════════════════════
class TestPatternExtractor:
    def test_fixed_width(self):
        ex = PatternExtractor()
        t  = FinancialTokenizer()
        spec_text = "FIELD_A  1-10   10  AN\nFIELD_B  11-20  10  N"
        tokens = t.tokenize(spec_text)
        fields = ex.extract_fields(tokens, 'custom')
        assert len(fields) >= 1

    def test_required_detection(self):
        ex = PatternExtractor()
        t  = FinancialTokenizer()
        spec_text = "NAME  1-20  20  AN  Required"
        tokens = t.tokenize(spec_text)
        fields = ex.extract_fields(tokens, 'custom')
        if fields:
            assert fields[0].get('required') is True


# ════════════════════════════════════════════════════════════════════════
# SpecEngine – Training
# ════════════════════════════════════════════════════════════════════════
class TestSpecEngine:
    CUSTOM_SPEC = """
    RECORD_TYPE    1    1    1   N   Required   Value=H
    COMPANY_ID     2    11   10  AN  Required
    AMOUNT         12   23   12  N   Required
    DATE           24   31   8   DT  Required   Format=YYYYMMDD
    DESCRIPTION    32   71   40  AN  Optional
    """

    def test_train_returns_spec(self, engine):
        spec = engine.train("TEST_CUSTOM", self.CUSTOM_SPEC, "custom", "Test spec")
        assert spec['name'] == "TEST_CUSTOM"
        assert spec['format_type'] == "custom"

    def test_train_extracts_fields(self, engine):
        spec = engine.train("TEST_FIELDS", self.CUSTOM_SPEC, "custom")
        assert spec['field_count'] > 0

    def test_list_includes_trained(self, engine):
        engine.train("TEST_LIST_CHECK", self.CUSTOM_SPEC, "custom")
        assert "TEST_LIST_CHECK" in engine.list_specs()

    def test_get_spec(self, engine):
        engine.train("TEST_GET", self.CUSTOM_SPEC, "custom")
        loaded = engine.get_spec("TEST_GET")
        assert loaded is not None
        assert loaded['name'] == "TEST_GET"

    def test_delete_spec(self, engine):
        engine.train("TEST_DELETE_ME", self.CUSTOM_SPEC, "custom")
        assert engine.delete_spec("TEST_DELETE_ME")
        assert engine.get_spec("TEST_DELETE_ME") is None

    def test_builtin_nacha_seeded(self, engine):
        assert engine.get_spec("NACHA") is not None

    def test_builtin_visa_seeded(self, engine):
        assert engine.get_spec("VISA_VCF") is not None

    def test_builtin_oracle_gl_seeded(self, engine):
        assert engine.get_spec("ORACLE_GL") is not None

    def test_identify_nacha(self, engine):
        sample = "1234567890PPD PAYROLL 220101"
        matches = engine.identify(sample)
        assert isinstance(matches, list)


# ════════════════════════════════════════════════════════════════════════
# FieldValidator
# ════════════════════════════════════════════════════════════════════════
class TestFieldValidator:
    def setup_method(self):
        self.fv = FieldValidator()

    def _run(self, value, field):
        r = ValidationResult()
        self.fv.validate(value, field, r, 1)
        return r

    def test_numeric_ok(self):
        r = self._run("12345", {"name": "X", "data_type": "numeric", "length": 5})
        assert r.failed == 0

    def test_numeric_fail(self):
        r = self._run("12ABC", {"name": "X", "data_type": "numeric", "length": 5})
        assert r.failed > 0

    def test_required_empty(self):
        r = self._run("", {"name": "X", "data_type": "alphanumeric", "length": 5, "required": True})
        assert r.failed > 0

    def test_optional_empty(self):
        r = self._run("", {"name": "X", "data_type": "alphanumeric", "length": 5, "required": False})
        assert r.failed == 0

    def test_length_exceeded(self):
        r = self._run("123456789", {"name": "X", "data_type": "numeric", "length": 5})
        assert r.failed > 0

    def test_allowed_values_ok(self):
        r = self._run("22", {"name": "TC", "data_type": "numeric", "length": 2,
                             "validation": {"allowed": ["22","27"]}})
        assert r.failed == 0

    def test_allowed_values_fail(self):
        r = self._run("99", {"name": "TC", "data_type": "numeric", "length": 2,
                             "validation": {"allowed": ["22","27"]}})
        assert r.failed > 0

    def test_date_yymmdd_ok(self):
        r = self._run("260419", {"name": "DT", "data_type": "date", "format": "YYMMDD", "length": 6})
        assert r.failed == 0

    def test_date_bad(self):
        r = self._run("999999", {"name": "DT", "data_type": "date", "format": "YYMMDD", "length": 6})
        assert r.failed > 0

    def test_valid_routing_number(self):
        r = self._run("021000021", {"name": "RTN", "data_type": "routing_number", "length": 9})
        assert r.failed == 0

    def test_invalid_routing_number(self):
        r = self._run("123456789", {"name": "RTN", "data_type": "routing_number", "length": 9})
        assert r.failed > 0

    def test_amount_ok(self):
        r = self._run("1234.56", {"name": "AMT", "data_type": "amount", "length": 10})
        assert r.failed == 0


# ════════════════════════════════════════════════════════════════════════
# Validator (file-level)
# ════════════════════════════════════════════════════════════════════════
class TestValidator:
    def test_oracle_gl_valid(self, validator):
        content = (
            "STATUS|LEDGER_ID|ACCOUNTING_DATE|CURRENCY_CODE|DATE_CREATED|CREATED_BY|"
            "ACTUAL_FLAG|USER_JE_SOURCE_NAME|USER_JE_CATEGORY_NAME|ENCUMBRANCE_TYPE_ID|"
            "BUDGET_VERSION_ID|BALANCED_JE_FLAG|BALANCING_SEGMENT_VALUE\n"
            "N|1001|01-JAN-2026|USD||Admin|A|Manual|Adjustment||||"
        )
        result = validator.validate("ORACLE_GL", content, delimiter="|")
        assert result.total_records >= 1

    def test_unknown_spec(self, validator):
        result = validator.validate("NONEXISTENT_SPEC_XYZ", "data")
        assert len(result.errors) > 0


# ════════════════════════════════════════════════════════════════════════
# Generator
# ════════════════════════════════════════════════════════════════════════
class TestGenerator:
    def test_nacha_generate(self, generator):
        output = generator.generate("NACHA", num_records=2, seed=42)
        lines  = [l for l in output.splitlines() if l.strip()]
        assert len(lines) >= 4   # header + batch_hdr + 2 detail + batch_ctrl + file_ctrl
        # All lines should be 94 chars
        for line in lines:
            if line != '9' * 94:
                assert len(line) == 94, f"Line length {len(line)}: {line[:20]}…"

    def test_oracle_gl_generate(self, generator):
        output = generator.generate("ORACLE_GL", num_records=3, seed=42)
        lines  = [l for l in output.splitlines() if l.strip()]
        assert len(lines) >= 4   # header row + 1 journal hdr + 3 lines

    def test_visa_vcf_generate(self, generator):
        output = generator.generate("VISA_VCF", num_records=5, seed=42)
        lines  = [l for l in output.splitlines() if l.strip()]
        assert len(lines) == 5

    def test_seed_reproducibility(self, generator):
        out1 = generator.generate("NACHA", num_records=2, seed=999)
        out2 = generator.generate("NACHA", num_records=2, seed=999)
        assert out1 == out2

    def test_generate_with_db_data(self, generator):
        db_data = [
            {"INDIVIDUAL_NAME": "TEST USER", "AMOUNT": "0000001000",
             "DFI_ACCOUNT_NUMBER": "123456789012345"},
        ]
        output = generator.generate("NACHA", num_records=1, seed=1, db_data=db_data)
        assert "TEST USER" in output

    def test_unknown_spec_returns_error(self, generator):
        output = generator.generate("NONEXISTENT_XYZ")
        assert "ERROR" in output


# ════════════════════════════════════════════════════════════════════════
# FieldGenerator
# ════════════════════════════════════════════════════════════════════════
class TestFieldGenerator:
    def setup_method(self):
        self.fg = FieldGenerator()

    def test_routing_number_checksum(self):
        from core.validator import FieldValidator
        fv = FieldValidator()
        for _ in range(20):
            rtn = self.fg._routing_number()
            assert len(rtn) == 9
            assert fv._validate_routing(rtn), f"Invalid RTN: {rtn}"

    def test_amount_numeric(self):
        val = self.fg._amount(10, {})
        assert val.isdigit()
        assert len(val) == 10

    def test_date_formats(self):
        for fmt in ['YYMMDD','YYYYMMDD','MMDD','DD-MON-YYYY']:
            val = self.fg._date(fmt)
            assert isinstance(val, str)
            assert len(val) > 0

    def test_fixed_value(self):
        val = self.fg.generate({"name": "X", "value": "1", "length": 1})
        assert val.strip() == "1"

    def test_allowed_values(self):
        field = {"name": "X", "data_type": "numeric", "length": 2,
                 "validation": {"allowed": ["22","27","32"]}}
        for _ in range(20):
            val = self.fg.generate(field)
            assert val in ["22","27","32"]


# ════════════════════════════════════════════════════════════════════════
# DBConnector
# ════════════════════════════════════════════════════════════════════════
class TestDBConnector:
    def test_not_connected_by_default(self):
        db = DBConnector()
        assert not db.connected

    def test_sqlite_connect(self, tmp_path):
        db = DBConnector()
        db_path = str(tmp_path / "test.db")
        db.connect(f"sqlite:///{db_path}")
        assert db.connected
        db.disconnect()
        assert not db.connected

    def test_fetch_from_sqlite(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test2.db")
        # Create a test table
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE tx (id INTEGER, name TEXT, amount REAL)")
        con.execute("INSERT INTO tx VALUES (1,'ALICE',100.0)")
        con.execute("INSERT INTO tx VALUES (2,'BOB',200.0)")
        con.commit()
        con.close()

        db = DBConnector()
        db.connect(f"sqlite:///{db_path}")
        rows = db.fetch("SELECT * FROM tx")
        assert len(rows) == 2
        assert rows[0]['name'] == 'ALICE'
        db.disconnect()

    def test_list_tables_sqlite(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test3.db")
        con = sqlite3.connect(db_path)
        con.execute("CREATE TABLE users (id INTEGER)")
        con.execute("CREATE TABLE orders (id INTEGER)")
        con.commit(); con.close()

        db = DBConnector()
        db.connect(f"sqlite:///{db_path}")
        tables = db.list_tables()
        assert 'users' in tables
        assert 'orders' in tables
        db.disconnect()

    def test_mock_data_nacha(self):
        db = DBConnector()
        rows = db.get_mock_data('nacha', 5)
        assert len(rows) == 5
        assert 'INDIVIDUAL_NAME' in rows[0]

    def test_mock_data_oracle_gl(self):
        db = DBConnector()
        rows = db.get_mock_data('oracle_gl', 3)
        assert len(rows) == 3
        assert 'SEGMENT1' in rows[0]

    def test_mask_password(self):
        db = DBConnector()
        masked = db._mask_password("oracle://user:secret@host/db")
        assert "secret" not in masked
        assert "****" in masked

    def test_presets_available(self):
        assert "oracle_sample" in DBConnector.PRESETS
        assert "postgres" in DBConnector.PRESETS
        assert "sqlite" in DBConnector.PRESETS
