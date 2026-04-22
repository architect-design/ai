"""
Data Synthesizer
================
Generates realistic, type-aware, constraint-conforming synthetic values
for every FieldType.

Key design:
  - Seeded random for reproducibility
  - Per-type generation strategies
  - Luhn-valid PANs, ABA-valid routing numbers
  - Date/amount ranges from learned constraints
  - Enum values sampled from allowed_values
  - Sequence counters that auto-increment
"""
from __future__ import annotations

import random
import re
import string
from datetime import date, timedelta
from typing import Any

from app.models.schema import FieldDef, FieldType, FieldConstraints, Justification


class DataSynthesizer:
    """
    Stateful synthesizer that maintains counters and context
    for a single file generation run.
    """

    # Realistic BIN prefixes for VISA
    _VISA_BINS = ["4111", "4242", "4000", "4532", "4916", "4929", "4539"]
    # Real valid ABA routing numbers for testing
    _SAMPLE_ROUTING_NUMBERS = [
        "021000021", "021000089", "021200339", "021202337",
        "021300077", "021300481", "021301011", "021301115",
        "022000020", "022000046", "022000046",
    ]
    _FIRST_NAMES = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer",
        "Michael", "Linda", "William", "Barbara", "David", "Elizabeth",
        "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
        "Charles", "Karen", "Chris", "Amanda", "Daniel", "Emily",
    ]
    _LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
        "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    ]
    _MERCHANT_NAMES = [
        "AMAZON MARKETPLACE", "WALMART SUPERCENTER", "TARGET CORP",
        "BEST BUY CO", "HOME DEPOT", "COSTCO WHOLESALE", "WALGREENS",
        "KROGER COMPANY", "CVS PHARMACY", "WHOLE FOODS MKT",
        "SHELL OIL", "CHEVRON STATION", "STARBUCKS COFFEE",
        "MCDONALDS CORP", "APPLE ONLINE STORE",
    ]
    _CITIES = [
        "NEW YORK NY", "LOS ANGELES CA", "CHICAGO IL", "HOUSTON TX",
        "PHOENIX AZ", "PHILADELPHIA PA", "SAN ANTONIO TX", "SAN DIEGO CA",
    ]
    _MCC_CODES = [
        "5411", "5912", "5999", "5310", "5251", "5541", "5542",
        "5812", "5814", "5999", "7011", "7372",
    ]

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._counters: dict[str, int] = {}
        self._context: dict[str, Any] = {}

    def reset_counters(self):
        self._counters.clear()
        self._context.clear()

    def next_counter(self, key: str, start: int = 1) -> int:
        self._counters[key] = self._counters.get(key, start - 1) + 1
        return self._counters[key]

    # ── Main dispatch ─────────────────────────────────────────────────────────

    def generate(self, field: FieldDef, context: dict[str, Any] | None = None) -> str:
        """Generate a single field value as a string."""
        ctx = context or {}
        val = self._dispatch(field, ctx)
        return self._format_value(val, field)

    def _dispatch(self, field: FieldDef, ctx: dict) -> Any:
        ft = field.field_type
        c = field.constraints

        if ft == FieldType.CONSTANT:
            return field.default_value or ""

        if ft == FieldType.SEQUENCE:
            key = f"seq_{field.name}"
            return self.next_counter(key)

        if ft == FieldType.ENUM:
            if c.allowed_values:
                return self._rng.choice(c.allowed_values)
            return ""

        if ft == FieldType.PAN:
            return self._gen_pan()

        if ft == FieldType.CVV:
            return self._gen_cvv()

        if ft == FieldType.EXPIRY:
            return self._gen_expiry(field)

        if ft == FieldType.ROUTING_NUMBER:
            return self._gen_routing_number()

        if ft == FieldType.ACCOUNT_NUMBER:
            return self._gen_account_number(field)

        if ft == FieldType.DATE:
            return self._gen_date(field)

        if ft == FieldType.DATETIME:
            return self._gen_datetime(field)

        if ft == FieldType.AMOUNT:
            return self._gen_amount(field)

        if ft == FieldType.NUMERIC:
            return self._gen_numeric(field)

        if ft == FieldType.BOOLEAN:
            return self._rng.choice(["true", "false"])

        if ft == FieldType.CHECKSUM:
            return self._gen_checksum(field, ctx)

        if ft == FieldType.COMPUTED:
            return self._gen_computed(field, ctx)

        if ft == FieldType.ALPHANUMERIC:
            return self._gen_alphanumeric(field)

        # Default: string
        return self._gen_string(field)

    # ── Type-specific generators ──────────────────────────────────────────────

    def _gen_pan(self) -> str:
        """Generate a Luhn-valid 16-digit PAN."""
        bin_prefix = self._rng.choice(self._VISA_BINS)
        remaining_length = 15 - len(bin_prefix)
        middle = "".join(str(self._rng.randint(0, 9)) for _ in range(remaining_length))
        partial = bin_prefix + middle
        check = self._luhn_check_digit(partial)
        return partial + str(check)

    def _gen_cvv(self) -> str:
        return f"{self._rng.randint(100, 999)}"

    def _gen_expiry(self, field: FieldDef) -> str:
        """Generate a future expiry date. Returns YYMM or MM/YY."""
        today = date.today()
        months_ahead = self._rng.randint(6, 60)
        exp_month = (today.month - 1 + months_ahead) % 12 + 1
        exp_year = today.year + (today.month - 1 + months_ahead) // 12
        yy = str(exp_year)[-2:]
        mm = f"{exp_month:02d}"
        fmt = field.format_string or ""
        if "MM/YY" in fmt.upper():
            return f"{mm}/{yy}"
        if "/" in fmt:
            return f"{mm}/{yy}"
        return f"{yy}{mm}"  # YYMM (VCF default)

    def _gen_routing_number(self) -> str:
        """Return a real ABA routing number from the sample list."""
        return self._rng.choice(self._SAMPLE_ROUTING_NUMBERS)

    def _gen_account_number(self, field: FieldDef) -> str:
        length = field.length or field.constraints.max_length or 10
        length = min(max(int(length), 6), 17)
        return "".join(str(self._rng.randint(0, 9)) for _ in range(length))

    def _gen_date(self, field: FieldDef) -> str:
        fmt = field.format_string or "YYYYMMDD"
        base = date.today()
        delta_days = self._rng.randint(-180, 0)   # recent past dates
        d = base + timedelta(days=delta_days)
        fmt_upper = fmt.upper()
        if "YYYY-MM-DD" in fmt_upper or "-" in fmt:
            return d.strftime("%Y-%m-%d")
        if "YYYYMMDD" in fmt_upper:           # must come before YYMMDD check
            return d.strftime("%Y%m%d")
        if "YYMMDD" in fmt_upper:
            return d.strftime("%y%m%d")
        if "MM/DD/YYYY" in fmt_upper:
            return d.strftime("%m/%d/%Y")
        return d.strftime("%Y%m%d")

    def _gen_datetime(self, field: FieldDef) -> str:
        from datetime import datetime
        dt = datetime.now()
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _gen_amount(self, field: FieldDef) -> str:
        c = field.constraints
        min_v = int(c.min_value or 100)
        max_v = int(c.max_value or 999_999)
        # Clamp to reasonable test values
        min_v = max(min_v, 1)
        max_v = min(max_v, 9_999_999)
        amount = self._rng.randint(min_v, max_v)
        length = field.length or field.constraints.max_length
        if length:
            return str(amount).zfill(int(length))
        return str(amount)

    def _gen_numeric(self, field: FieldDef) -> str:
        c = field.constraints
        min_v = int(c.min_value or 1)
        max_v = int(c.max_value or 9999)
        val = self._rng.randint(min_v, max_v)
        length = field.length or field.constraints.max_length
        if length:
            return str(val).zfill(int(length))
        return str(val)

    def _gen_alphanumeric(self, field: FieldDef) -> str:
        length = field.length or field.constraints.max_length or 10
        chars = string.ascii_uppercase + string.digits
        return "".join(self._rng.choice(chars) for _ in range(int(length)))

    def _gen_string(self, field: FieldDef) -> str:
        """Generate a realistic string based on field name."""
        name = field.name.lower()
        length = field.length or field.constraints.max_length or 20

        if "name" in name and "merchant" in name:
            return self._rng.choice(self._MERCHANT_NAMES)
        if "cardholder" in name or ("name" in name and "individual" in name):
            fn = self._rng.choice(self._FIRST_NAMES)
            ln = self._rng.choice(self._LAST_NAMES)
            return f"{fn} {ln}"
        if "city" in name:
            return self._rng.choice(self._CITIES)
        if "description" in name:
            return self._rng.choice(["PAYROLL", "VENDOR PMT", "TRANSFER", "REIMBURS"])
        if "company" in name:
            return self._rng.choice(
                ["ACME CORP", "INITECH LLC", "GLOBEX INC", "UMBRELLA CO"]
            )
        if "terminal" in name:
            return self._gen_alphanumeric_of(8)
        if "merchant_id" in name:
            return self._gen_alphanumeric_of(15)
        if "auth" in name or "authorization" in name:
            return self._gen_alphanumeric_of(6)
        if "reference" in name:
            return self._gen_alphanumeric_of(12)

        # Fallback: random alphanumeric of correct length
        length = min(int(length), 100)
        return self._gen_alphanumeric_of(length)

    def _gen_alphanumeric_of(self, n: int) -> str:
        chars = string.ascii_uppercase + string.digits
        return "".join(self._rng.choice(chars) for _ in range(n))

    def _gen_checksum(self, field: FieldDef, ctx: dict) -> str:
        """
        For most checksum fields, return a placeholder that the
        generator's post-processing step will replace.
        """
        algo = field.constraints.checksum_algorithm or "mod10"
        return f"__CHECKSUM_{algo.upper()}__"

    def _gen_computed(self, field: FieldDef, ctx: dict) -> str:
        return ""

    # ── Value formatting ──────────────────────────────────────────────────────

    def _format_value(self, value: Any, field: FieldDef) -> str:
        """Pad/truncate value to fit the field's fixed-width slot."""
        raw = str(value) if value is not None else ""

        target_len = field.length or field.constraints.max_length
        if not target_len:
            return raw

        target_len = int(target_len)

        if len(raw) > target_len:
            raw = raw[:target_len]

        if field.justification == Justification.RIGHT:
            return raw.rjust(target_len, field.pad_char or "0")
        else:
            return raw.ljust(target_len, field.pad_char or " ")

    # ── Checksum helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _luhn_check_digit(partial: str) -> int:
        """Compute the Luhn check digit for a partial PAN string."""
        digits = [int(d) for d in reversed(partial)]
        total = 0
        for i, d in enumerate(digits):
            if i % 2 == 0:  # position 0 is the rightmost of partial
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return (10 - (total % 10)) % 10

    # ── Convenience generators for specific contexts ───────────────────────────

    def gen_individual_name(self) -> str:
        return f"{self._rng.choice(self._FIRST_NAMES)} {self._rng.choice(self._LAST_NAMES)}"

    def gen_company_name(self) -> str:
        return self._rng.choice(["ACME CORPORATION", "INITECH LLC", "GLOBEX INC", "UMBRELLA CORP"])

    def gen_trace_number(self, odfi_routing: str, sequence: int) -> str:
        odfi_prefix = odfi_routing[:8]
        return f"{odfi_prefix}{sequence:07d}"

    def gen_mcc(self) -> str:
        return self._rng.choice(self._MCC_CODES)
