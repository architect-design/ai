"""
Test Data Generator
====================
Generates realistic synthetic test data for any registered spec.
Supports NACHA, VISA VCF, Oracle GL, and custom formats.
Can also accept data pulled from a database (via DBConnector).
"""

import re
import random
import string
import json
from datetime import datetime, timedelta
from typing import Any


# ── Shared RNG (seeded for reproducibility) ───────────────────────────────
RNG = random.Random()


class FieldGenerator:
    """Generates a single realistic field value from a field definition."""

    COMPANY_NAMES  = ["ACME CORP", "GLOBEX INC", "INITECH LLC", "UMBRELLA CO",
                      "STARK INDUSTRIES", "WAYNE ENTERPRISES", "OSCORP", "CYBERDYNE"]
    FIRST_NAMES    = ["JAMES","MARY","JOHN","PATRICIA","ROBERT","JENNIFER","MICHAEL",
                      "LINDA","WILLIAM","BARBARA","DAVID","ELIZABETH","RICHARD","SUSAN"]
    LAST_NAMES     = ["SMITH","JOHNSON","WILLIAMS","BROWN","JONES","GARCIA","MILLER",
                      "DAVIS","WILSON","MOORE","TAYLOR","ANDERSON","THOMAS","JACKSON"]
    CITIES         = ["NEW YORK","LOS ANGELES","CHICAGO","HOUSTON","PHOENIX","SAN ANTONIO",
                      "PHILADELPHIA","SAN DIEGO","DALLAS","SAN JOSE","AUSTIN","JACKSONVILLE"]
    STATES         = ["NY","CA","IL","TX","AZ","PA","FL","OH","MI","GA","NC","WA"]
    CURRENCIES     = ["840","826","978","036","124","392","756","752","208","578"]
    MCC_CODES      = ["5411","5812","5999","7011","5912","5310","5661","5732","5122","4111"]
    MERCHANTS      = ["WALMART STORE","TARGET CORP","AMAZON PRIME","BEST BUY","HOME DEPOT",
                      "KROGER GROCERY","WALGREENS","COSTCO WHOLESALE","MACYS DEPT","WENDYS"]
    GL_SOURCES     = ["Manual","Payroll","Receivables","Payables","Assets","Projects"]
    GL_CATEGORIES  = ["Adjustment","Accrual","Reclass","Close","Opening","Reversal"]

    def generate(self, field: dict) -> str:
        name   = field.get('name', '').upper()
        dtype  = field.get('data_type', 'alphanumeric')
        length = field.get('length', 10)
        rules  = field.get('validation', {})

        # Fixed value shortcut
        if 'value' in field:
            return str(field['value']).ljust(length)[:length]

        # Allowed values – pick one
        if rules.get('allowed'):
            return RNG.choice(rules['allowed'])

        # ── Semantic dispatch ──────────────────────────────────────────
        if re.search(r'ROUTING|RDFI|ODFI|ABA|RTN', name):
            return self._routing_number()

        if re.search(r'ACCOUNT.*NUM|DFI_ACCOUNT', name):
            return self._account_number(length)

        if re.search(r'AMOUNT|AMT|DEBIT|CREDIT|DR$|CR$', name):
            return self._amount(length, rules)

        if re.search(r'DATE', name):
            fmt = field.get('format', 'YYMMDD')
            return self._date(fmt)

        if re.search(r'TIME$', name):
            return '{:02d}{:02d}'.format(RNG.randint(0, 23), RNG.randint(0, 59))

        if re.search(r'INDIVIDUAL_NAME|CARDHOLDER|PERSON', name):
            fn = RNG.choice(self.FIRST_NAMES)
            ln = RNG.choice(self.LAST_NAMES)
            full = f"{fn} {ln}"
            return full[:length].ljust(length)

        if re.search(r'COMPANY_NAME|MERCHANT_NAME', name):
            return RNG.choice(self.COMPANY_NAMES)[:length].ljust(length)

        if re.search(r'MERCHANT_CITY|CITY', name):
            return RNG.choice(self.CITIES)[:length].ljust(length)

        if re.search(r'STATE', name):
            return RNG.choice(self.STATES)[:2]

        if re.search(r'CURRENCY', name):
            return RNG.choice(self.CURRENCIES)[:length]

        if re.search(r'MCC|MERCHANT_CATEGORY', name):
            return RNG.choice(self.MCC_CODES)[:length]

        if re.search(r'MERCHANT(?!_)', name):
            return RNG.choice(self.MERCHANTS)[:length].ljust(length)

        if re.search(r'COUNTRY', name):
            return RNG.choice(["840","826","036","124","392"])[:length]

        if re.search(r'AUTH.*CODE|AUTHORIZATION_CODE', name):
            return ''.join(RNG.choices(string.ascii_uppercase + string.digits, k=6))

        if re.search(r'TRACE', name):
            return str(RNG.randint(10**12, 10**15-1))[:length].zfill(length)

        if re.search(r'BATCH.*NUM|BATCH_NUMBER', name):
            return str(RNG.randint(1, 9999999)).zfill(length)

        if re.search(r'ENTRY.*COUNT|COUNT', name):
            return str(RNG.randint(1, 999)).zfill(length)

        if re.search(r'HASH', name):
            return str(RNG.randint(10**8, 10**10-1))[:length].zfill(length)

        if re.search(r'ZIP|POSTAL', name):
            return str(RNG.randint(10000, 99999))[:length]

        if re.search(r'DESCRIPTION|DESC', name):
            words = ["PAYMENT","INVOICE","SALARY","TRANSFER","REFUND","FEE","INTEREST","SERVICE"]
            return (RNG.choice(words) + " " + str(RNG.randint(1000,9999)))[:length].ljust(length)

        if re.search(r'SEGMENT\d', name):
            return str(RNG.randint(1000, 9999))

        if re.search(r'LEDGER', name):
            return str(RNG.randint(1, 9999)).zfill(4)

        if re.search(r'SOURCE', name):
            return RNG.choice(self.GL_SOURCES)[:length]

        if re.search(r'CATEGORY', name):
            return RNG.choice(self.GL_CATEGORIES)[:length]

        if re.search(r'(FILLER|RESERVED|BLANK)', name) or dtype == 'filler':
            return ' ' * length

        if re.search(r'FLAG|INDICATOR|STATUS', name):
            return RNG.choice(['Y', 'N'])

        # ── Type-based fallback ────────────────────────────────────────
        return self._by_type(dtype, length)

    # ── Primitive generators ──────────────────────────────────────────
    def _routing_number(self) -> str:
        """Generate valid ABA routing number (checksum correct)."""
        weights = [3, 7, 1, 3, 7, 1, 3, 7]
        digits  = [RNG.randint(0, 9) for _ in range(8)]
        chk = (10 - (sum(d * w for d, w in zip(digits, weights)) % 10)) % 10
        return ''.join(map(str, digits)) + str(chk)

    def _account_number(self, length: int) -> str:
        n = RNG.randint(10**(length-1), 10**length - 1)
        return str(n)[:length].ljust(length)

    def _amount(self, length: int, rules: dict) -> str:
        lo = rules.get('min', 100)
        hi = rules.get('max', 99999999)
        val = round(RNG.uniform(lo, hi), 2)
        # Return as zero-padded cents string (NACHA style) or decimal
        cents = int(val * 100)
        s = str(cents).zfill(length)
        return s[:length]

    def _date(self, fmt: str) -> str:
        base = datetime.now()
        delta = timedelta(days=RNG.randint(-30, 30))
        d = base + delta
        mapping = {
            'YYMMDD':      d.strftime('%y%m%d'),
            'YYYYMMDD':    d.strftime('%Y%m%d'),
            'MMDD':        d.strftime('%m%d'),
            'DD-MON-YYYY': d.strftime('%d-%b-%Y').upper(),
            'MMDDYYYY':    d.strftime('%m%d%Y'),
        }
        return mapping.get(fmt, d.strftime('%y%m%d'))

    def _by_type(self, dtype: str, length: int) -> str:
        if dtype == 'numeric':
            return str(RNG.randint(0, 10**length - 1)).zfill(length)[:length]
        elif dtype == 'alpha':
            return ''.join(RNG.choices(string.ascii_uppercase, k=length))
        elif dtype == 'boolean':
            return RNG.choice(['Y', 'N'])
        elif dtype == 'amount':
            return str(RNG.randint(100, 9999999)).zfill(length)[:length]
        else:  # alphanumeric
            chars = string.ascii_uppercase + string.digits
            return ''.join(RNG.choices(chars, k=length))


class Generator:
    """Generates complete test data files for a given spec."""

    def __init__(self, kb):
        self.kb  = kb
        self.fg  = FieldGenerator()

    def generate(
        self,
        spec_name: str,
        num_records: int = 5,
        seed: int | None = None,
        db_data: list[dict] | None = None,
    ) -> str:
        """
        Generate a test data file.

        Args:
            spec_name:   Name of the registered spec.
            num_records: Number of data records (detail lines).
            seed:        Random seed for reproducibility.
            db_data:     Optional list of dicts from DB to use as source values.

        Returns:
            File content as string.
        """
        if seed is not None:
            RNG.seed(seed)

        spec = self.kb.load(spec_name)
        if not spec:
            return f"ERROR: Unknown spec '{spec_name}'"

        fmt = spec.get('format_type', 'custom')

        if fmt == 'nacha':
            return self._gen_nacha(spec, num_records, db_data)
        elif fmt == 'oracle_gl':
            return self._gen_oracle_gl(spec, num_records, db_data)
        elif fmt == 'visa_vcf':
            return self._gen_visa_vcf(spec, num_records, db_data)
        else:
            return self._gen_generic(spec, num_records, db_data)

    # ── NACHA ────────────────────────────────────────────────────────
    def _gen_nacha(self, spec: dict, n: int, db_data) -> str:
        records_def = spec.get('records', {})
        lines = []
        batch_no = RNG.randint(1, 999)

        # File Header (type 1)
        lines.append(self._build_fixed_record(records_def.get('1', {}).get('fields', []), 94))

        # Batch Header (type 5)
        lines.append(self._build_fixed_record(records_def.get('5', {}).get('fields', []), 94))

        # Entry Detail (type 6)
        routing_hash = 0
        total_debit  = 0
        total_credit = 0
        entry_fields = records_def.get('6', {}).get('fields', [])

        for i in range(n):
            row = db_data[i] if db_data and i < len(db_data) else None
            line = self._build_fixed_record_with_db(entry_fields, 94, row)
            lines.append(line)
            # Accumulate hash/totals from generated data
            try:
                rtn = line[3:11]
                routing_hash += int(rtn)
                amt = int(line[29:39])
                tc  = line[1:3]
                if tc in ('22', '32'):
                    total_credit += amt
                else:
                    total_debit  += amt
            except Exception:
                pass

        # Batch Control (type 8)
        bc_fields = records_def.get('8', {}).get('fields', [])
        bc_line   = self._build_fixed_record(bc_fields, 94)
        lines.append(bc_line)

        # File Control (type 9)
        fc_fields = records_def.get('9', {}).get('fields', [])
        fc_line   = self._build_fixed_record(fc_fields, 94)
        lines.append(fc_line)

        # Padding to multiple of 10 (NACHA blocks of 10)
        while len(lines) % 10 != 0:
            lines.append('9' * 94)

        return '\n'.join(lines)

    # ── Oracle GL ────────────────────────────────────────────────────
    def _gen_oracle_gl(self, spec: dict, n: int, db_data) -> str:
        records_def = spec.get('records', {})
        delim = spec.get('delimiter', '|')
        rows  = []

        # Header line (column names)
        hdr_fields = records_def.get('HEADER', {}).get('fields', [])
        rows.append(delim.join(f['name'] for f in hdr_fields))

        # Journal header row
        rows.append(self._build_delimited_record(hdr_fields, delim))

        # Journal lines
        line_fields = records_def.get('LINE', {}).get('fields', [])
        for i in range(n):
            row = db_data[i] if db_data and i < len(db_data) else None
            rows.append(self._build_delimited_record_with_db(line_fields, delim, row))

        return '\n'.join(rows)

    # ── VISA VCF ─────────────────────────────────────────────────────
    def _gen_visa_vcf(self, spec: dict, n: int, db_data) -> str:
        records_def = spec.get('records', {})
        lines = []
        tcr0_fields = records_def.get('TCR0', {}).get('fields', [])

        for i in range(n):
            row = db_data[i] if db_data and i < len(db_data) else None
            lines.append(self._build_fixed_record_with_db(tcr0_fields, 145, row))

        return '\n'.join(lines)

    # ── Generic (custom) ──────────────────────────────────────────────
    def _gen_generic(self, spec: dict, n: int, db_data) -> str:
        fields  = spec.get('fields', [])
        delim   = spec.get('delimiter', ',')
        rows    = []

        # Column header
        rows.append(delim.join(f.get('name', f'FIELD{i}') for i, f in enumerate(fields)))

        for i in range(n):
            row = db_data[i] if db_data and i < len(db_data) else None
            rows.append(self._build_delimited_record_with_db(fields, delim, row))

        return '\n'.join(rows)

    # ── Low-level builders ────────────────────────────────────────────
    def _build_fixed_record(self, fields: list[dict], total_len: int) -> str:
        buf = [' '] * total_len
        for field in fields:
            start  = (field.get('start') or 1) - 1
            length = field.get('length', 1)
            val    = self.fg.generate(field)
            val    = (str(val) + ' ' * length)[:length]
            buf[start:start + length] = list(val)
        return ''.join(buf)

    def _build_fixed_record_with_db(self, fields: list[dict], total_len: int,
                                     db_row: dict | None) -> str:
        if not db_row:
            return self._build_fixed_record(fields, total_len)
        buf = [' '] * total_len
        for field in fields:
            start  = (field.get('start') or 1) - 1
            length = field.get('length', 1)
            fname  = field.get('name', '')
            # Try exact match then case-insensitive
            val    = db_row.get(fname) or db_row.get(fname.lower())
            if val is None:
                val = self.fg.generate(field)
            val = (str(val) + ' ' * length)[:length]
            buf[start:start + length] = list(val)
        return ''.join(buf)

    def _build_delimited_record(self, fields: list[dict], delim: str) -> str:
        return delim.join(
            str(self.fg.generate(f)).strip()
            for f in fields
        )

    def _build_delimited_record_with_db(self, fields: list[dict], delim: str,
                                         db_row: dict | None) -> str:
        if not db_row:
            return self._build_delimited_record(fields, delim)
        parts = []
        for f in fields:
            fname = f.get('name', '')
            val   = db_row.get(fname) or db_row.get(fname.lower())
            if val is None:
                val = self.fg.generate(f)
            parts.append(str(val).strip())
        return delim.join(parts)
