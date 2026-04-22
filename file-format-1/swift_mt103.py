"""
SWIFT MT103 Format Definition
==============================
Built-in spec for SWIFT MT103 Single Customer Credit Transfer messages.
Tag-based format; each field is a SWIFT tag (e.g. :20:, :32A:, :59:).
"""

SWIFT_MT103_SPEC = {
    "name": "SWIFT_MT103",
    "format_type": "swift_mt103",
    "description": "SWIFT MT103 Single Customer Credit Transfer",
    "record_length": "variable",
    "tag_delimiter": ":",
    "line_prefix": ":",
    "records": {
        "MESSAGE": {
            "name": "MT103 Message",
            "fields": [
                {
                    "name": "TAG_20_SENDER_REFERENCE",
                    "tag": "20",
                    "start": None, "end": None,
                    "length": 16,
                    "data_type": "alphanumeric",
                    "required": True,
                    "validation": {
                        "pattern": r'^[A-Z0-9/\-\?:\(\)\.,\' \+]{1,16}$',
                        "no_leading_slash": True,
                        "no_trailing_slash": True,
                    },
                    "description": "Unique transaction reference; max 16 chars; no leading/trailing slash",
                },
                {
                    "name": "TAG_23B_BANK_OPERATION_CODE",
                    "tag": "23B",
                    "start": None, "end": None,
                    "length": 4,
                    "data_type": "alpha",
                    "required": True,
                    "validation": {"allowed": ["CRED", "SPAY", "SSTD", "SPRI"]},
                    "description": "Bank operation code: CRED/SPAY/SSTD/SPRI",
                },
                {
                    "name": "TAG_32A_VALUE_DATE",
                    "tag": "32A_date",
                    "start": None, "end": None,
                    "length": 6,
                    "data_type": "date",
                    "format": "YYMMDD",
                    "required": True,
                    "description": "Value date in YYMMDD format",
                },
                {
                    "name": "TAG_32A_CURRENCY",
                    "tag": "32A_ccy",
                    "start": None, "end": None,
                    "length": 3,
                    "data_type": "alpha",
                    "required": True,
                    "validation": {
                        "allowed": [
                            "USD","EUR","GBP","JPY","CHF","AUD","CAD","SEK",
                            "NOK","DKK","NZD","SGD","HKD","MXN","ZAR","BRL",
                            "INR","CNY","RUB","TRY","AED","SAR","THB","MYR",
                        ]
                    },
                    "description": "ISO 4217 currency code",
                },
                {
                    "name": "TAG_32A_AMOUNT",
                    "tag": "32A_amt",
                    "start": None, "end": None,
                    "length": 15,
                    "data_type": "amount",
                    "required": True,
                    "description": "Amount with comma decimal; e.g. 10000,00",
                },
                {
                    "name": "TAG_50_ORDERING_CUSTOMER_ACCOUNT",
                    "tag": "50K",
                    "start": None, "end": None,
                    "length": 35,
                    "data_type": "alphanumeric",
                    "required": True,
                    "description": "Ordering customer name/address (line 1 may be /account/)",
                },
                {
                    "name": "TAG_57A_ACCOUNT_WITH_INSTITUTION",
                    "tag": "57A",
                    "start": None, "end": None,
                    "length": 11,
                    "data_type": "alphanumeric",
                    "required": False,
                    "validation": {"pattern": r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'},
                    "description": "BIC of beneficiary bank; 8 or 11 chars",
                },
                {
                    "name": "TAG_59_BENEFICIARY_CUSTOMER",
                    "tag": "59",
                    "start": None, "end": None,
                    "length": 35,
                    "data_type": "alphanumeric",
                    "required": True,
                    "description": "Beneficiary name/address (line 1 may be /account/)",
                },
                {
                    "name": "TAG_70_REMITTANCE_INFORMATION",
                    "tag": "70",
                    "start": None, "end": None,
                    "length": 140,
                    "data_type": "alphanumeric",
                    "required": False,
                    "description": "Payment details for beneficiary; up to 4 lines of 35 chars",
                },
                {
                    "name": "TAG_71A_DETAILS_OF_CHARGES",
                    "tag": "71A",
                    "start": None, "end": None,
                    "length": 3,
                    "data_type": "alpha",
                    "required": True,
                    "validation": {"allowed": ["BEN", "SHA", "OUR"]},
                    "description": "Charge bearer: BEN=beneficiary SHA=shared OUR=remitter",
                },
                {
                    "name": "TAG_72_SENDER_TO_RECEIVER_INFO",
                    "tag": "72",
                    "start": None, "end": None,
                    "length": 210,
                    "data_type": "alphanumeric",
                    "required": False,
                    "description": "Up to 6 lines of 35 chars; free-format instructions",
                },
            ]
        }
    },
    "fields": [],
    "vocabulary": {
        "swift": 10, "mt103": 10, "bic": 5, "iban": 5, "credit": 5,
        "transfer": 5, "beneficiary": 5, "ordering": 5, "currency": 4,
        "amount": 4, "charges": 4, "reference": 4, "date": 3,
    },
}

# Flatten for compatibility
_all_swift = []
for _rec in SWIFT_MT103_SPEC["records"].values():
    for _f in _rec["fields"]:
        _fc = dict(_f)
        _fc["record_type"] = "MESSAGE"
        _fc["record_name"] = "MT103 Message"
        _all_swift.append(_fc)
SWIFT_MT103_SPEC["fields"] = _all_swift


class SwiftMT103Generator:
    """Generates realistic SWIFT MT103 messages."""

    import random as _random
    import string as _string
    from datetime import datetime as _dt, timedelta as _td

    CURRENCIES = ["USD", "EUR", "GBP", "CHF", "JPY", "AUD", "CAD", "SGD"]
    CHARGE_CODES = ["SHA", "OUR", "BEN"]
    BICS = [
        "BOFAUS3NXXX", "CHASUS33XXX", "CITIUS33XXX", "WFBIUS6WXXX",
        "NWBKGB2LXXX", "BARCGB22XXX", "DEUTDEDBXXX", "BNPAFRPPXXX",
        "UBSWCHZHXXX", "ANZBNZ22XXX",
    ]
    NAMES = [
        "JOHN DOE", "JANE SMITH", "ACME CORPORATION", "GLOBAL TRADE LTD",
        "FIRST NATIONAL BANK", "PACIFIC RIM HOLDINGS", "CONTINENTAL AG",
        "SUNRISE VENTURES INC", "OCEANIC PARTNERS LLC", "METRO FINANCIAL CORP",
    ]
    CITIES = ["NEW YORK", "LONDON", "FRANKFURT", "ZURICH", "SINGAPORE",
              "SYDNEY", "TORONTO", "PARIS", "TOKYO", "AMSTERDAM"]

    def generate(self, n: int = 5, seed: int | None = None) -> str:
        import random, string
        from datetime import datetime, timedelta
        rng = random.Random(seed)
        messages = []
        for i in range(1, n + 1):
            ref = f"TXN{datetime.now().strftime('%Y%m%d')}{i:04d}"
            ccy = rng.choice(self.CURRENCIES)
            amt = round(rng.uniform(500, 500000), 2)
            amt_str = f"{amt:,.2f}".replace(",", "")
            # SWIFT amount uses comma decimal
            amt_swift = f"{amt_str.replace('.', ',')}"
            val_date = (datetime.now() + timedelta(days=rng.randint(0, 2))).strftime('%y%m%d')
            sender_bic = rng.choice(self.BICS)
            recv_bic   = rng.choice([b for b in self.BICS if b != sender_bic])
            name_ord   = rng.choice(self.NAMES)
            name_ben   = rng.choice([n for n in self.NAMES if n != name_ord])
            city_ord   = rng.choice(self.CITIES)
            city_ben   = rng.choice(self.CITIES)
            acct_ord   = ''.join(rng.choices(string.digits, k=16))
            acct_ben   = ''.join(rng.choices(string.digits, k=16))
            charge     = rng.choice(self.CHARGE_CODES)
            seq = f"{rng.randint(1000, 9999):04d}"
            inv_no = f"INV-{datetime.now().year}-{i:05d}"

            msg = (
                f"{{1:F01{sender_bic}0000000000}}\n"
                f"{{2:I103{recv_bic}N}}\n"
                f"{{3:{{108:REF{seq}}}}}\n"
                f"{{4:\n"
                f":20:{ref}\n"
                f":23B:CRED\n"
                f":32A:{val_date}{ccy}{amt_swift}\n"
                f":50K:/{acct_ord}\n"
                f"{name_ord}\n"
                f"{city_ord}\n"
                f":57A:{recv_bic}\n"
                f":59:/{acct_ben}\n"
                f"{name_ben}\n"
                f"{city_ben}\n"
                f":70:/INV/{inv_no}\n"
                f"PAYMENT FOR SERVICES RENDERED\n"
                f":71A:{charge}\n"
                f"-}}\n"
                f"{{5:{{CHK:{''.join(rng.choices(string.ascii_uppercase + string.digits, k=12))}}}}}"
            )
            messages.append(msg)
        return "\n\n" + ("\n\n---\n\n".join(messages)) + "\n"
