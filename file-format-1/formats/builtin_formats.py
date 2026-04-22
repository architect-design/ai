"""
Built-in Financial Format Definitions
=======================================
NACHA ACH, VISA VCF, and Oracle GL format specs
expressed as structured dicts – used to seed the KnowledgeBase
on first run and as ground-truth references for validation.
"""

# ════════════════════════════════════════════════════════════════════
# NACHA  ACH File Format  (NACHA Operating Rules)
# ════════════════════════════════════════════════════════════════════
NACHA_SPEC = {
    "name": "NACHA",
    "format_type": "nacha",
    "description": "NACHA ACH (Automated Clearing House) fixed-width 94-character record format",
    "record_length": 94,
    "line_delimiter": "\n",
    "records": {
        "1": {
            "name": "File Header Record",
            "fields": [
                {"name": "RECORD_TYPE_CODE",    "start": 1,  "end": 1,  "length": 1,  "data_type": "numeric",       "value": "1",    "required": True},
                {"name": "PRIORITY_CODE",        "start": 2,  "end": 3,  "length": 2,  "data_type": "numeric",       "required": True, "default": "01"},
                {"name": "IMMEDIATE_DESTINATION","start": 4,  "end": 13, "length": 10, "data_type": "routing_number","required": True},
                {"name": "IMMEDIATE_ORIGIN",     "start": 14, "end": 23, "length": 10, "data_type": "numeric",       "required": True},
                {"name": "FILE_CREATION_DATE",   "start": 24, "end": 29, "length": 6,  "data_type": "date",          "format": "YYMMDD","required": True},
                {"name": "FILE_CREATION_TIME",   "start": 30, "end": 33, "length": 4,  "data_type": "numeric",       "required": False},
                {"name": "FILE_ID_MODIFIER",     "start": 34, "end": 34, "length": 1,  "data_type": "alphanumeric",  "required": True},
                {"name": "RECORD_SIZE",          "start": 35, "end": 37, "length": 3,  "data_type": "numeric",       "value": "094","required": True},
                {"name": "BLOCKING_FACTOR",      "start": 38, "end": 39, "length": 2,  "data_type": "numeric",       "value": "10", "required": True},
                {"name": "FORMAT_CODE",          "start": 40, "end": 40, "length": 1,  "data_type": "numeric",       "value": "1",  "required": True},
                {"name": "IMMEDIATE_DESTINATION_NAME","start":41,"end":63,"length":23,"data_type":"alphanumeric","required":True},
                {"name": "IMMEDIATE_ORIGIN_NAME","start": 64, "end": 86, "length": 23, "data_type": "alphanumeric",  "required": True},
                {"name": "REFERENCE_CODE",       "start": 87, "end": 94, "length": 8,  "data_type": "alphanumeric",  "required": False},
            ]
        },
        "5": {
            "name": "Batch Header Record",
            "fields": [
                {"name": "RECORD_TYPE_CODE",     "start": 1,  "end": 1,  "length": 1,  "data_type": "numeric",      "value": "5",  "required": True},
                {"name": "SERVICE_CLASS_CODE",   "start": 2,  "end": 4,  "length": 3,  "data_type": "numeric",      "required": True,
                 "validation": {"allowed": ["200","220","225"]}},
                {"name": "COMPANY_NAME",         "start": 5,  "end": 20, "length": 16, "data_type": "alphanumeric", "required": True},
                {"name": "COMPANY_DISCRETIONARY_DATA","start":21,"end":40,"length":20,"data_type":"alphanumeric","required":False},
                {"name": "COMPANY_IDENTIFICATION","start":41,"end":50, "length": 10, "data_type": "alphanumeric",   "required": True},
                {"name": "STANDARD_ENTRY_CLASS_CODE","start":51,"end":53,"length":3, "data_type": "alpha",          "required": True,
                 "validation": {"allowed": ["PPD","CCD","CTX","WEB","TEL","ARC","BOC","POP","RCK","ENR"]}},
                {"name": "COMPANY_ENTRY_DESCRIPTION","start":54,"end":63,"length":10,"data_type":"alphanumeric","required":True},
                {"name": "COMPANY_DESCRIPTIVE_DATE","start":64,"end":69,"length":6, "data_type": "date",            "required": False},
                {"name": "EFFECTIVE_ENTRY_DATE", "start": 70, "end": 75, "length": 6,  "data_type": "date",         "format": "YYMMDD","required": True},
                {"name": "SETTLEMENT_DATE",      "start": 76, "end": 78, "length": 3,  "data_type": "numeric",      "required": False},
                {"name": "ORIGINATOR_STATUS_CODE","start":79,"end":79, "length": 1,  "data_type": "numeric",        "required": True},
                {"name": "ODFI_IDENTIFICATION",  "start": 80, "end": 87, "length": 8,  "data_type": "routing_number","required": True},
                {"name": "BATCH_NUMBER",         "start": 88, "end": 94, "length": 7,  "data_type": "numeric",      "required": True},
            ]
        },
        "6": {
            "name": "Entry Detail Record",
            "fields": [
                {"name": "RECORD_TYPE_CODE",     "start": 1,  "end": 1,  "length": 1,  "data_type": "numeric",      "value": "6",  "required": True},
                {"name": "TRANSACTION_CODE",     "start": 2,  "end": 3,  "length": 2,  "data_type": "numeric",      "required": True,
                 "validation": {"allowed": ["22","23","24","27","28","29","32","33","34","37","38","39"]}},
                {"name": "RDFI_ROUTING_TRANSIT", "start": 4,  "end": 11, "length": 8,  "data_type": "routing_number","required": True},
                {"name": "CHECK_DIGIT",          "start": 12, "end": 12, "length": 1,  "data_type": "numeric",      "required": True},
                {"name": "DFI_ACCOUNT_NUMBER",   "start": 13, "end": 29, "length": 17, "data_type": "account_number","required": True},
                {"name": "AMOUNT",               "start": 30, "end": 39, "length": 10, "data_type": "amount",       "required": True},
                {"name": "INDIVIDUAL_IDENTIFICATION_NUMBER","start":40,"end":54,"length":15,"data_type":"alphanumeric","required":False},
                {"name": "INDIVIDUAL_NAME",      "start": 55, "end": 76, "length": 22, "data_type": "alphanumeric", "required": True},
                {"name": "DISCRETIONARY_DATA",   "start": 77, "end": 78, "length": 2,  "data_type": "alphanumeric", "required": False},
                {"name": "ADDENDA_RECORD_INDICATOR","start":79,"end":79,"length":1,  "data_type": "numeric",        "required": True,
                 "validation": {"allowed": ["0","1"]}},
                {"name": "TRACE_NUMBER",         "start": 80, "end": 94, "length": 15, "data_type": "numeric",      "required": True},
            ]
        },
        "8": {
            "name": "Batch Control Record",
            "fields": [
                {"name": "RECORD_TYPE_CODE",     "start": 1,  "end": 1,  "length": 1,  "data_type": "numeric",  "value": "8",  "required": True},
                {"name": "SERVICE_CLASS_CODE",   "start": 2,  "end": 4,  "length": 3,  "data_type": "numeric",  "required": True},
                {"name": "ENTRY_ADDENDA_COUNT",  "start": 5,  "end": 10, "length": 6,  "data_type": "numeric",  "required": True},
                {"name": "ENTRY_HASH",           "start": 11, "end": 20, "length": 10, "data_type": "numeric",  "required": True},
                {"name": "TOTAL_DEBIT_AMOUNT",   "start": 21, "end": 32, "length": 12, "data_type": "amount",   "required": True},
                {"name": "TOTAL_CREDIT_AMOUNT",  "start": 33, "end": 44, "length": 12, "data_type": "amount",   "required": True},
                {"name": "COMPANY_IDENTIFICATION","start":45,"end":54, "length": 10, "data_type": "alphanumeric","required": True},
                {"name": "MESSAGE_AUTHENTICATION_CODE","start":55,"end":73,"length":19,"data_type":"alphanumeric","required":False},
                {"name": "RESERVED",             "start": 74, "end": 79, "length": 6,  "data_type": "filler",   "required": False},
                {"name": "ODFI_IDENTIFICATION",  "start": 80, "end": 87, "length": 8,  "data_type": "routing_number","required": True},
                {"name": "BATCH_NUMBER",         "start": 88, "end": 94, "length": 7,  "data_type": "numeric",  "required": True},
            ]
        },
        "9": {
            "name": "File Control Record",
            "fields": [
                {"name": "RECORD_TYPE_CODE",     "start": 1,  "end": 1,  "length": 1,  "data_type": "numeric",  "value": "9",  "required": True},
                {"name": "BATCH_COUNT",          "start": 2,  "end": 7,  "length": 6,  "data_type": "numeric",  "required": True},
                {"name": "BLOCK_COUNT",          "start": 8,  "end": 13, "length": 6,  "data_type": "numeric",  "required": True},
                {"name": "ENTRY_ADDENDA_COUNT",  "start": 14, "end": 21, "length": 8,  "data_type": "numeric",  "required": True},
                {"name": "ENTRY_HASH",           "start": 22, "end": 31, "length": 10, "data_type": "numeric",  "required": True},
                {"name": "TOTAL_DEBIT_AMOUNT",   "start": 32, "end": 43, "length": 12, "data_type": "amount",   "required": True},
                {"name": "TOTAL_CREDIT_AMOUNT",  "start": 44, "end": 55, "length": 12, "data_type": "amount",   "required": True},
                {"name": "RESERVED",             "start": 56, "end": 94, "length": 39, "data_type": "filler",   "required": False},
            ]
        },
    },
    "fields": [],   # flattened below
    "vocabulary": {},
}

# Flatten all fields for compatibility with the engine
_all_nacha = []
for _rt, _rec in NACHA_SPEC["records"].items():
    for _f in _rec["fields"]:
        _fc = dict(_f)
        _fc["record_type"] = _rt
        _fc["record_name"] = _rec["name"]
        _all_nacha.append(_fc)
NACHA_SPEC["fields"] = _all_nacha


# ════════════════════════════════════════════════════════════════════
# VISA  VCF  (Visa Card Format / Base II)
# ════════════════════════════════════════════════════════════════════
VISA_VCF_SPEC = {
    "name": "VISA_VCF",
    "format_type": "visa_vcf",
    "description": "VISA VCF (Visa Card Format) Base II transaction clearing file",
    "record_length": "variable",
    "records": {
        "TCR0": {
            "name": "Transaction Core Record 0",
            "fields": [
                {"name": "TRANSACTION_CODE",      "start": 1,  "end": 2,  "length": 2,  "data_type": "numeric",      "required": True,
                 "validation": {"allowed": ["05","06","08","25","26","28"]}},
                {"name": "FUNCTION_CODE",         "start": 3,  "end": 5,  "length": 3,  "data_type": "numeric",      "required": True},
                {"name": "MESSAGE_REASON_CODE",   "start": 6,  "end": 9,  "length": 4,  "data_type": "numeric",      "required": False},
                {"name": "CARD_ACCEPTOR_BIN",     "start": 10, "end": 15, "length": 6,  "data_type": "numeric",      "required": True},
                {"name": "ACQUIRER_COUNTRY_CODE", "start": 16, "end": 18, "length": 3,  "data_type": "numeric",      "required": True},
                {"name": "ACQUIRER_CURRENCY_CODE","start": 19, "end": 21, "length": 3,  "data_type": "numeric",      "required": True},
                {"name": "ACCOUNT_NUMBER",        "start": 22, "end": 35, "length": 14, "data_type": "numeric",      "required": True},
                {"name": "ACCOUNT_NUMBER_EXT",    "start": 36, "end": 41, "length": 6,  "data_type": "numeric",      "required": False},
                {"name": "PURCHASE_DATE",         "start": 42, "end": 45, "length": 4,  "data_type": "date",         "format": "MMDD","required": True},
                {"name": "TRANSACTION_AMOUNT",    "start": 46, "end": 57, "length": 12, "data_type": "amount",       "required": True},
                {"name": "TRANSACTION_CURRENCY",  "start": 58, "end": 60, "length": 3,  "data_type": "numeric",      "required": True},
                {"name": "CARDHOLDER_AMOUNT",     "start": 61, "end": 72, "length": 12, "data_type": "amount",       "required": False},
                {"name": "CARDHOLDER_CURRENCY",   "start": 73, "end": 75, "length": 3,  "data_type": "numeric",      "required": False},
                {"name": "AUTHORIZATION_CODE",    "start": 76, "end": 81, "length": 6,  "data_type": "alphanumeric", "required": True},
                {"name": "AUTHORIZATION_DATE",    "start": 82, "end": 85, "length": 4,  "data_type": "date",         "format": "MMDD","required": False},
                {"name": "MERCHANT_NAME",         "start": 86, "end": 110,"length": 25, "data_type": "alphanumeric", "required": True},
                {"name": "MERCHANT_CITY",         "start": 111,"end": 123,"length": 13, "data_type": "alphanumeric", "required": True},
                {"name": "MERCHANT_COUNTRY_CODE", "start": 124,"end": 126,"length": 3,  "data_type": "numeric",      "required": True},
                {"name": "MERCHANT_ZIP",          "start": 127,"end": 135,"length": 9,  "data_type": "alphanumeric", "required": False},
                {"name": "MERCHANT_STATE",        "start": 136,"end": 137,"length": 2,  "data_type": "alpha",        "required": False},
                {"name": "MERCHANT_CATEGORY_CODE","start": 138,"end": 141,"length": 4,  "data_type": "numeric",      "required": True},
                {"name": "POS_ENTRY_MODE",        "start": 142,"end": 143,"length": 2,  "data_type": "numeric",      "required": True,
                 "validation": {"allowed": ["01","02","03","05","07","10","90","91"]}},
            ]
        },
    },
    "fields": [],
    "vocabulary": {},
}

_all_visa = []
for _rt, _rec in VISA_VCF_SPEC["records"].items():
    for _f in _rec["fields"]:
        _fc = dict(_f)
        _fc["record_type"] = _rt
        _fc["record_name"] = _rec["name"]
        _all_visa.append(_fc)
VISA_VCF_SPEC["fields"] = _all_visa


# ════════════════════════════════════════════════════════════════════
# Oracle  GL  Journal Entry  flat-file format
# ════════════════════════════════════════════════════════════════════
ORACLE_GL_SPEC = {
    "name": "ORACLE_GL",
    "format_type": "oracle_gl",
    "description": "Oracle General Ledger Journal Entry Interface flat-file format",
    "delimiter": "|",
    "has_header": True,
    "records": {
        "HEADER": {
            "name": "Header Record",
            "fields": [
                {"name": "STATUS",               "length": 1,  "data_type": "alpha",        "required": True,
                 "validation": {"allowed": ["N","P","C","E"]}},
                {"name": "LEDGER_ID",            "length": 15, "data_type": "alphanumeric",  "required": True},
                {"name": "ACCOUNTING_DATE",      "length": 11, "data_type": "date",          "format": "DD-MON-YYYY","required": True},
                {"name": "CURRENCY_CODE",        "length": 15, "data_type": "alpha",         "required": True},
                {"name": "DATE_CREATED",         "length": 11, "data_type": "date",          "format": "DD-MON-YYYY","required": False},
                {"name": "CREATED_BY",           "length": 100,"data_type": "alphanumeric",  "required": False},
                {"name": "ACTUAL_FLAG",          "length": 1,  "data_type": "alpha",         "required": True,
                 "validation": {"allowed": ["A","B","E"]}},
                {"name": "USER_JE_SOURCE_NAME",  "length": 25, "data_type": "alphanumeric",  "required": True},
                {"name": "USER_JE_CATEGORY_NAME","length": 25, "data_type": "alphanumeric",  "required": True},
                {"name": "ENCUMBRANCE_TYPE_ID",  "length": 15, "data_type": "numeric",       "required": False},
                {"name": "BUDGET_VERSION_ID",    "length": 15, "data_type": "numeric",       "required": False},
                {"name": "BALANCED_JE_FLAG",     "length": 1,  "data_type": "alpha",         "required": False,
                 "validation": {"allowed": ["Y","N"]}},
                {"name": "BALANCING_SEGMENT_VALUE","length":25,"data_type":"alphanumeric",   "required": False},
            ]
        },
        "LINE": {
            "name": "Journal Line Record",
            "fields": [
                {"name": "STATUS",               "length": 1,  "data_type": "alpha",         "required": True},
                {"name": "EFFECTIVE_DATE",        "length": 11, "data_type": "date",          "format": "DD-MON-YYYY","required": True},
                {"name": "CODE_COMBINATION_ID",  "length": 15, "data_type": "numeric",       "required": False},
                {"name": "SEGMENT1",             "length": 25, "data_type": "alphanumeric",  "required": True,  "label": "Company"},
                {"name": "SEGMENT2",             "length": 25, "data_type": "alphanumeric",  "required": True,  "label": "Department"},
                {"name": "SEGMENT3",             "length": 25, "data_type": "alphanumeric",  "required": True,  "label": "Account"},
                {"name": "SEGMENT4",             "length": 25, "data_type": "alphanumeric",  "required": False, "label": "Sub-Account"},
                {"name": "SEGMENT5",             "length": 25, "data_type": "alphanumeric",  "required": False, "label": "Product"},
                {"name": "ENTERED_DR",           "length": 22, "data_type": "amount",        "required": False},
                {"name": "ENTERED_CR",           "length": 22, "data_type": "amount",        "required": False},
                {"name": "ACCOUNTED_DR",         "length": 22, "data_type": "amount",        "required": False},
                {"name": "ACCOUNTED_CR",         "length": 22, "data_type": "amount",        "required": False},
                {"name": "DESCRIPTION",          "length": 240,"data_type": "alphanumeric",  "required": False},
                {"name": "ATTRIBUTE1",           "length": 150,"data_type": "alphanumeric",  "required": False},
                {"name": "ATTRIBUTE2",           "length": 150,"data_type": "alphanumeric",  "required": False},
                {"name": "REFERENCE1",           "length": 100,"data_type": "alphanumeric",  "required": False},
                {"name": "REFERENCE2",           "length": 100,"data_type": "alphanumeric",  "required": False},
            ]
        },
    },
    "fields": [],
    "vocabulary": {},
}

_all_gl = []
for _rt, _rec in ORACLE_GL_SPEC["records"].items():
    for _f in _rec["fields"]:
        _fc = dict(_f)
        _fc["record_type"] = _rt
        _fc["record_name"] = _rec["name"]
        _all_gl.append(_fc)
ORACLE_GL_SPEC["fields"] = _all_gl


# ════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════
BUILTIN_FORMATS: dict[str, dict] = {
    "NACHA":      NACHA_SPEC,
    "VISA_VCF":   VISA_VCF_SPEC,
    "ORACLE_GL":  ORACLE_GL_SPEC,
}


def seed_knowledge_base(kb) -> None:
    """Seed the KnowledgeBase with all built-in formats (idempotent)."""
    for name, spec in BUILTIN_FORMATS.items():
        if not kb.load(name):
            kb.save(name, spec)
