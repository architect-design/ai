# 🏦 Financial LLM Studio

A **custom, from-scratch rule-learning engine** for financial file formats.  
**No external LLM used** (no GPT, Claude, HuggingFace, or any foundation model).

---

## ✨ Capabilities

| Feature | Description |
|---|---|
| 🧠 **Train** | Upload any spec doc (TXT/CSV/PDF/JSON) – engine learns field rules automatically |
| ✅ **Validate** | Validate data files against any trained or built-in spec |
| ⚙️ **Generate** | Produce realistic synthetic test data for any format |
| 🗄️ **Database** | Pull real data from Oracle / Postgres / MySQL / SQLite to drive generation |

---

## 🗂️ Project Structure

```
financial_llm/
├── app.py                        # Streamlit UI (5 tabs)
├── requirements.txt
├── README.md
│
├── core/
│   ├── spec_engine.py            # 🧠 Custom LLM engine
│   │     ├── FinancialTokenizer  # Breaks spec docs into structured tokens
│   │     ├── PatternExtractor    # Extracts field definitions from tokens
│   │     ├── KnowledgeBase       # JSON-based persistent spec store
│   │     ├── InferenceEngine     # TF-IDF cosine similarity for auto-detection
│   │     └── SpecEngine          # Public API: train / get / list / identify
│   │
│   ├── validator.py              # ✅ Rule-based field and file validator
│   │     ├── FieldValidator      # Per-field type/length/domain checks
│   │     └── Validator           # Orchestrates NACHA / GL / VISA / custom
│   │
│   ├── generator.py              # ⚙️ Synthetic test data generator
│   │     ├── FieldGenerator      # Realistic per-field value synthesis
│   │     └── Generator           # Full-file generation (fixed-width + delimited)
│   │
│   └── db_connector.py           # 🗄️ Universal DB connector (SQLAlchemy)
│
├── formats/
│   └── builtin_formats.py        # NACHA, VISA VCF, Oracle GL specs
│
├── models/
│   └── trained_specs/            # JSON files – one per learned spec
│
└── tests/
    └── test_all.py               # 33-test suite (pytest)
```

---

## 🧠 How the Custom LLM Engine Works

This is **not** a neural network. It is a **hybrid statistical + rule-based** system:

```
  Upload Spec Doc
        │
        ▼
  FinancialTokenizer          ← Breaks text into structured token objects
  (line-by-line analysis)       {line, words, numbers, data_type, is_field}
        │
        ▼
  PatternExtractor            ← Regex + heuristics → field definitions
  (5 parsing strategies)        {name, start, end, length, data_type, required, validation}
        │
        ▼
  KnowledgeBase               ← Stores specs as JSON + vocabulary index
  (JSON store)                  models/trained_specs/<SPEC_NAME>.json
        │
        ▼
  InferenceEngine             ← TF-IDF cosine similarity
  (auto-detection)              Matches unknown files to known specs
```

### Field types understood automatically:
`numeric` • `alphanumeric` • `alpha` • `date` • `amount` • `routing_number` • `account_number` • `boolean` • `filler`

### Validation rules extracted:
- Required / Optional
- Allowed values (e.g. transaction codes)
- Date format (YYMMDD, YYYYMMDD, DD-MON-YYYY …)
- Value range (min / max)
- Fixed constant values

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install streamlit pandas sqlalchemy

# For Oracle:
pip install oracledb            # modern thin driver (recommended)
# pip install cx_Oracle         # legacy (needs Oracle Instant Client)

# For PostgreSQL:
pip install psycopg2-binary

# For MySQL:
pip install pymysql
```

### 2. Run the Streamlit UI
```bash
cd financial_llm
streamlit run app.py
```
Open http://localhost:8501

### 3. Use as a Python library
```python
from core.spec_engine  import SpecEngine
from core.validator    import Validator
from core.generator    import Generator
from core.db_connector import DBConnector
from formats.builtin_formats import seed_knowledge_base

# Bootstrap
engine = SpecEngine()
seed_knowledge_base(engine.kb)   # loads NACHA, VISA VCF, Oracle GL

# ── Train a custom spec ───────────────────────────────────────────────
with open("my_spec.txt") as f:
    spec_text = f.read()

spec = engine.train(
    spec_name   = "MY_BANK_FORMAT_V3",
    spec_text   = spec_text,
    format_type = "custom",
    description = "Outbound payment file for First National Bank",
)
print(f"Learned {spec['field_count']} fields")

# ── Validate a data file ──────────────────────────────────────────────
validator = Validator(engine.kb)
with open("data.txt") as f:
    content = f.read()

result = validator.validate("MY_BANK_FORMAT_V3", content)
print(f"Valid: {result.is_valid}  Score: {result.score:.1f}%")
for err in result.errors[:5]:
    print(f"  Rec {err['record']}  {err['field']}: {err['message']}")

# ── Generate test data ────────────────────────────────────────────────
generator = Generator(engine.kb)
output = generator.generate("NACHA", num_records=10, seed=42)
with open("test_nacha.ach", "w") as f:
    f.write(output)

# ── Generate from database ────────────────────────────────────────────
db = DBConnector()
db.connect("postgresql://user:pass@localhost/mydb")
rows = db.fetch("SELECT * FROM gl_entries WHERE journal_id = 1001")
output = generator.generate("ORACLE_GL", num_records=len(rows), db_data=rows)
db.disconnect()
```

---

## 📐 Built-in Format Specs

### NACHA ACH (94-character fixed-width)
Records: File Header (1), Batch Header (5), Entry Detail (6),  
         Addenda (7), Batch Control (8), File Control (9)  
Validates: ABA routing checksums, transaction codes, amounts, dates

### VISA VCF Base II
Record: TCR0 Transaction Core  
Validates: BIN ranges, POS entry modes, MCC codes, currency codes, amounts

### Oracle General Ledger
Records: Journal Header, Journal Line (pipe-delimited)  
Validates: segment combinations, DR/CR amounts, date formats, status codes

---

## 🗄️ Database Connection Strings

| Database | Connection String |
|---|---|
| Oracle (modern) | `oracle+oracledb://user:pass@host:1521/?service_name=ORCL` |
| Oracle (legacy) | `oracle+cx_Oracle://user:pass@host:1521/SID` |
| PostgreSQL | `postgresql://user:pass@host:5432/mydb` |
| MySQL | `mysql+pymysql://user:pass@host:3306/mydb` |
| SQL Server | `mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server` |
| SQLite | `sqlite:///path/to/db.sqlite3` |

---

## 🧪 Running Tests

```bash
cd financial_llm
pip install pytest
pytest tests/test_all.py -v
```

Expected: **32+ tests passing**

---

## 📋 UI Tab Reference

| Tab | Purpose |
|---|---|
| 🧠 Train Spec | Upload/paste spec doc → engine learns field rules |
| ✅ Validate File | Upload data file → validate against any spec |
| ⚙️ Generate Data | Pick spec + row count → download synthetic test file |
| 🗄️ Database | Connect Oracle/PG/MySQL, browse tables, run SQL, feed generator |
| 📋 Spec Browser | View all specs, field tables, raw JSON, delete |

---

## 🔒 Security Notes

- Database passwords are **never stored to disk** (only masked display string kept in memory)
- SQL queries are **row-limited** automatically (configurable `max_rows`)
- Table names are **sanitised** before use in dynamic SQL
- All spec files stored as plain JSON in `models/trained_specs/`
