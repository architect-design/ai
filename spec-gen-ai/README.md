# SpecGenAI — Specification-Driven Generative AI System

A production-ready, **zero-LLM** system that learns file structures from uploaded specifications and generates valid synthetic test data on demand.  Supports **VISA VCF**, **ACH/NACHA**, custom **JSON Schema**, and raw **sample data** files.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Folder Structure](#folder-structure)
3. [How It Works](#how-it-works)
4. [Quick Start](#quick-start)
5. [REST API Reference](#rest-api-reference)
6. [Specification Format](#specification-format)
7. [Layer Descriptions](#layer-descriptions)
8. [Custom ML / Inference Logic](#custom-ml--inference-logic)
9. [Running Tests](#running-tests)
10. [Design Decisions](#design-decisions)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST Layer                               │
│  POST /upload  │  POST /train  │  GET /specs  │  POST /generate        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                        Service Layer                                    │
│                     SpecGenAIService                                    │
│  upload_file() │ train_spec() │ list_specs() │ generate_file()         │
└──────┬──────────────┬──────────────────────────────┬────────────────────┘
       │              │                              │
┌──────▼──────┐ ┌─────▼──────────────┐ ┌────────────▼──────────────────┐
│  Storage    │ │   Training Pipeline │ │      Generation Pipeline      │
│  Service    │ │                     │ │                               │
│  ─────────  │ │  ┌──────────────┐  │ │  ┌────────────────────────┐  │
│  Uploads    │ │  │   Parsers    │  │ │  │  Generator (ACH/VCF/   │  │
│  Specs      │ │  │  ACH/VCF/    │  │ │  │  JSON)                 │  │
│  Outputs    │ │  │  JSON/Sample │  │ │  │  DataSynthesizer       │  │
└─────────────┘ │  └──────┬───────┘  │ │  └────────────────────────┘  │
                │         │          │ │                               │
                │  ┌──────▼───────┐  │ │  ┌────────────────────────┐  │
                │  │  SpecLearner │  │ │  │   ValidationEngine     │  │
                │  │  FieldInfer  │  │ │  │                        │  │
                │  │  PatternDet  │  │ │  └────────────────────────┘  │
                │  └──────┬───────┘  │ └───────────────────────────────┘
                │         │          │
                │  ┌──────▼───────┐  │
                │  │  RuleEngine  │  │
                │  │  (deps +     │  │
                │  │  constraints)│  │
                │  └──────────────┘  │
                └────────────────────┘
                         │
                ┌────────▼────────┐
                │   SchemaModel   │  ←  The central artefact
                │ (JSON on disk)  │     specs/{uuid}.json
                └─────────────────┘
```

---

## Folder Structure

```
spec_gen_ai/
├── app/
│   ├── main.py                     # FastAPI app factory + lifespan
│   ├── core/
│   │   ├── config.py               # Pydantic-settings configuration
│   │   └── exceptions.py           # Custom exception hierarchy
│   ├── models/
│   │   ├── schema.py               # Internal SchemaModel domain objects
│   │   └── requests.py             # Pydantic API request/response models
│   ├── parsers/
│   │   ├── __init__.py             # Parser factory: get_parser(spec_type)
│   │   ├── base_parser.py          # Abstract BaseParser
│   │   ├── ach_parser.py           # ACH/NACHA 94-char fixed-width parser
│   │   ├── vcf_parser.py           # VISA VCF / Base II parser
│   │   ├── json_schema_parser.py   # JSON Schema + SpecGenAI native format
│   │   └── sample_data_parser.py   # Statistical inference from raw data
│   ├── learner/
│   │   ├── field_inferrer.py       # Multi-strategy field type inference
│   │   ├── pattern_detector.py     # Sequence/batch pattern discovery
│   │   └── spec_learner.py         # Orchestrates full training pipeline
│   ├── rule_engine/
│   │   └── rule_engine.py          # Field/record/file rules + dep graph
│   ├── generator/
│   │   ├── data_synthesizer.py     # Type-aware, seeded data generation
│   │   └── base_generator.py       # ACH / VCF / JSON generators + factory
│   ├── validation/
│   │   └── validator.py            # ValidationEngine + ValidationReport
│   ├── storage/
│   │   └── storage_service.py      # File-system persistence layer
│   ├── services/
│   │   └── spec_service.py         # Business logic orchestration
│   └── api/
│       └── routes/
│           └── main_router.py      # All FastAPI route handlers
├── tests/
│   └── test_core.py                # 58 unit tests (100% pass)
├── uploads/                        # Runtime: uploaded raw files
├── specs/                          # Runtime: trained SchemaModel JSON files
├── outputs/                        # Runtime: generated output files
├── sample_spec.json                # Example SpecGenAI native spec format
├── requirements.txt
└── README.md
```

---

## How It Works

### Training Flow

```
User uploads spec file(s)                   → POST /upload  → upload_id
User submits train request with upload_ids  → POST /train

  For each uploaded file:
    1. Parser.parse_file()      → partial SchemaModel
       ├── ACHParser:       reads 94-char NACHA format + standard field defs
       ├── VCFParser:       reads VISA Base II field set + optional tagged data
       ├── JSONSchemaParser: reads SpecGenAI native JSON or JSON Schema draft
       └── SampleDataParser: delimiter detection → column extraction → inference

    2. SpecLearner.train()
       ├── Merge multiple SchemaModels (if >1 file)
       ├── FieldInferrer.enrich() per field:
       │     Strategy 1 – Regex pattern voting (14 patterns, weighted)
       │     Strategy 2 – Statistical distribution (cardinality, length)
       │     Strategy 3 – Field name heuristics (name-to-type mapping)
       │     Strategy 4 – Checksum validation (Luhn, ABA/mod10)
       │     → Best type wins by weighted vote; confidence 0–1
       ├── PatternDetector.detect_from_file()
       │     → Topological ordering, batch pattern, repeating record types
       ├── Dependency graph build (checksum ← data, trace ← sequence)
       └── Constant/default value inference

    3. Persist SchemaModel as specs/{uuid}.json
```

### Generation Flow

```
User submits generate request with spec_id  → POST /generate

  1. Load SchemaModel from specs/{uuid}.json
  2. get_generator(model) → ACHGenerator | VCFGenerator | JSONGenerator
  3. For each record in ordered_records():
       RuleEngine.resolve_field_order() → topological field ordering
       For each field (dependency order):
         DataSynthesizer.generate(field):
           CONSTANT   → literal value
           SEQUENCE   → auto-increment counter
           ENUM       → random choice from allowed_values
           PAN        → Luhn-valid 16-digit VISA number
           EXPIRY     → future YYMM date
           ROUTING    → real ABA routing number
           ACCOUNT    → random digits, length-constrained
           DATE       → recent past date in spec format
           AMOUNT     → constrained integer in cents
           STRING     → name-aware realistic values
           CHECKSUM   → placeholder → post-process
  4. ACH: compute batch hashes, pad to 10-record blocks
     VCF: compute trailer totals
  5. ValidationEngine.validate(records) → ValidationReport
  6. Save to outputs/{gen_id}.ext  (if output_format=file)
```

---

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 3. End-to-end example (curl)

#### Upload a spec file

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@sample_spec.json" \
  -F "spec_type=json"
# → {"upload_id": "abc-123", ...}
```

#### Train the system

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "upload_ids": ["abc-123"],
    "spec_name": "PaymentProcessor",
    "spec_type": "json",
    "description": "Internal payment file spec"
  }'
# → {"spec_id": "xyz-789", "records_learned": 3, "fields_learned": 19, ...}
```

#### Generate 50 synthetic records

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "spec_id": "xyz-789",
    "record_count": 50,
    "seed": 42,
    "output_format": "file"
  }'
# → {"generation_id": "gen-456", "output_path": "...", "validation_passed": true}
```

#### Download the generated file

```bash
curl -O http://localhost:8000/api/v1/download/gen-456
```

#### ACH example — upload a native ACH file

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@my_payroll.ach" \
  -F "spec_type=ach"

curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "upload_ids": ["ach-upload-id"],
    "spec_name": "PayrollACH",
    "spec_type": "ach"
  }'

curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"spec_id": "...", "record_count": 100, "output_format": "preview"}'
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/api/v1/stats` | Storage statistics |
| `POST` | `/api/v1/upload` | Upload spec or sample file |
| `POST` | `/api/v1/train` | Parse + learn → SchemaModel |
| `GET`  | `/api/v1/specs` | List all trained specs |
| `GET`  | `/api/v1/specs/{id}` | Get full spec detail |
| `DELETE` | `/api/v1/specs/{id}` | Delete a spec |
| `POST` | `/api/v1/generate` | Generate synthetic data |
| `GET`  | `/api/v1/download/{gen_id}` | Download generated file |

### POST /upload

**Form fields:**
- `file` (binary) – the spec or sample file
- `spec_type` (string) – one of: `vcf`, `ach`, `json`, `sample`

**Response:**
```json
{
  "upload_id": "3f1e2d...",
  "filename": "my_spec.json",
  "spec_type": "json",
  "file_size_bytes": 4096,
  "message": "File uploaded successfully. Use upload_id='...' in train request."
}
```

### POST /train

**Body:**
```json
{
  "upload_ids": ["3f1e2d..."],
  "spec_name": "MySpec",
  "spec_type": "json",
  "description": "Optional description",
  "override_existing": false
}
```

**Response:**
```json
{
  "spec_id": "a1b2c3...",
  "spec_name": "MySpec",
  "spec_type": "json",
  "records_learned": 3,
  "fields_learned": 19,
  "inference_stats": {
    "total_fields": 19,
    "enriched_fields": 11,
    "total_records": 3,
    "pattern_ordering": ["HDR", "TXN", "TRL"]
  },
  "trained_at": "2024-01-15T12:00:00",
  "message": "Training complete."
}
```

### POST /generate

**Body:**
```json
{
  "spec_id": "a1b2c3...",
  "record_count": 100,
  "seed": 42,
  "output_format": "file",
  "overrides": {
    "odfi_routing": "021000021"
  }
}
```

**output_format values:**
- `file` – saves to disk, returns `output_path`
- `preview` – returns first 20 lines in `preview_lines`
- `json` – returns structured records in `payload`

---

## Specification Format

### SpecGenAI Native JSON Format

See `sample_spec.json` for a complete example. Key structure:

```json
{
  "spec_name": "MyPaymentFile",
  "spec_type": "json",
  "file_structure": {
    "format": "delimited",        // "fixed_width" | "delimited" | "json" | "tagged"
    "delimiter": "|",
    "encoding": "utf-8",
    "file_extension": ".psv",
    "header_records": ["HDR"],
    "detail_records": ["TXN"],
    "trailer_records": ["TRL"]
  },
  "records": [
    {
      "record_type_id": "TXN",
      "name": "Transaction",
      "category": "detail",       // "header" | "detail" | "trailer" | "batch_header" | ...
      "repeatable": true,
      "ordering": 10,
      "fields": [
        {
          "name": "pan",
          "type": "pan",           // see field types below
          "length": 16,
          "required": true
        }
      ]
    }
  ]
}
```

### Supported Field Types

| Type | Description | Auto-Generated As |
|------|-------------|-------------------|
| `string` | Free text | Name-aware realistic strings |
| `numeric` | Integer | Random integer in min/max range |
| `amount` | Money/cents | Padded integer |
| `date` | Date | Recent past date in format_string format |
| `datetime` | Date+time | ISO 8601 datetime |
| `boolean` | True/false | `true` / `false` |
| `pan` | Payment card number | Luhn-valid 16-digit VISA number |
| `cvv` | Card security code | 3-digit number |
| `expiry` | Card expiry | Future YYMM date |
| `routing_number` | ABA routing | Real valid ABA routing number |
| `account_number` | Bank account | Random digits |
| `enum` | Fixed value set | Random choice from `allowed_values` |
| `sequence` | Auto-increment | 1, 2, 3, ... |
| `constant` | Fixed literal | Always `default_value` |
| `checksum` | Computed checksum | Post-processed by generator |
| `computed` | Derived field | Formula-based (extensible) |
| `alphanumeric` | A-Z + 0-9 | Random uppercase alphanumeric |

---

## Layer Descriptions

### Parser Layer (`app/parsers/`)

Each parser converts raw file bytes into a **partial** `SchemaModel`. The parsers know the structural grammar of their format but defer statistical enrichment to the learner.

- **ACHParser** – Embeds the complete NACHA standard field definitions for all 6 record types (1, 5, 6, 7, 8, 9). Handles both native 94-char ACH files and JSON spec overrides.
- **VCFParser** – Embeds Visa Base II clearing record fields with PAN/expiry/CVV/amount/MCC constraints. Detects tagged `KEY=VALUE` format automatically.
- **JSONSchemaParser** – Handles three JSON modes: SpecGenAI native, JSON Schema draft-07/2020, or flat sample arrays.
- **SampleDataParser** – Pure statistical inference: delimiter detection (pipe/tab/comma/fixed-width via character-position entropy), column extraction, type voting.

### Learner Layer (`app/learner/`)

- **FieldInferrer** – Multi-strategy voting system (regex + statistical + name heuristics + checksum validation). Each strategy produces weighted votes; the type with the highest vote-share wins. Includes Luhn (card numbers) and ABA (routing numbers) checksum validation as strong inference signals.
- **PatternDetector** – Kahn's algorithm topological sort for field dependency ordering. Greedy walk through transition matrix for file-level record ordering detection.
- **SpecLearner** – Orchestrates parse → merge → enrich → pattern-detect → dependency-build → finalise.

### Rule Engine (`app/rule_engine/`)

Three rule levels:
1. **FieldRule** – constraint validation (min/max length, allowed values, pattern, Luhn, ABA)
2. **RecordRule** – cross-field validation (fixed-width total length)
3. **FileRule** – aggregate validation (batch counts, file totals)

Dependency resolution uses **Kahn's topological sort** — fields with no dependencies come first; dependent fields follow in order.

### Generator Layer (`app/generator/`)

- **DataSynthesizer** – Seeded `random.Random` for reproducibility. Generates Luhn-valid PANs, real ABA routing numbers, future expiry dates, realistic merchant names, ISO-format dates.
- **ACHGenerator** – Produces fully spec-compliant 94-char NACHA files with correct batch headers, entry detail records, batch controls, file control, and 10-record block padding.
- **VCFGenerator** – Produces fixed-width VCF files with realistic card transaction data, header and trailer with total counts/amounts.
- **JSONGenerator** – Produces JSON arrays or pipe-delimited files from any custom spec.

### Validation Layer (`app/validation/`)

`ValidationEngine` validates generated (or uploaded) record sets against all `FieldConstraints` derived from the `SchemaModel`. Produces structured `ValidationReport` with per-field error details, error counts, and a pass/fail summary.

---

## Custom ML / Inference Logic

This system implements all "learning" with pure Python — **no OpenAI, no HuggingFace, no external LLMs**.

### Multi-Strategy Type Inference (FieldInferrer)

```python
# 4 independent inference strategies, each casting weighted votes:

# Strategy 1: Regex pattern matching
# 14 regex patterns with tuned weights; dominant type suppresses overlapping generics
votes[DATE] += match_rate * 3.0  # 8-digit numeric pattern → date

# Strategy 2: Statistical distribution
# Cardinality ratio for ENUM detection, length consistency for fixed-width hints
if len(unique_vals) / n < 0.3: votes[ENUM] += 1.2

# Strategy 3: Field name heuristics
# Keyword matching: "pan" | "card_number" → PAN bonus 2.0
# "routing" | "aba" → ROUTING_NUMBER bonus 2.0

# Strategy 4: Checksum validation
# Luhn check on all sample values → PAN confidence +2.0
# ABA check on all sample values → ROUTING_NUMBER confidence +2.0

# Winner: highest vote share becomes inferred type + confidence score
best_type = max(votes, key=lambda t: votes[t])
confidence = votes[best_type] / sum(votes.values())
```

### Pattern Detection (PatternDetector)

```python
# Kahn's topological sort for field generation ordering
# Greedy walk through record-type transition matrix for file ordering
# Batch pattern detection: find [batch_header ... batch_trailer] sub-sequences
```

### Column Boundary Detection (SampleDataParser)

```python
# For fixed-width files: character-position entropy analysis
# Positions where space frequency > 80% across all rows = column boundary
space_freq[i] = count(rows where row[i] == ' ') / total_rows
boundaries = positions where space_freq > 0.8
```

---

## Running Tests

```bash
# All 58 unit tests
pytest tests/test_core.py -v

# With coverage report
pytest tests/test_core.py --cov=app --cov-report=term-missing

# Single test class
pytest tests/test_core.py::TestACHGenerator -v

# Single test
pytest tests/test_core.py::TestDataSynthesizer::test_pan_is_luhn_valid -v
```

**Test coverage includes:**
- Schema model serialisation roundtrip
- All 4 parsers (ACH, VCF, JSON, Sample)
- Field inferrer (date, PAN, enum, routing, amount, format string)
- Pattern detector (ordering, batch pattern, repeating types)
- Rule engine (allowed values, min length, dependency resolution)
- Data synthesizer (Luhn validity, sequence counters, enum sampling, ABA validity, seed reproducibility)
- ACH generator (94-char lines, 10-record block padding, record count)
- VCF generator (header/trailer, record count)
- JSON generator (JSON array output, factory dispatch)
- Validation engine (valid records pass, invalid amounts caught, report structure)

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| No external LLM | Deterministic, auditable, works offline, no API cost |
| SchemaModel as central artefact | Single source of truth; parsers produce it, generators consume it |
| JSON persistence | Human-readable, diff-friendly, no database dependency |
| Seeded random generation | Reproducible test data for regression testing |
| Voting-based type inference | More robust than single-rule classification; handles ambiguous data |
| Kahn's topological sort for deps | Prevents circular dependency deadlocks at generation time |
| Real ABA routing numbers | Downstream systems often validate routing numbers; fake ones fail |
| Luhn-valid PANs | Card processing systems validate Luhn; invalid PANs are rejected |
| NACHA 10-record block padding | ACH files must be padded to multiples of 10; omitting this breaks bank systems |
| Per-format generators | ACH/VCF have complex aggregate rules (hash totals, counts) that generic generators can't handle |
