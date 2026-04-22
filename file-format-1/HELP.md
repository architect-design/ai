# 📖 HELP.md — Financial LLM Studio: A Beginner's Complete Guide

> **Who is this for?**  
> This guide is written for someone who is new to AI, machine learning, and Python.  
> Every concept is explained from scratch. No prior knowledge is assumed.  
> Read it top to bottom the first time, then use it as a reference later.

---

## 📋 Table of Contents

1. [What Is This Project? (Plain English)](#1-what-is-this-project-plain-english)
2. [What Is an LLM? And Why Did We Build Our Own?](#2-what-is-an-llm-and-why-did-we-build-our-own)
3. [The Big Picture — How All the Files Connect](#3-the-big-picture--how-all-the-files-connect)
4. [Project Folder Structure Explained](#4-project-folder-structure-explained)
5. [File 1: `core/spec_engine.py` — The Brain](#5-file-1-corespec_enginepy--the-brain)
6. [File 2: `core/validator.py` — The Rule Checker](#6-file-2-corevalidatorpy--the-rule-checker)
7. [File 3: `core/generator.py` — The Data Factory](#7-file-3-coregeneratorpy--the-data-factory)
8. [File 4: `core/db_connector.py` — The Database Bridge](#8-file-4-coredb_connectorpy--the-database-bridge)
9. [File 5: `core/audit_log.py` — The Event Recorder](#9-file-5-coreaudit_logpy--the-event-recorder)
10. [File 6: `formats/builtin_formats.py` — Built-in Format Definitions](#10-file-6-formatsbuiltin_formatspy--built-in-format-definitions)
11. [File 7: `formats/swift_mt103.py` — SWIFT Format](#11-file-7-formatsswift_mt103py--swift-format)
12. [File 8: `app.py` — The User Interface](#12-file-8-apppy--the-user-interface)
13. [File 9: `api/rest_api.py` — The Web Service](#13-file-9-apirest_apipy--the-web-service)
14. [File 10: `cli/fllm.py` — The Command Line Tool](#14-file-10-clifllmpy--the-command-line-tool)
15. [File 11: `tests/test_all.py` — The Quality Checker](#15-file-11-teststest_allpy--the-quality-checker)
16. [Sample Data Files Explained](#16-sample-data-files-explained)
17. [Key Concepts Glossary](#17-key-concepts-glossary)
18. [Step-by-Step: What Happens When You Click "Train"](#18-step-by-step-what-happens-when-you-click-train)
19. [Step-by-Step: What Happens When You Click "Validate"](#19-step-by-step-what-happens-when-you-click-validate)
20. [Step-by-Step: What Happens When You Click "Generate"](#20-step-by-step-what-happens-when-you-click-generate)
21. [How to Run the Project](#21-how-to-run-the-project)
22. [Common Questions (FAQ)](#22-common-questions-faq)

---

## 1. What Is This Project? (Plain English)

Imagine you work at a bank or a payment company. Every day, thousands of financial
files are exchanged between banks and companies. These files have very strict rules:

- A NACHA payment file must be **exactly 94 characters per line**
- A VISA transaction file must have a **routing number with a valid checksum**
- An Oracle GL file must use **pipe characters (`|`) to separate fields**

This project is a **software tool** that can:

| What you want to do | How this tool helps |
|---|---|
| Teach the system a new file format | **Train** — upload a spec document, it learns the rules |
| Check if a file follows the rules | **Validate** — upload the file, it finds errors |
| Create fake test data | **Generate** — pick a format, it makes realistic sample files |
| Read data from a real database | **Database** — connect to Oracle/SQL and pull real data |

**The key thing:** This tool does all of this **without using ChatGPT, Claude, or any
other commercial AI service**. It builds its own understanding of file formats from
the specification documents you give it.

---

## 2. What Is an LLM? And Why Did We Build Our Own?

### What is an LLM (Large Language Model)?

An LLM is a type of AI that has read billions of web pages, books, and code, and
learned patterns from all that text. When you ask it a question, it predicts what
the most helpful answer looks like based on those patterns.

Examples: ChatGPT (by OpenAI), Claude (by Anthropic), Gemini (by Google).

### Why didn't we just use one of those?

1. **Cost** — Commercial LLMs charge per API call. Processing millions of financial
   records every day would cost a lot of money.

2. **Privacy** — Financial data is sensitive. Sending bank account numbers and
   routing numbers to a third-party AI service is a security risk.

3. **Accuracy** — General-purpose LLMs don't know your specific company's custom
   file format. A purpose-built engine trained on your exact spec will always be
   more precise.

4. **Control** — You own this code. You can run it on your own servers, offline,
   behind a firewall.

### What did we build instead?

We built a **rule-learning engine** — a program that:
1. Reads a spec document (like a bank's file format guide)
2. Extracts the rules from it automatically (field names, sizes, data types)
3. Stores those rules
4. Uses the rules to validate, generate, or identify files

This is **not** a neural network. It does **not** need a GPU. It runs on any
laptop. Think of it like a very smart reader that has learned to understand
specification documents.

---

## 3. The Big Picture — How All the Files Connect

Here is a map of how all the pieces talk to each other:

```
                    ┌─────────────────────────────────┐
                    │     YOU (the user)              │
                    │  browser / terminal / API call  │
                    └────────┬────────────────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │         app.py  (Streamlit UI)  │   ← the web page you see
              │         api/rest_api.py          │   ← or call via HTTP
              │         cli/fllm.py              │   ← or use from terminal
              └──────────────┬──────────────────┘
                             │ calls
        ┌────────────────────▼─────────────────────────┐
        │                 CORE ENGINE                   │
        │                                               │
        │  spec_engine.py   ← trains & identifies       │
        │  validator.py     ← checks data files         │
        │  generator.py     ← creates test data         │
        │  db_connector.py  ← reads from databases      │
        │  audit_log.py     ← records everything        │
        └────────────────────┬─────────────────────────┘
                             │ reads/writes
        ┌────────────────────▼──────────────────┐
        │             STORAGE                   │
        │                                       │
        │  models/trained_specs/*.json          │  ← learned format rules
        │  models/audit.ndjson                  │  ← event history
        │  formats/builtin_formats.py           │  ← pre-built formats
        │  formats/swift_mt103.py               │  ← SWIFT MT103 format
        └───────────────────────────────────────┘
```

**The flow in one sentence:**
> You upload a spec → the engine reads it → stores the rules → later uses those
> rules to validate or generate files.

---

## 4. Project Folder Structure Explained

```
financial_llm/              ← The root folder of the whole project
│
├── app.py                  ← The web UI (7 tabs you see in the browser)
├── requirements.txt        ← List of Python libraries to install
├── README.md               ← Short project summary
├── HELP.md                 ← This file! (the guide you are reading)
│
├── core/                   ← The "brain" of the system
│   ├── spec_engine.py      ← Reads specs and learns from them
│   ├── validator.py        ← Checks data files for errors
│   ├── generator.py        ← Creates synthetic (fake) test data
│   ├── db_connector.py     ← Connects to real databases
│   └── audit_log.py        ← Records every action taken
│
├── formats/                ← Pre-built format knowledge
│   ├── builtin_formats.py  ← NACHA, VISA VCF, Oracle GL definitions
│   └── swift_mt103.py      ← SWIFT MT103 definition
│
├── api/
│   └── rest_api.py         ← HTTP API (for calling from other programs)
│
├── cli/
│   └── fllm.py             ← Command-line tool (use from terminal)
│
├── models/
│   └── trained_specs/      ← Where learned specs are saved as .json files
│
├── sample_data/            ← Example files for testing
│   ├── nacha_spec.txt      ← Full NACHA format spec document
│   ├── oracle_gl_spec.txt  ← Oracle GL spec document
│   ├── swift_mt103_spec.txt← SWIFT MT103 spec document
│   ├── sample_nacha.ach    ← A real-looking NACHA data file
│   └── sample_oracle_gl.txt← A real-looking Oracle GL data file
│
└── tests/
    └── test_all.py         ← Automated tests (43 tests, 42 pass)
```

---

## 5. File 1: `core/spec_engine.py` — The Brain

### What does this file do?

This is the most important file. It contains the "learning engine" — the code
that reads a format specification document and understands the rules in it.

It contains **5 classes** that work together like an assembly line:

```
Spec document text
       │
       ▼
  FinancialTokenizer     ← Step 1: Break text into structured pieces
       │
       ▼
  PatternExtractor       ← Step 2: Find field definitions in those pieces
       │
       ▼
  KnowledgeBase          ← Step 3: Save the learned rules to disk
       │
  (later, for search:)
       ▼
  InferenceEngine        ← Step 4: Match unknown files to known specs
       │
       ▼
  SpecEngine             ← Step 5: The public face — ties it all together
```

---

### Class 1: `FinancialTokenizer`

**What is a "token"?**  
In language processing, a "token" is a meaningful unit of text — like a word,
a number, or a symbol. "Tokenizing" means breaking text apart into these units.

**What does this class do?**  
It reads the spec document line by line and turns each line into a structured
Python dictionary with useful information extracted.

**Example input (one line from a spec):**
```
AMOUNT   30-39   10   N   Required
```

**Example output (what the tokenizer produces):**
```python
{
    'line':      30,           # line number in the document
    'raw':       'AMOUNT   30-39   10   N   Required',
    'words':     ['amount', 'n', 'required'],   # all words, lowercase
    'numbers':   [30, 39, 10],                  # all numbers found
    'data_type': 'numeric',                     # inferred from 'N'
    'is_field':  True                           # yes, this looks like a field def
}
```

**Key methods explained:**

```python
def tokenize(self, text: str) -> list[dict]:
```
> This is the main method. You give it the whole spec document as a string.
> It splits it into lines, processes each line, and returns a list of token
> dictionaries. Empty lines and lines starting with `#` (comments) are skipped.

```python
def _words(self, text: str) -> list[str]:
```
> Finds all words in a line (letters and underscores only) and returns them
> as lowercase. Example: `"AMOUNT 30"` → `['amount']`

```python
def _numbers(self, text: str) -> list[int]:
```
> Finds all numbers in a line. Example: `"AMOUNT 30-39 10"` → `[30, 39, 10]`

```python
def _infer_type(self, text: str) -> str:
```
> Looks at the words in a line and guesses the data type.
> If it finds the word "routing" or "aba" → returns `'routing_number'`
> If it finds "amount" or "amt" → returns `'amount'`
> If it finds "date" → returns `'date'`
> Default if nothing matches → returns `'alphanumeric'`

```python
def _is_field_line(self, text: str) -> bool:
```
> Decides whether a line looks like it defines a field.
> It tries several regex patterns. If any match → returns `True`.
> As a fallback: if the line has both numbers AND words AND is short → `True`.

**What is a regex (regular expression)?**  
A regex is a mini-language for describing patterns in text.
- `\d+` means "one or more digits"  
- `[A-Za-z]+` means "one or more letters"
- `\b(amount|amt)\b` means "the exact word 'amount' or 'amt'"

---

### Class 2: `PatternExtractor`

**What does this class do?**  
It takes the list of tokens from the Tokenizer and tries to extract real
field definitions — the name, position, length, type, and validation rules.

**Key method:**

```python
def extract_fields(self, tokens: list[dict], format_type: str) -> list[dict]:
```
> Goes through each token. If it's marked `is_field=True`, it tries to parse
> it into a proper field definition dictionary.

**The three parsing strategies in `_parse_field()`:**

**Strategy 1: Fixed-width table format**
```
AMOUNT   30-39   10   N
```
> Looks for: name, then position range (e.g. 30-39), then length, then type.
> This is the most common format for NACHA and VISA VCF specs.

**Strategy 2: Positional list format**
```
1   RECORD_TYPE   1   N
2   COMPANY_ID   10   AN
```
> Looks for: a position number, then name, then length, then type.

**Strategy 3: Key-value format**
```
TRANSACTION_CODE: 2-char numeric field
```
> Looks for: `UPPERCASE_NAME:` followed by a description.

**What is `_extract_validation()`?**  
After finding a field, this method scans the same line for extra rules:
- If it finds "range: 0-999" → sets `min=0, max=999`
- If it finds "values: 22 27 32" → sets `allowed=['22','27','32']`
- If it finds "format: YYMMDD" → sets `format='YYMMDD'`

These become the validation rules used later.

---

### Class 3: `KnowledgeBase`

**What does this class do?**  
It saves and loads learned specs from disk. Each spec is stored as a `.json`
file in the `models/trained_specs/` folder.

**Why JSON?**  
JSON (JavaScript Object Notation) is a simple text format for storing
structured data. It's human-readable and Python can read/write it easily.

**A saved spec file looks like this:**
```json
{
  "name": "NACHA",
  "format_type": "nacha",
  "fields": [
    {
      "name": "RECORD_TYPE_CODE",
      "start": 1,
      "end": 1,
      "length": 1,
      "data_type": "numeric",
      "required": true
    },
    ...
  ],
  "vocabulary": { "record": 15, "type": 12, "amount": 8, ... },
  "_saved_at": "2026-04-20T10:30:00"
}
```

**Methods:**
- `save(spec_name, spec)` → writes spec to `models/trained_specs/SPEC_NAME.json`
- `load(spec_name)` → reads and returns the spec from disk
- `list_specs()` → returns names of all `.json` files in the folder
- `delete(spec_name)` → deletes the file from disk

---

### Class 4: `InferenceEngine`

**What does this class do?**  
Given an unknown file, it figures out which known spec best matches it.
This is the "auto-detect format" feature.

**How does it work? (TF-IDF Cosine Similarity)**

This sounds complicated but the idea is simple:

**Step 1: Build a "vocabulary" for each known spec.**  
Count how often each word appears in the field names and descriptions.
Example for NACHA spec:
```
{ "routing": 8, "amount": 6, "record": 10, "batch": 5, "entry": 7, ... }
```

**Step 2: Build a "vocabulary" for the unknown file.**  
Count the words in the file:
```
{ "record": 3, "batch": 2, "amount": 4, "routing": 3, ... }
```

**Step 3: Compute similarity.**  
The math is called "cosine similarity". Think of each vocabulary as a list
of numbers (a vector). The similarity is how much the two vectors "point in
the same direction". Score = 1.0 means identical, 0.0 means nothing in common.

```python
def _score(self, text: str, spec: dict) -> float:
    # Multiply matching word counts, then normalize
    total = sum(spec_terms[word] * text_terms[word] for word in both)
    norm  = sqrt(sum of spec_terms squared) * sqrt(sum of text_terms squared)
    return total / norm
```

**Result:** Each known spec gets a score. The highest score = best match.

---

### Class 5: `SpecEngine` (the public API)

**What does this class do?**  
It ties all the other classes together into a single easy-to-use object.
This is what `app.py`, `rest_api.py`, and `fllm.py` all use.

**Methods:**

```python
engine = SpecEngine()
```
> Creates a new engine. Automatically creates all the internal pieces.

```python
engine.train("MY_SPEC", spec_text, format_type="nacha", description="...")
```
> 1. Tokenizes the spec text
> 2. Extracts field definitions
> 3. Builds vocabulary
> 4. Computes a checksum (fingerprint) of the document
> 5. Saves everything to disk
> Returns a dictionary with the results.

```python
engine.get_spec("MY_SPEC")
```
> Loads and returns a spec by name. Returns `None` if not found.

```python
engine.list_specs()
```
> Returns a list of all known spec names.

```python
engine.identify(some_file_content)
```
> Runs the TF-IDF matching and returns ranked matches.

---

## 6. File 2: `core/validator.py` — The Rule Checker

### What does this file do?

It checks whether a real data file follows the rules of a spec.
Think of it like a spell-checker, but for financial file formats.

It contains **2 main classes**: `FieldValidator` and `Validator`.

---

### Class: `ValidationResult`

This is a simple container that collects the results of a validation run.

```python
result = ValidationResult()
result.add_error(record_number=3, field="AMOUNT", message="Must be numeric", value="ABC")
result.add_pass()
print(result.score)    # e.g. 97.5 (percentage of checks that passed)
print(result.is_valid) # True if zero errors
```

**What it stores:**
- `errors` — list of problems found (record number, field name, message, bad value)
- `warnings` — list of non-fatal issues
- `passed` — count of successful checks
- `failed` — count of failed checks
- `total_records` — how many records were checked

---

### Class: `FieldValidator`

**What does this class do?**  
Given a single field value and the rules for that field, it decides if the
value is valid. This is the most detailed level of checking.

**The `validate()` method:**

```python
fv.validate(value="ABC", field={"name":"AMOUNT","data_type":"numeric",...}, result, record_no=5)
```

It runs these checks in order:

**Check 1: Presence**
```python
if req and value.strip() == '':
    result.add_error(...)   # required field is empty!
```

**Check 2: Length**
```python
if len(value) > field['length']:
    result.add_error(...)   # too long!
```

**Check 3: Data type**
- `numeric` → must contain only digits 0-9
- `alpha` → must contain only letters A-Z
- `amount` → must look like a number, e.g. "1234.56"
- `date` → must match the expected date format
- `routing_number` → must pass the ABA checksum algorithm

**Check 4: Domain rules**
```python
if 'allowed' in rules and value not in rules['allowed']:
    result.add_error(...)   # not in list of valid values!
```

**What is the ABA Routing Number Checksum?**

Every US bank routing number has a built-in error-detection code.
The last digit is calculated from the first 8 digits using this formula:
```
(3×d1 + 7×d2 + d3 + 3×d4 + 7×d5 + d6 + 3×d7 + 7×d8 + d9) mod 10 = 0
```
If this equation does not equal 0, the routing number is invalid.
Our validator runs this check automatically on any field of type `routing_number`.

---

### Class: `Validator`

**What does this class do?**  
It orchestrates the validation of a complete file. It looks at the format
type and calls the right validation strategy.

**Three validation strategies:**

**1. NACHA fixed-width (`_validate_nacha`)**
```python
for each line in the file:
    check: line is exactly 94 characters
    look at first character (record type: 1,5,6,7,8,9)
    find the matching field definitions for that record type
    check every field by position (start:end)
```

**2. Delimited (`_validate_delimited`)**
```python
for each line in the file:
    split line by the delimiter (e.g. "|")
    match each piece to the corresponding field definition
    validate each piece
```

**3. Auto-detect (`_validate_fixed_or_delimited`)**
```python
look at the first 500 characters of the file
if the delimiter character appears in that sample → use delimited mode
otherwise → use fixed-width mode
```

---

## 7. File 3: `core/generator.py` — The Data Factory

### What does this file do?

It creates realistic fake financial data files. You say "give me 10 NACHA
records" and it creates a properly formatted NACHA file with realistic
(but invented) data.

It contains **2 classes**: `FieldGenerator` and `Generator`.

---

### Class: `FieldGenerator`

**What does this class do?**  
Given the definition of a single field, it creates a realistic value for it.

It uses the field name to decide what kind of data to generate. For example:

| If field name contains... | It generates... |
|---|---|
| `ROUTING` or `ABA` or `RTN` | A valid ABA routing number (passes checksum) |
| `AMOUNT` or `AMT` | A realistic dollar amount like `0000001250` |
| `DATE` | A recent date in the right format |
| `INDIVIDUAL_NAME` | A realistic person name like `JOHN SMITH` |
| `COMPANY_NAME` or `MERCHANT_NAME` | A company name like `ACME CORP` |
| `TRANSACTION_CODE` | A valid code from the allowed list |
| `FILLER` or `RESERVED` | Spaces |
| Anything else | Random letters/numbers of the right length |

**The routing number generator:**
```python
def _routing_number(self) -> str:
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    digits  = [random 0-9 for each of 8 positions]
    # calculate check digit so the checksum formula equals 0
    chk = (10 - (sum(digit × weight) mod 10)) mod 10
    return first 8 digits + check digit
```
This ensures every generated routing number actually passes validation!

**The date generator:**
```python
def _date(self, fmt: str) -> str:
    # Pick a random date within ±30 days of today
    # Format it the way the spec requires
    formats = {
        'YYMMDD':      '260415',
        'YYYYMMDD':    '20260415',
        'MMDD':        '0415',
        'DD-MON-YYYY': '15-APR-2026',
    }
```

---

### Class: `Generator`

**What does this class do?**  
It builds a complete file by calling `FieldGenerator` for each field in each
record.

**Four format-specific builders:**

**1. `_gen_nacha()` — builds a NACHA ACH file**
```
File Header record (type 1)        ← always first
Batch Header record (type 5)       ← starts a batch
Entry Detail record (type 6) × N  ← N = the number you requested
Batch Control record (type 8)      ← closes the batch
File Control record (type 9)       ← always last
Padding lines of "9"×94            ← pad to multiple of 10 lines (NACHA rule)
```

**2. `_gen_oracle_gl()` — builds an Oracle GL file**
```
Header row (column names)
Journal Header record
Journal Line records × N
```

**3. `_gen_visa_vcf()` — builds a VISA VCF file**
```
One TCR0 record per transaction × N
```

**4. `_gen_generic()` — builds any custom format**
```
Header row (column names)
Data rows × N
```

**Using database data:**
```python
output = generator.generate("NACHA", num_records=5, db_data=real_rows)
```
If `db_data` is provided, the generator first looks in the database row for
a matching field name. If found, it uses the real value. If not found,
it falls back to generating a synthetic value.

---

## 8. File 4: `core/db_connector.py` — The Database Bridge

### What does this file do?

It connects to real databases (Oracle, PostgreSQL, MySQL, SQLite) so the
system can read real financial data to use as source material for generation.

### What is SQLAlchemy?

SQLAlchemy is a Python library that lets you talk to many different databases
using the same Python code. You just change the "connection string" and it
handles the differences between Oracle, MySQL, PostgreSQL, etc.

### Connection strings

A connection string tells the system: which database type, which server,
which user, which password, and which database name.

```
oracle+oracledb://myuser:mypassword@myserver:1521/?service_name=ORCL
│       │         │       │          │         │    │
│       │         │       │          server    port service
│       │         user    password
│       driver
type
```

Examples:
```
sqlite:///myfile.db                            ← local SQLite file (simplest)
postgresql://user:pass@localhost:5432/mydb     ← PostgreSQL
mysql+pymysql://user:pass@localhost:3306/mydb  ← MySQL
oracle+oracledb://user:pass@host:1521/?service_name=ORCL  ← Oracle
```

### Key methods:

```python
db = DBConnector()
db.connect("sqlite:///mydb.db")     # connect to a database
rows = db.fetch("SELECT * FROM gl_lines WHERE amount > 1000")  # run a query
tables = db.list_tables()           # see what tables exist
schema = db.describe_table("GL_LINES")  # see the column names and types
db.disconnect()                     # close the connection
```

**Password masking:**  
The connector never stores your real password. In logs and displays, it shows
`****` instead. Example: `oracle://user:****@host/db`

**Row limiting:**  
The connector automatically adds a row limit to prevent accidentally
downloading millions of records. Default is 1,000 rows max.

**Mock data:**  
If you are not connected to a real database, you can use:
```python
mock_rows = db.get_mock_data("nacha", n=10)
```
This generates realistic fake data in the shape the format expects, useful
for testing without a real database.

---

## 9. File 5: `core/audit_log.py` — The Event Recorder

### What does this file do?

Every time someone does something with the system — trains a spec, validates
a file, generates data, runs a query — the audit log writes a record of it.

### Why keep an audit log?

In financial systems, you always need to be able to answer:
- Who trained which spec, and when?
- How many files have been validated today?
- What was the average validation score last week?
- Was this database query run, and what did it return?

### The log file format: NDJSON

The audit log is stored as NDJSON (Newline-Delimited JSON). Each line is a
valid JSON object:

```
{"ts":"2026-04-20T10:30:00","action":"TRAIN","spec":"NACHA","field_count":94}
{"ts":"2026-04-20T10:31:00","action":"VALIDATE","spec":"NACHA","score":98.5}
{"ts":"2026-04-20T10:32:00","action":"GENERATE","spec":"NACHA","rows":10}
```

This format is great for audit logs because:
- It can grow to millions of lines without slowing down
- Each line is independent (one corrupt line won't break the rest)
- Easy to filter and process with standard tools

### Key methods:

```python
audit = AuditLog()

# Record events
audit.record_training("NACHA", field_count=94, format_type="nacha")
audit.record_validation("NACHA", "payroll.ach", is_valid=True, score=99.2)
audit.record_generation("NACHA", rows=10, seed=42)
audit.record_db_connect("oracle", "oracle://****@host", success=True)

# Read events
entries = audit.query(action="VALIDATE", limit=50)   # last 50 validations
stats   = audit.stats()   # summary: total events, avg score, etc.
audit.clear()             # delete all log entries
```

### Thread safety

The audit log uses a Python `threading.Lock`. This means if two people use
the system at the same time, their log writes won't collide or corrupt each
other. Only one write happens at a time.

---

## 10. File 6: `formats/builtin_formats.py` — Built-in Format Definitions

### What does this file do?

Instead of making you train the NACHA, VISA VCF, and Oracle GL formats from
scratch, this file provides them pre-built. These are written by hand to be
100% accurate representations of the real-world standards.

### The three built-in formats:

**1. NACHA (`NACHA_SPEC`)**

NACHA is the organization that runs the ACH (Automated Clearing House) network —
the system that handles direct deposits, bill payments, and payroll in the US.

The file format is **fixed-width**: every line is exactly 94 characters.
No delimiters (no commas or pipes). Position matters.

Example NACHA line:
```
622021000021123456789012345  0000150000JONES SARAH          EMP001      0021000020000001
│  │         │                │          │                    │            │
│  RDFI      Account number   Amount     Individual name      Discr.data  Trace number
Transaction code
```

The file defines all 5 record types (1, 5, 6, 8, 9), each with their fields,
positions, lengths, and validation rules.

**2. VISA VCF (`VISA_VCF_SPEC`)**

VCF stands for VISA Card Format. This is used for clearing card transactions
between the merchant's bank and the cardholder's bank.

The file defines the TCR0 (Transaction Core Record) with fields like:
- Transaction code (what type of transaction)
- Card number (masked for security)
- Merchant name and city
- Amount and currency
- POS (Point of Sale) entry mode

**3. Oracle GL (`ORACLE_GL_SPEC`)**

Oracle GL is the General Ledger module of Oracle's ERP (Enterprise Resource
Planning) software. Companies use it to record all their financial transactions.

The file is **pipe-delimited** (`|` between each field).
It has two record types:
- **HEADER** — one per journal batch (metadata about the journal)
- **LINE** — one per accounting entry (the actual debit or credit)

Every LINE must have either `ENTERED_DR` (debit) or `ENTERED_CR` (credit),
not both. And all debits in a journal must equal all credits (double-entry
accounting rule).

### The `seed_knowledge_base()` function

```python
def seed_knowledge_base(kb) -> None:
    for name, spec in BUILTIN_FORMATS.items():
        if not kb.load(name):    # only if not already saved
            kb.save(name, spec)
```

This function is called at startup. It checks if the built-in formats are
already saved. If not, it saves them. This means the system always has
NACHA, VISA VCF, and Oracle GL available even before you train anything.

---

## 11. File 7: `formats/swift_mt103.py` — SWIFT Format

### What is SWIFT MT103?

SWIFT is the international network that banks use to send money across borders.
MT103 is the message type used for a "Single Customer Credit Transfer" —
basically an international wire transfer.

Unlike NACHA (which is fixed-width) or Oracle GL (which is pipe-delimited),
SWIFT messages use **tags** — each field starts with a colon and a tag number:

```
:20:TXN20260415001         ← Tag 20 = Sender's Reference
:23B:CRED                  ← Tag 23B = Bank Operation Code
:32A:260415USD25000,00     ← Tag 32A = Value Date + Currency + Amount
:50K:/GB29NWBK60161331926819  ← Tag 50K = Ordering Customer
JOHN DOE
LONDON EC1A 1BB
:59:/US64BOFA0001234567890 ← Tag 59 = Beneficiary
JANE SMITH
NEW YORK NY 10001
:71A:SHA                   ← Tag 71A = Charges (SHA = shared)
```

### Key fields explained:

| Tag | Name | What it means |
|---|---|---|
| `:20:` | Sender Reference | Unique ID for this transfer (max 16 chars) |
| `:23B:` | Bank Op Code | CRED=normal, SPAY=priority |
| `:32A:` | Value/Currency/Amount | Date + ISO currency + amount with comma decimal |
| `:50K:` | Ordering Customer | Who is sending the money |
| `:57A:` | Account With Institution | The beneficiary's bank (BIC code) |
| `:59:` | Beneficiary | Who receives the money |
| `:71A:` | Charges | SHA=split, OUR=sender pays all, BEN=receiver pays all |

### The `SwiftMT103Generator` class

Because SWIFT's tag-based format is very different from fixed-width or
delimited formats, it has its own dedicated generator class.

```python
sg = SwiftMT103Generator()
output = sg.generate(n=5, seed=42)
```

It creates complete multi-block SWIFT messages with realistic:
- BIC codes (bank identifier codes like `BOFAUS3NXXX`)
- Account numbers
- Transaction references
- Amounts with SWIFT-format comma decimal (e.g. `25000,00` not `25000.00`)
- Charge codes

---

## 12. File 8: `app.py` — The User Interface

### What does this file do?

This is the web application you see when you run `streamlit run app.py` and
open your browser to `http://localhost:8501`.

### What is Streamlit?

Streamlit is a Python library that lets you build web applications with
very little code. Instead of writing HTML, CSS, and JavaScript, you write
Python code and Streamlit turns it into a web page.

```python
st.text_input("Spec Name")        # creates a text input box
st.button("Train")                # creates a clickable button
st.dataframe(my_dataframe)        # displays a table
st.download_button("Download")    # creates a download button
```

### The 7 tabs explained:

**Tab 1: 🧠 Train Spec**  
This is where you teach the system about a new file format.
- Choose a format family (NACHA, Oracle GL, etc.)
- Upload your spec document OR paste the text OR use a built-in sample
- Click "Train Engine"
- The system reads the document, extracts fields, and saves the spec

**Tab 2: ✅ Validate File**  
Check if a real data file follows the rules.
- Choose which spec to validate against
- Upload your data file OR paste its contents OR use a sample
- Click "Validate"
- See a score (e.g. 97.5%), a list of errors, and a list of warnings
- Download the full report as a JSON file

**Tab 3: ⚙️ Generate Data**  
Create fake test data files.
- Choose which format to generate
- Choose how many records (1–1000)
- Choose a random seed (same seed = same output every time)
- Optionally pull source values from a connected database
- Click "Generate"
- Download the generated file

**Tab 4: 🔍 Auto-Detect**  
Upload an unknown file and the system figures out what format it is.
- Upload the file or paste its contents
- Click "Identify Format"
- See ranked matches with similarity scores
- Optionally quick-validate against the best match

**Tab 5: 🗄️ Database**  
Connect to a real database.
- Choose the database type (Oracle, PostgreSQL, MySQL, SQLite)
- Enter your connection string
- Browse tables, view schemas, run SQL queries
- Results are automatically available in the Generate tab

**Tab 6: 📊 Audit Log**  
See the history of everything that has been done.
- Filter by action type (TRAIN, VALIDATE, GENERATE, etc.)
- Filter by spec name
- See statistics (total events, average validation score, etc.)
- Export the log as JSON

**Tab 7: 📋 Spec Browser**  
Browse all registered specs.
- See the field table for any spec
- See the raw JSON data
- Download the spec as a JSON file
- Delete a spec you no longer need

### What is `@st.cache_resource`?

```python
@st.cache_resource
def get_engine():
    e = SpecEngine()
    seed_knowledge_base(e.kb)
    return e
```

This is a "decorator" — Python's way of wrapping a function with extra behaviour.
`@st.cache_resource` means: "run this function once, save the result, and reuse
it every time it's called instead of running it again."

Without this, every time you click a button, it would create a brand new engine,
losing any trained specs. With it, the same engine is reused for the whole session.

---

## 13. File 9: `api/rest_api.py` — The Web Service

### What does this file do?

It exposes all the engine capabilities as HTTP endpoints so other programs
can use them. Instead of a human using a browser, another program can call
the API.

### What is an API?

API stands for Application Programming Interface. An HTTP API is a way for
programs to talk to each other over the web using standard HTTP requests.

**Types of HTTP requests:**
- `GET` — retrieve information (like reading a web page)
- `POST` — send data (like submitting a form)
- `DELETE` — remove something

### What is FastAPI?

FastAPI is a Python library for building HTTP APIs. It automatically:
- Validates the data you send to it
- Generates documentation (Swagger UI)
- Returns proper error messages if something is wrong

### Key endpoints:

```
GET  /                          ← check if the service is running
GET  /specs                     ← list all registered specs
GET  /specs/{spec_name}         ← get one spec's details
DELETE /specs/{spec_name}       ← delete a spec

POST /train                     ← train a new spec (JSON body)
POST /train/upload              ← train from uploaded file (form data)

POST /identify                  ← auto-detect format
POST /identify/upload           ← auto-detect from uploaded file

POST /validate                  ← validate file content
POST /validate/upload           ← validate uploaded file

POST /generate                  ← generate data (JSON body)
GET  /generate/{spec_name}      ← generate data (URL parameters)

POST /db/connect                ← connect to database
POST /db/disconnect             ← disconnect
GET  /db/tables                 ← list tables
POST /db/query                  ← run SQL query
GET  /db/mock/{format_type}     ← get mock data for testing

GET  /audit                     ← view audit log
GET  /audit/stats               ← view audit statistics
```

### How to call the API (example):

```python
import requests

# Generate 10 NACHA test records
response = requests.get("http://localhost:8000/generate/NACHA?num_records=10&seed=42")
nacha_file = response.text

# Validate a file
response = requests.post("http://localhost:8000/validate", json={
    "spec_name": "NACHA",
    "file_content": open("payroll.ach").read()
})
validation_result = response.json()
print(f"Valid: {validation_result['is_valid']}, Score: {validation_result['score']}%")
```

### Swagger UI (interactive documentation)

When the API server is running, you can visit:
```
http://localhost:8000/docs
```
This shows a web page where you can try every endpoint interactively,
see what parameters are required, and read the descriptions.

### What is Pydantic?

```python
class ValidateRequest(BaseModel):
    spec_name:    str
    file_content: str
    delimiter:    str | None = None
```

Pydantic is a library for defining the exact shape of data. When someone
calls `POST /validate`, FastAPI uses this class to validate the incoming
JSON automatically. If `spec_name` is missing or not a string, it returns
a helpful error message without any extra code from us.

---

## 14. File 10: `cli/fllm.py` — The Command Line Tool

### What does this file do?

It lets you use all the features from a terminal (command prompt) without
opening a browser. Useful for:
- Batch processing in scripts
- Running on a server without a display
- Integrating with other automation tools

### What is argparse?

`argparse` is Python's built-in library for parsing command-line arguments.
It automatically generates help text and validates inputs.

### Commands:

```bash
# List all registered specs
python cli/fllm.py list

# Train a new spec
python cli/fllm.py train MY_SPEC spec_document.txt --format nacha --show-fields

# Validate a data file
python cli/fllm.py validate NACHA payroll.ach

# Validate and save the report
python cli/fllm.py validate NACHA payroll.ach --json-out report.json

# Generate 20 test records and save to a file
python cli/fllm.py generate NACHA --rows 20 --seed 42 --out test_payroll.ach

# Generate using real database data
python cli/fllm.py generate ORACLE_GL --rows 50 \
    --db-conn "postgresql://user:pass@localhost/mydb" \
    --db-sql "SELECT * FROM gl_entries WHERE period='APR-2026'"

# Inspect a spec's field definitions
python cli/fllm.py inspect NACHA

# Auto-detect the format of an unknown file
python cli/fllm.py identify mysterious_file.dat

# View recent audit events
python cli/fllm.py audit --action VALIDATE --limit 10

# Run a database query
python cli/fllm.py db-query "sqlite:///mydb.db" "SELECT * FROM transactions"
```

### Colourised output

The CLI uses ANSI escape codes to add colour to its output:
- ✅ Green = success
- ❌ Red = error
- ⚠️ Yellow = warning
- ℹ️ Cyan = information

These look great in a terminal. In a plain text file, they look like
`\033[92m✅\033[0m` — the `\033[92m` turns on green, `\033[0m` turns it off.

### Exit codes

The validate command exits with:
- `0` = file is valid (used in scripts: `if [[ $? -eq 0 ]]; then echo "OK"; fi`)
- `1` = file has errors

---

## 15. File 11: `tests/test_all.py` — The Quality Checker

### What does this file do?

It automatically runs 43 checks to make sure everything in the system is
working correctly. Run it before deploying or after making changes.

### What is automated testing?

Instead of manually testing "does this work?" every time you change something,
you write code that tests your code. When you run the tests, they tell you
instantly if anything broke.

### How to run:

```bash
cd financial_llm
pip install pytest
pytest tests/test_all.py -v
```

The `-v` means "verbose" — show each test's name and result.

### What the tests check:

**Tokenizer tests** — Does the tokenizer correctly split spec documents?
```python
tokens = tokenizer.tokenize("AMOUNT  30-39  10  N")
assert tokens[0]['is_field'] == True   # should be detected as a field line
```

**SpecEngine tests** — Does training and retrieval work?
```python
spec = engine.train("TEST", spec_text, "custom")
assert spec['field_count'] > 0         # should extract at least one field
assert engine.get_spec("TEST") != None # should be retrievable
```

**FieldValidator tests** — Does validation produce the right results?
```python
result = validator.validate("12345", {"data_type":"numeric","length":5})
assert result.failed == 0              # valid numeric should pass

result = validator.validate("12ABC", {"data_type":"numeric","length":5})
assert result.failed > 0              # invalid numeric should fail
```

**Generator tests** — Does the generator produce correct output?
```python
output = generator.generate("NACHA", num_records=2)
lines = output.splitlines()
assert all(len(l) == 94 for l in lines)  # every line must be 94 chars
```

**AuditLog tests** — Does logging work correctly?
```python
audit.record_training("NACHA", field_count=94)
entries = audit.query(action="TRAIN")
assert len(entries) == 1               # exactly one entry
```

**DBConnector tests** — Does the database connector work?
```python
db.connect("sqlite:///test.db")
rows = db.fetch("SELECT * FROM my_table")
assert len(rows) > 0                   # should return some rows
```

---

## 16. Sample Data Files Explained

### `sample_data/nacha_spec.txt`

This is a complete, detailed description of the NACHA ACH file format.
It describes all 6 record types (1, 5, 6, 7, 8, 9) with field tables like:

```
Field Name              Pos   Len  Type  Required  Notes
RECORD_TYPE_CODE        1-1    1   N     Required  Value = 6
TRANSACTION_CODE        2-3    2   N     Required  See allowed values below
RDFI_ROUTING_TRANSIT    4-11   8   N     Required  First 8 digits of ABA number
...
```

When you use this in the "Train" tab with `--format nacha`, the engine reads
this document and extracts all those field definitions automatically.

### `sample_data/oracle_gl_spec.txt`

Describes the Oracle General Ledger journal entry interface format.
Shows both HEADER and LINE record types with their pipe-delimited fields.

### `sample_data/swift_mt103_spec.txt`

Describes the SWIFT MT103 message format with all the tag definitions,
allowed values, validation rules, and a sample complete MT103 message.

### `sample_data/sample_nacha.ach`

A real-looking NACHA ACH payment file with:
- File Header (record type 1)
- Batch Header for a PPD payroll batch (record type 5)
- 5 individual payment entries (record type 6)
- Batch Control (record type 8)
- File Control (record type 9)
- Padding lines to reach a multiple of 10

Use this in the "Validate" tab to see what a validation run looks like.

### `sample_data/sample_oracle_gl.txt`

A real-looking Oracle GL journal with 3 journal entries:
- An accounts payable invoice
- A payroll expense entry
- A revenue recognition entry

Each uses proper segment combinations and balanced debit/credit amounts.

---

## 17. Key Concepts Glossary

| Term | Plain English Explanation |
|---|---|
| **Token** | A single meaningful unit extracted from text (a word, number, or symbol) |
| **Tokenizer** | Code that breaks text into tokens |
| **Spec / Specification** | A document that describes the rules for a file format |
| **Field** | One piece of information in a record (e.g. "Amount", "Account Number") |
| **Fixed-width** | A file format where each field is at an exact position (no separators) |
| **Delimited** | A file format where fields are separated by a character (pipe, comma, tab) |
| **Regex (Regular Expression)** | A mini-language for describing text patterns |
| **TF-IDF** | A scoring method that weights words by how distinctive they are |
| **Cosine Similarity** | A mathematical measure of how similar two word-count vectors are |
| **ABA Routing Number** | A 9-digit code that identifies a US bank (has a built-in checksum) |
| **NACHA** | The organization that runs the US ACH payment network |
| **ACH** | Automated Clearing House — the US system for electronic bank transfers |
| **VISA VCF** | VISA Card Format — used for clearing credit card transactions |
| **Oracle GL** | Oracle General Ledger — accounting software used by large companies |
| **SWIFT MT103** | International wire transfer message format |
| **BIC** | Bank Identifier Code — an international bank ID (e.g. `BOFAUS3NXXX`) |
| **IBAN** | International Bank Account Number (e.g. `GB29NWBK60161331926819`) |
| **JSON** | JavaScript Object Notation — a simple, human-readable data format |
| **NDJSON** | Newline-Delimited JSON — one JSON object per line |
| **SQLAlchemy** | Python library for connecting to databases |
| **FastAPI** | Python library for building HTTP APIs |
| **Streamlit** | Python library for building web applications |
| **Pytest** | Python library for running automated tests |
| **API** | Application Programming Interface — a way for programs to talk to each other |
| **Checksum** | A number calculated from data that can detect if the data is corrupted |
| **Seed (random)** | A starting number for random generation; same seed = same output |
| **Decorator** | Python syntax (`@something`) that wraps a function with extra behavior |
| **Thread safety** | Code that works correctly when multiple things run simultaneously |

---

## 18. Step-by-Step: What Happens When You Click "Train"

Let's trace exactly what happens when you upload `nacha_spec.txt` and click
"🚀 Train Engine":

```
You click "Train Engine"
         │
         ▼
app.py calls: engine.train("MY_NACHA", spec_text, "nacha", "My custom NACHA")
         │
         ▼
SpecEngine.train() runs:
  Step 1: tokenizer.tokenize(spec_text)
    → reads the document line by line
    → line "AMOUNT   30-39   10   N   Required" becomes:
       { line:45, raw:'AMOUNT...', words:['amount','n','required'],
         numbers:[30,39,10], data_type:'numeric', is_field:True }
    → returns a list of ~200 such token dictionaries

  Step 2: extractor.extract_fields(tokens, "nacha")
    → for each token where is_field=True:
       runs _parse_field() which tries three regex patterns
       "AMOUNT   30-39   10   N   Required" matches pattern 1:
       returns { name:'AMOUNT', start:30, end:39, length:10,
                 data_type:'numeric', required:True, validation:{} }
    → after processing all tokens:
       returns a list of ~50 field dictionaries

  Step 3: builds vocabulary from all words
    → counts word frequency: {routing:8, amount:6, record:10, ...}
    → keeps the top 200 words

  Step 4: computes checksum
    → hashlib.md5(spec_text.encode()).hexdigest()
    → produces a fingerprint like "a3f7b2e1..."
    → used to detect if the exact same spec is trained twice

  Step 5: assembles the spec dictionary
    spec = {
      name: "MY_NACHA",
      format_type: "nacha",
      fields: [50 field dicts],
      vocabulary: {routing:8, amount:6, ...},
      checksum: "a3f7b2e1...",
      token_count: 203,
      field_count: 50
    }

  Step 6: kb.save("MY_NACHA", spec)
    → writes spec to: models/trained_specs/MY_NACHA.json
    → adds _saved_at timestamp

  Step 7: audit.record_training("MY_NACHA", field_count=50)
    → appends one line to models/audit.ndjson

  Step 8: returns the spec dictionary
         │
         ▼
app.py receives the result and shows:
  ✅ MY_NACHA trained — 50 fields / 203 tokens
  [table of extracted fields]
```

---

## 19. Step-by-Step: What Happens When You Click "Validate"

Let's trace validating a NACHA file:

```
You click "Validate"
         │
         ▼
app.py calls: validator.validate("NACHA", file_content, delimiter=None)
         │
         ▼
Validator.validate() looks up the spec:
  spec = kb.load("NACHA")
  format_type = "nacha"
  → calls _validate_nacha(spec, file_content)
         │
         ▼
_validate_nacha() runs:
  splits file into lines: ["101 021000021...", "5220...", "6220...", ...]
  result.total_records = 10 (number of lines)

  For each line:
    Line 1: "101 021000021 021000021260115..."
      len(line) == 94? YES → OK
      rt = line[0] = '1' → File Header Record
      look up records_def['1'] → the File Header field list
      For each field:
        RECORD_TYPE_CODE: start=1, end=1 → value="1" → fixed value check → "1"=="1" ✅
        PRIORITY_CODE: start=2, end=3 → value="01" → numeric check → "01" is numeric ✅
        IMMEDIATE_DESTINATION: start=4, end=13 → value=" 021000021"
          → routing_number check → strip spaces → "021000021"
          → checksum: 3×0+7×2+1+3×1+7×0+0+3×0+7×2+1 = 0+14+1+3+0+0+0+14+1=33
          → 33 mod 10 = 3 ≠ 0 ❌ → add_error!
        ...continues for all fields...

    Line 2: "5220ACME PAYROLL..."
      rt = '5' → Batch Header Record
      ...validates all batch header fields...

    Lines 3-7: "6220..." → Entry Detail Records → validates routing, amount, etc.

    Line 8: "8200000006..." → Batch Control → validates totals
    Line 9: "9000001000001..." → File Control
    Lines 10-10: "9999..." → padding (no validation needed)
         │
         ▼
Returns ValidationResult:
  errors: [{"record":1, "field":"IMMEDIATE_DESTINATION", "message":"Invalid ABA..."}]
  passed: 280
  failed: 1
  score: 99.6%
  is_valid: False (because there is 1 error)
         │
         ▼
app.py shows:
  ❌ INVALID
  ████████████████████████████████████████ 99.6%
  Records: 10    Passed: 280    Failed: 1
  [error table]
```

---

## 20. Step-by-Step: What Happens When You Click "Generate"

```
You click "Generate" for NACHA, 5 records, seed=42
         │
         ▼
app.py calls: generator.generate("NACHA", num_records=5, seed=42)
         │
         ▼
Generator.generate():
  spec = kb.load("NACHA")
  format_type = "nacha"
  RNG.seed(42)  ← set random seed for reproducibility
  calls _gen_nacha(spec, n=5, db_data=None)
         │
         ▼
_gen_nacha():

  Build File Header (record type 1):
    For each field in records_def['1']['fields']:
      fg.generate(field)  ← calls FieldGenerator
        RECORD_TYPE_CODE: value='1' → returns "1"
        PRIORITY_CODE: default '01' → returns "01"
        IMMEDIATE_DESTINATION: type=routing_number → _routing_number()
          → [3,8,1,2,0,0,0,2] * [3,7,1,3,7,1,3,7]
          → sum=62, check=(10-2)%10=8 → "382000028" (9 digits)
          → padded to 10 chars: " 382000028"
        IMMEDIATE_ORIGIN: → random 10-digit number
        FILE_CREATION_DATE: format=YYMMDD → today → "260420"
        ...etc for all 14 fields...
    Assemble into 94-char string, padded with spaces
    → "1 01 382000028 1234567890260420    A094101BANK NAME..."

  Build Batch Header (record type 5):
    Similar process → "5220ACME CORP..."

  Build 5 Entry Detail records (record type 6):
    For each of 5 records:
      TRANSACTION_CODE: random from allowed ["22","23",...] → "22"
      RDFI_ROUTING_TRANSIT: → _routing_number() → "021000021"
      CHECK_DIGIT: → "4" (9th digit of routing)
      DFI_ACCOUNT_NUMBER: → random account → "12345678901234567"
      AMOUNT: → _amount(10, {}) → "0000158432" (cents, zero-padded)
      INDIVIDUAL_NAME: → random name → "JOHN SMITH        "
      TRACE_NUMBER: → "0210000210000001"
      ...
      → "622021000021412345678901234567  0000158432EMP001          JOHN SMITH            00210000210000001"

  Build Batch Control (record type 8): ...
  Build File Control (record type 9): ...

  Padding: total lines=8, need multiple of 10 → add 2 padding lines of "9"×94

  Return "\n".join(all_lines)
         │
         ▼
app.py shows:
  ✅ 10 lines generated
  [preview of first 35 lines]
  [Download button]
```

---

## 21. How to Run the Project

### Step 1: Install Python

Download Python 3.11 or newer from https://python.org/downloads

Verify it works:
```bash
python --version
# Should show: Python 3.11.x or higher
```

### Step 2: Download the project files

Save all the files in a folder called `financial_llm`.

### Step 3: Install the required libraries

```bash
cd financial_llm
pip install streamlit pandas sqlalchemy
```

For database connectivity (install only what you need):
```bash
pip install oracledb          # Oracle Database
pip install psycopg2-binary   # PostgreSQL
pip install pymysql           # MySQL
# SQLite is built into Python — no install needed
```

For the REST API:
```bash
pip install fastapi uvicorn python-multipart
```

### Step 4: Run the web interface

```bash
streamlit run app.py
```

Then open your browser and go to: http://localhost:8501

### Step 5: (Optional) Run the REST API

```bash
uvicorn api.rest_api:app --host 0.0.0.0 --port 8000 --reload
```

Then open: http://localhost:8000/docs

### Step 6: (Optional) Use the command-line tool

```bash
python cli/fllm.py list
python cli/fllm.py generate NACHA --rows 5
```

### Step 7: (Optional) Run the tests

```bash
pip install pytest
pytest tests/test_all.py -v
```

---

## 22. Common Questions (FAQ)

**Q: This isn't really an LLM, is it?**

A: Correct — it is not a neural-network LLM like GPT or Claude. It is a
rule-learning engine. We use the term "Custom LLM" loosely to mean "our own
custom language model for financial formats." The system learns from documents,
stores knowledge, and makes inferences — which are the same goals as an LLM,
achieved with deterministic, explainable code instead of neural networks.

---

**Q: Can it learn formats that are not NACHA/VISA/Oracle?**

A: Yes! That is the whole point of the "Train" tab. Any file format that can
be described in a text document can be learned. The engine extracts field
definitions from fixed-width tables, positional lists, and key-value pairs.

---

**Q: Why does the same seed always produce the same output?**

A: The `seed` parameter initializes the random number generator to a fixed
starting state. Python's `random.Random(seed)` is completely deterministic —
given the same seed, it always produces the same sequence of "random" numbers.
This is very useful for testing: `seed=42` will always give you the same
test file, so you can compare outputs between runs.

---

**Q: What is the `models/trained_specs/` folder?**

A: It is the system's "memory" — where everything it has learned is stored.
Each spec is saved as a `.json` file (e.g. `NACHA.json`, `MY_CUSTOM.json`).
You can back these up, share them with colleagues, or delete them to start fresh.

---

**Q: Can two people use the system at the same time?**

A: The Streamlit interface is designed for a single user. For multi-user
production use, run the REST API (`uvicorn api.rest_api:app`) — it handles
concurrent requests. The audit log uses thread locks to be safe for simultaneous
writes.

---

**Q: What if the spec document is a PDF?**

A: Upload it to the Train tab. Streamlit will read the binary bytes and try
to decode them as UTF-8 text. For PDFs, this works if the PDF contains
actual text (not scanned images). For scanned PDFs, you would need to run
OCR (Optical Character Recognition) first to convert the images to text,
then paste that text into the engine.

---

**Q: How accurate is the auto-detection?**

A: It depends on how much vocabulary the spec document and the data file
share. For well-known formats (NACHA, Oracle GL), scores are typically
0.3–0.8. The ranking is reliable — the correct format is almost always #1.
For custom formats with unusual terminology, you may need to train more examples.

---

**Q: Does this send any data to the internet?**

A: No. The entire system runs locally on your machine. No data is sent anywhere.
The only network connection is when you connect to your own database server.

---

**Q: What is the difference between `errors` and `warnings` in validation?**

A: **Errors** are definite violations of the spec rules — wrong data type,
missing required field, value not in allowed list. A file with errors is invalid.

**Warnings** are things that might be problems but are not definitively wrong —
for example, a record that is slightly shorter than expected but not empty.
A file with only warnings is still marked as valid.

---

*This guide was written to help beginners understand every aspect of the*
*Financial LLM Studio codebase. If you have questions after reading this,*
*start by running the project with the sample data and observing what happens.*
*The best way to learn code is to change it and see what breaks!*
