#!/usr/bin/env python
"""
example_usage.py
================
Shows the full SpecGenAI workflow programmatically (no HTTP needed).
Run from the spec_gen_ai/ directory:

    python example_usage.py

Demonstrates:
  1. Parsing an ACH spec
  2. Parsing a VCF spec
  3. Parsing the native JSON spec (sample_spec.json)
  4. Training from a sample CSV
  5. Generating ACH, VCF and JSON files
  6. Validating generated output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure we're in the right directory
sys.path.insert(0, str(Path(__file__).parent))

from app.parsers.ach_parser import ACHParser
from app.parsers.vcf_parser import VCFParser
from app.parsers.json_schema_parser import JSONSchemaParser
from app.parsers.sample_data_parser import SampleDataParser
from app.learner.spec_learner import SpecLearner
from app.generator.base_generator import get_generator
from app.validation.validator import ValidationEngine


def separator(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────
# 1. Parse ACH specification
# ─────────────────────────────────────────────────────────────
separator("1. ACH Parser")

ach_parser = ACHParser()
ach_model = ach_parser.parse_content('{"spec_type": "ach"}', source_name="payroll.ach")

print(f"  ACH spec parsed: {len(ach_model.records)} record types")
for rec in sorted(ach_model.records, key=lambda r: r.ordering):
    print(f"    [{rec.record_type_id}] {rec.name:20s}  "
          f"{len(rec.fields)} fields  "
          f"fixed_width={rec.fixed_width}  "
          f"length={rec.record_length}")


# ─────────────────────────────────────────────────────────────
# 2. Parse VCF specification
# ─────────────────────────────────────────────────────────────
separator("2. VCF Parser")

vcf_parser = VCFParser()
vcf_model = vcf_parser.parse_content("{}", source_name="clearing.vcf")

detail = vcf_model.get_record("DETAIL")
print(f"  VCF spec parsed: {len(vcf_model.records)} record types")
print(f"  Detail record has {len(detail.fields)} fields")
print(f"  Sample fields:")
for f in detail.fields[:6]:
    print(f"    {f.name:35s}  type={f.field_type.value:20s}  len={f.length}")


# ─────────────────────────────────────────────────────────────
# 3. Parse native JSON specification
# ─────────────────────────────────────────────────────────────
separator("3. JSON Native Spec Parser")

spec_path = Path(__file__).parent / "sample_spec.json"
if spec_path.exists():
    json_parser = JSONSchemaParser()
    json_model = json_parser.parse_file(spec_path)
    print(f"  Spec name: {json_model.spec_name}")
    print(f"  Format:    {json_model.file_structure.format}")
    print(f"  Records:   {len(json_model.records)}")
    for rec in json_model.ordered_records():
        print(f"    [{rec.record_type_id:6s}] {rec.name:20s}  {len(rec.fields)} fields")
else:
    print("  (sample_spec.json not found, skipping)")


# ─────────────────────────────────────────────────────────────
# 4. Learn from sample CSV data
# ─────────────────────────────────────────────────────────────
separator("4. Sample Data Parser + Field Inferrer")

SAMPLE_CSV = """\
txn_id,card_number,amount,transaction_date,currency,status,routing_number
TXN001,4111111111111111,10000,20240115,840,APPROVED,021000021
TXN002,4242424242424242,25000,20240116,840,DECLINED,021000021
TXN003,4000000000000002,5000,20240117,978,APPROVED,021000021
TXN004,4532015112830366,99999,20240118,840,APPROVED,021000021
TXN005,4916338506082832,1500,20240119,826,APPROVED,021000021
"""

sample_parser = SampleDataParser()
sample_model = sample_parser.parse_content(SAMPLE_CSV, source_name="transactions.csv")

print(f"  Detected format: {sample_model.file_structure.format}")
print(f"  Inferred fields:")
rec = sample_model.records[0]
for f in rec.fields:
    print(f"    {f.name:25s}  type={f.field_type.value:20s}  "
          f"confidence={f.inferred_confidence:.2f}")


# ─────────────────────────────────────────────────────────────
# 5. Generate ACH file
# ─────────────────────────────────────────────────────────────
separator("5. ACH File Generation (10 entries)")

ach_model.is_trained = True
ach_gen = get_generator(ach_model, seed=42)
ach_result = ach_gen.generate(record_count=10)

lines = ach_result.content.splitlines()
print(f"  Generated {len(lines)} lines ({len(lines)//10} blocks of 10)")
print(f"  Record count: {ach_result.record_count} entry detail records")
print(f"  Validation passed: {ach_result.validation_passed}")
print(f"  First 5 lines:")
for line in lines[:5]:
    rt = line[0]
    label = {"1": "FileHdr", "5": "BchHdr", "6": "Entry",
             "8": "BchCtrl", "9": "FileCtrl", "9"*94: "Padding"}.get(rt, f"Type={rt}")
    print(f"    [{label:8s}] {line[:40]}…")

assert all(len(line) == 94 for line in lines), "All ACH lines must be 94 chars"
assert len(lines) % 10 == 0, "ACH file must be padded to multiple of 10 records"
print(f"  ✓ All lines are 94 chars")
print(f"  ✓ File padded to {len(lines)} lines (multiple of 10)")


# ─────────────────────────────────────────────────────────────
# 6. Generate VCF file
# ─────────────────────────────────────────────────────────────
separator("6. VCF File Generation (5 transactions)")

vcf_model.is_trained = True
vcf_gen = get_generator(vcf_model, seed=99)
vcf_result = vcf_gen.generate(record_count=5)

vcf_lines = vcf_result.content.splitlines()
print(f"  Generated {len(vcf_lines)} lines (1 header + 5 detail + 1 trailer)")
print(f"  Header:  {vcf_lines[0][:40]}…")
print(f"  Detail1: {vcf_lines[1][:40]}…")
print(f"  Trailer: {vcf_lines[-1][:40]}…")

# Check PANs are Luhn-valid
detail_rec = vcf_model.get_record("DETAIL")
pan_field = detail_rec.get_field("primary_account_number")

def luhn(pan):
    digits = [int(d) for d in pan if d.isdigit()]
    if len(digits) < 13: return False
    digits.reverse()
    return sum(d if i%2==0 else (d*2-9 if d*2>9 else d*2)
               for i, d in enumerate(digits)) % 10 == 0

for i, row in enumerate(vcf_result.records):
    if row.get("_record_type") == "DETAIL":
        pan = row.get("primary_account_number", "").strip()
        assert luhn(pan), f"Row {i} PAN {pan} failed Luhn!"

print(f"  ✓ All PAN values pass Luhn checksum")


# ─────────────────────────────────────────────────────────────
# 7. Generate JSON file from custom spec
# ─────────────────────────────────────────────────────────────
separator("7. JSON File Generation (custom spec)")

spec_path = Path(__file__).parent / "sample_spec.json"
if spec_path.exists():
    json_model = JSONSchemaParser().parse_file(spec_path)
    json_model.is_trained = True
    json_gen = get_generator(json_model, seed=7)
    json_result = json_gen.generate(record_count=3, overrides={})

    # The spec is delimited (PSV) format, not a JSON array
    lines = json_result.content.splitlines()
    print(f"  Generated {json_result.record_count} records → {len(lines)} lines")
    for line in lines[:5]:
        fields = line.split("|")
        print(f"    [{fields[0]:3s}]  {' | '.join(fields[1:4])}{'…' if len(fields)>4 else ''}")
    print(f"  Validation passed: {json_result.validation_passed}")
else:
    print("  (sample_spec.json not found, skipping)")


# ─────────────────────────────────────────────────────────────
# 8. Validate generated output
# ─────────────────────────────────────────────────────────────
separator("8. Validation Engine")

validator = ValidationEngine(ach_model)
report = validator.validate(ach_result.records)
print(f"  Total records validated: {report.total_records}")
print(f"  Records with errors:     {report.records_with_errors}")
print(f"  Total errors:            {report.total_errors}")
print(f"  Summary: {report.summary}")


# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
separator("All examples completed successfully ✓")
print()
