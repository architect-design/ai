#!/usr/bin/env python3
"""
Financial LLM Studio  –  Command Line Interface
================================================
A fully-featured CLI for training specs, validating files,
generating test data, and querying databases.

Usage:
    python cli/fllm.py --help
    python cli/fllm.py train  spec_name  path/to/spec.txt  --format nacha
    python cli/fllm.py validate NACHA path/to/data.ach
    python cli/fllm.py generate NACHA --rows 10 --seed 42 --out test.ach
    python cli/fllm.py list
    python cli/fllm.py inspect NACHA
    python cli/fllm.py identify path/to/unknown.dat
    python cli/fllm.py audit --limit 20
    python cli/fllm.py db-query "sqlite:///my.db" "SELECT * FROM gl_lines"
"""

import os
import sys
import json
import argparse
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.spec_engine   import SpecEngine
from core.validator     import Validator
from core.generator     import Generator
from core.db_connector  import DBConnector, DBConnectionError
from core.audit_log     import AuditLog
from formats.builtin_formats import seed_knowledge_base
from formats.swift_mt103     import SWIFT_MT103_SPEC

# ── ANSI colours ─────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def ok(msg):   print(f"{GREEN}✅ {msg}{RESET}")
def err(msg):  print(f"{RED}❌ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"{CYAN}ℹ️  {msg}{RESET}")
def hdr(msg):  print(f"\n{BOLD}{BLUE}{'─'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{'─'*60}")


def _bootstrap():
    engine = SpecEngine()
    seed_knowledge_base(engine.kb)
    engine.kb.save("SWIFT_MT103", SWIFT_MT103_SPEC)
    return engine


# ════════════════════════════════════════════════════════════════════════
# Commands
# ════════════════════════════════════════════════════════════════════════

def cmd_list(args):
    engine = _bootstrap()
    specs  = engine.list_specs()
    hdr("Registered Specs")
    if not specs:
        warn("No specs registered yet. Run: fllm train <name> <spec_file>")
        return
    for name in sorted(specs):
        spec = engine.get_spec(name) or {}
        ft   = spec.get("format_type", "?")
        fc   = spec.get("field_count", len(spec.get("fields", [])))
        ts   = spec.get("_saved_at", "built-in")[:10]
        print(f"  {BOLD}{name:<30}{RESET}  {CYAN}{ft:<15}{RESET}  "
              f"{fc:>4} fields  {DIM}{ts}{RESET}")
    print()


def cmd_train(args):
    engine = _bootstrap()
    hdr(f"Training: {args.spec_name}")

    if not os.path.exists(args.spec_file):
        err(f"File not found: {args.spec_file}")
        sys.exit(1)

    with open(args.spec_file, encoding="utf-8", errors="replace") as fh:
        spec_text = fh.read()

    print(f"  Format type : {args.format}")
    print(f"  File        : {args.spec_file}  ({len(spec_text):,} chars)")

    result = engine.train(
        spec_name   = args.spec_name,
        spec_text   = spec_text,
        format_type = args.format,
        description = args.description,
    )

    AuditLog().record_training(args.spec_name, field_count=result["field_count"],
                               format_type=args.format, source_file=args.spec_file)

    ok(f"Spec '{args.spec_name}' trained successfully")
    print(f"  Tokens analysed : {result['token_count']}")
    print(f"  Fields extracted: {result['field_count']}")

    if args.show_fields and result["fields"]:
        print()
        print(f"  {'NAME':<35} {'START':>6} {'END':>6} {'LEN':>5}  {'TYPE':<15} REQ")
        print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*5}  {'─'*15} {'─'*3}")
        for f in result["fields"][:40]:
            print(
                f"  {f.get('name',''):<35} "
                f"{str(f.get('start','') or ''):>6} "
                f"{str(f.get('end','') or ''):>6} "
                f"{str(f.get('length','') or ''):>5}  "
                f"{f.get('data_type',''):.<15} "
                f"{'Y' if f.get('required') else 'N'}"
            )
        if len(result["fields"]) > 40:
            print(f"  … and {len(result['fields'])-40} more fields")
    print()


def cmd_validate(args):
    engine    = _bootstrap()
    validator = Validator(engine.kb)
    hdr(f"Validating against: {args.spec_name}")

    if not engine.get_spec(args.spec_name):
        err(f"Spec '{args.spec_name}' not found. Run: fllm list")
        sys.exit(1)

    if not os.path.exists(args.data_file):
        err(f"File not found: {args.data_file}")
        sys.exit(1)

    with open(args.data_file, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    print(f"  File   : {args.data_file}  ({len(content.splitlines())} lines)")
    result = validator.validate(args.spec_name, content, args.delimiter or None)
    report = result.to_dict()

    AuditLog().record_validation(
        args.spec_name, args.data_file,
        is_valid=report["is_valid"], score=report["score"],
        records=report["total_records"], errors=report["failed"],
    )

    bar_len = 40
    filled  = int(report["score"] / 100 * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    colour  = GREEN if report["is_valid"] else RED

    print(f"\n  Score  : {colour}{bar}{RESET}  {report['score']:.1f}%")
    print(f"  Records: {report['total_records']}")
    print(f"  Passed : {GREEN}{report['passed']}{RESET}")
    print(f"  Failed : {RED}{report['failed']}{RESET}")

    if report["errors"]:
        print(f"\n  {RED}{'─'*58}{RESET}")
        print(f"  {BOLD}ERRORS{RESET} (showing first {min(20, len(report['errors']))}):")
        for e in report["errors"][:20]:
            print(f"  {DIM}Rec {e['record']:>4}{RESET}  {YELLOW}{e['field']:<30}{RESET}  {e['message']}")

    if report["warnings"]:
        print(f"\n  {YELLOW}Warnings: {len(report['warnings'])}{RESET}")
        for w in report["warnings"][:10]:
            print(f"  {DIM}Rec {w['record']:>4}{RESET}  {w['field']:<30}  {w['message']}")

    if report["is_valid"]:
        ok("File is VALID")
    else:
        err(f"File is INVALID ({report['failed']} error(s))")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2)
        info(f"Report written to: {args.json_out}")

    sys.exit(0 if report["is_valid"] else 1)


def cmd_generate(args):
    engine    = _bootstrap()
    generator = Generator(engine.kb)
    hdr(f"Generating: {args.spec_name}")

    if not engine.get_spec(args.spec_name):
        err(f"Spec '{args.spec_name}' not found. Run: fllm list")
        sys.exit(1)

    # DB source?
    db_data = None
    if args.db_conn and args.db_sql:
        info(f"Connecting to DB …")
        db = DBConnector()
        try:
            db.connect(args.db_conn)
            db_data = db.fetch(args.db_sql, max_rows=args.rows)
            ok(f"Loaded {len(db_data)} rows from DB")
            db.disconnect()
        except DBConnectionError as exc:
            err(str(exc))
            sys.exit(1)

    output = generator.generate(
        spec_name   = args.spec_name,
        num_records = args.rows,
        seed        = args.seed,
        db_data     = db_data,
    )

    AuditLog().record_generation(args.spec_name, rows=args.rows,
                                 seed=args.seed, from_db=bool(db_data))

    lines = output.splitlines()
    ok(f"Generated {len(lines)} lines")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(output)
        ok(f"Written to: {args.out}")
    else:
        print("\n" + "\n".join(lines[:30]))
        if len(lines) > 30:
            warn(f"… {len(lines)-30} more lines (use --out to save all)")
    print()


def cmd_inspect(args):
    engine = _bootstrap()
    spec   = engine.get_spec(args.spec_name)
    hdr(f"Spec: {args.spec_name}")

    if not spec:
        err(f"Spec '{args.spec_name}' not found")
        sys.exit(1)

    print(f"  Format type : {spec.get('format_type','?')}")
    print(f"  Description : {spec.get('description','—')}")
    print(f"  Saved at    : {spec.get('_saved_at','built-in')}")
    print(f"  Fields      : {len(spec.get('fields',[]))}")

    if args.json:
        print(json.dumps(spec, indent=2))
        return

    fields = spec.get("fields", [])
    if fields:
        print()
        print(f"  {'#':>3}  {'NAME':<35} {'S':>5} {'E':>5} {'L':>4}  {'TYPE':<15} {'REQ'}")
        print(f"  {'─'*3}  {'─'*35} {'─'*5} {'─'*5} {'─'*4}  {'─'*15} {'─'*3}")
        for i, f in enumerate(fields, 1):
            req = f"{GREEN}Y{RESET}" if f.get("required") else f"{DIM}N{RESET}"
            print(
                f"  {i:>3}  {f.get('name',''):<35} "
                f"{str(f.get('start','') or ''):>5} "
                f"{str(f.get('end','') or ''):>5} "
                f"{str(f.get('length','') or ''):>4}  "
                f"{f.get('data_type',''):<15} {req}"
            )
    print()


def cmd_identify(args):
    engine = _bootstrap()
    hdr(f"Identifying: {args.data_file}")

    if not os.path.exists(args.data_file):
        err(f"File not found: {args.data_file}")
        sys.exit(1)

    with open(args.data_file, encoding="utf-8", errors="replace") as fh:
        sample = fh.read(3000)

    matches = engine.identify(sample)

    if not matches:
        warn("No specs registered — cannot identify format")
        return

    print(f"  {'RANK':<5} {'SPEC':<25} {'SCORE':>10}")
    print(f"  {'─'*5} {'─'*25} {'─'*10}")
    for i, m in enumerate(matches, 1):
        bar_w = int(m["score"] * 40)
        bar   = "█" * bar_w
        print(f"  {i:<5} {m['spec']:<25} {m['score']:>10.4f}  {BLUE}{bar}{RESET}")

    best = matches[0]
    print()
    ok(f"Best match: {best['spec']} (score {best['score']:.4f})")
    print()


def cmd_audit(args):
    audit   = AuditLog()
    entries = audit.query(action=args.action, spec=args.spec, limit=args.limit)
    stats   = audit.stats()
    hdr("Audit Log")

    print(f"  Total events  : {stats['total_events']}")
    print(f"  Specs trained : {stats['specs_trained']}")
    print(f"  Validations   : {stats['validations_run']}")
    print(f"  Files generated: {stats['files_generated']}")
    print(f"  Avg valid score: {stats['avg_valid_score']}%")
    print()

    if not entries:
        info("No audit entries match your filters.")
        return

    print(f"  {'TIME':<24} {'ACTION':<12} {'SPEC':<20} {'DETAILS'}")
    print(f"  {'─'*24} {'─'*12} {'─'*20} {'─'*30}")
    for e in entries[:args.limit]:
        ts     = e.get("ts", "")[:19].replace("T", " ")
        action = e.get("action", "")
        spec   = e.get("spec", e.get("db_type", ""))
        detail = ""
        if action == "VALIDATE":
            score = e.get("score", 0)
            valid = e.get("is_valid", False)
            colour = GREEN if valid else RED
            detail = f"{colour}{'✅' if valid else '❌'} {score:.1f}%{RESET} ({e.get('records',0)} recs)"
        elif action == "GENERATE":
            detail = f"{e.get('rows',0)} rows"
        elif action == "TRAIN":
            detail = f"{e.get('field_count',0)} fields"
        elif action == "DB_QUERY":
            detail = f"{e.get('rows_returned',0)} rows"
        print(f"  {DIM}{ts}{RESET}  {CYAN}{action:<12}{RESET} {spec:<20} {detail}")
    print()


def cmd_db_query(args):
    hdr("Database Query")
    db = DBConnector()
    try:
        info(f"Connecting …")
        db.connect(args.conn_str)
        ok(f"Connected: {db.conn_str}")
        rows = db.fetch(args.sql, max_rows=args.max_rows)
        ok(f"{len(rows)} rows returned")

        if rows:
            cols = list(rows[0].keys())
            col_w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
            header = "  " + "  ".join(f"{c:<{col_w[c]}}" for c in cols)
            print(header)
            print("  " + "  ".join("─" * col_w[c] for c in cols))
            for row in rows:
                print("  " + "  ".join(f"{str(row.get(c,'')):<{col_w[c]}}" for c in cols))

        db.disconnect()
    except DBConnectionError as exc:
        err(str(exc))
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════
# Argument Parser
# ════════════════════════════════════════════════════════════════════════
def build_parser():
    parser = argparse.ArgumentParser(
        prog        = "fllm",
        description = "Financial LLM Studio – Command Line Interface",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = textwrap.dedent("""
        Examples:
          fllm list
          fllm train MY_ACH_V2 spec.txt --format nacha --show-fields
          fllm validate NACHA payroll.ach --delimiter ""
          fllm generate ORACLE_GL --rows 25 --seed 99 --out gl_test.txt
          fllm generate NACHA --rows 5 --db-conn sqlite:///mydb.db --db-sql "SELECT * FROM ach_records"
          fllm inspect SWIFT_MT103
          fllm identify unknown_file.dat
          fllm audit --action VALIDATE --limit 10
          fllm db-query "sqlite:///mydb.db" "SELECT * FROM orders LIMIT 5"
        """)
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # list
    p = sub.add_parser("list", help="List all registered specs")
    p.set_defaults(func=cmd_list)

    # train
    p = sub.add_parser("train", help="Train engine on a spec document")
    p.add_argument("spec_name",  help="Unique spec identifier")
    p.add_argument("spec_file",  help="Path to spec document (.txt/.csv/.md/.json)")
    p.add_argument("--format",   default="custom",
                   choices=["nacha","visa_vcf","oracle_gl","swift_mt103","custom"],
                   help="Format family hint (default: custom)")
    p.add_argument("--description", default="", help="Human-readable description")
    p.add_argument("--show-fields", action="store_true", help="Print extracted fields")
    p.set_defaults(func=cmd_train)

    # validate
    p = sub.add_parser("validate", help="Validate a data file against a spec")
    p.add_argument("spec_name",  help="Spec name (from fllm list)")
    p.add_argument("data_file",  help="Path to data file to validate")
    p.add_argument("--delimiter", default="", help="Field delimiter (blank=fixed-width)")
    p.add_argument("--json-out", metavar="PATH", help="Save validation report as JSON")
    p.set_defaults(func=cmd_validate)

    # generate
    p = sub.add_parser("generate", help="Generate a test data file")
    p.add_argument("spec_name",  help="Spec name (from fllm list)")
    p.add_argument("--rows",     type=int, default=5,  help="Number of detail records (default 5)")
    p.add_argument("--seed",     type=int, default=None, help="Random seed")
    p.add_argument("--out",      metavar="PATH", help="Output file path")
    p.add_argument("--db-conn",  metavar="CONN", help="DB connection string for source data")
    p.add_argument("--db-sql",   metavar="SQL",  help="SQL query to pull source rows")
    p.set_defaults(func=cmd_generate)

    # inspect
    p = sub.add_parser("inspect", help="Inspect a spec's field definitions")
    p.add_argument("spec_name", help="Spec name")
    p.add_argument("--json",    action="store_true", help="Output raw JSON")
    p.set_defaults(func=cmd_inspect)

    # identify
    p = sub.add_parser("identify", help="Auto-detect the format of an unknown file")
    p.add_argument("data_file", help="Path to unknown file")
    p.set_defaults(func=cmd_identify)

    # audit
    p = sub.add_parser("audit", help="View audit log")
    p.add_argument("--action", choices=["TRAIN","VALIDATE","GENERATE","DB_CONNECT","DB_QUERY","DELETE"],
                   help="Filter by action type")
    p.add_argument("--spec",   help="Filter by spec name")
    p.add_argument("--limit",  type=int, default=20, help="Max entries to show")
    p.set_defaults(func=cmd_audit)

    # db-query
    p = sub.add_parser("db-query", help="Run a SQL query against a database")
    p.add_argument("conn_str", help="SQLAlchemy connection string")
    p.add_argument("sql",      help="SQL SELECT statement")
    p.add_argument("--max-rows", type=int, default=50, help="Max rows to return")
    p.set_defaults(func=cmd_db_query)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    print(f"\n{BOLD}{CYAN}Financial LLM Studio{RESET}  {DIM}Custom Rule-Learning Engine{RESET}\n")
    args.func(args)


if __name__ == "__main__":
    main()
