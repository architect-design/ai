"""
Financial LLM Studio  –  REST API
===================================
FastAPI-based REST API exposing all engine capabilities as HTTP endpoints.

Run:
    pip install fastapi uvicorn python-multipart
    cd financial_llm
    uvicorn api.rest_api:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs  (Swagger UI)
      http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
import json
import io
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
    from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False

from core.spec_engine import SpecEngine
from core.validator   import Validator
from core.generator   import Generator
from core.db_connector import DBConnector, DBConnectionError
from core.audit_log   import AuditLog
from formats.builtin_formats import seed_knowledge_base
from formats.swift_mt103     import SWIFT_MT103_SPEC

# ── Bootstrap ─────────────────────────────────────────────────────────────
_engine    = SpecEngine()
seed_knowledge_base(_engine.kb)
_engine.kb.save("SWIFT_MT103", SWIFT_MT103_SPEC)

_validator = Validator(_engine.kb)
_generator = Generator(_engine.kb)
_db        = DBConnector()
_audit     = AuditLog()


if not FASTAPI_OK:
    raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

app = FastAPI(
    title       = "Financial LLM Studio API",
    description = "Custom rule-learning engine for NACHA, VISA VCF, Oracle GL, SWIFT MT103, and custom formats",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ════════════════════════════════════════════════════════════════════════
class TrainRequest(BaseModel):
    spec_name:   str = Field(..., description="Unique spec identifier", example="MY_BANK_FMT_V2")
    spec_text:   str = Field(..., description="Raw specification document text")
    format_type: str = Field("custom", description="nacha|visa_vcf|oracle_gl|swift_mt103|custom")
    description: str = Field("", description="Human-readable description")
    metadata:    dict[str, Any] = Field({}, description="Arbitrary metadata")

class ValidateRequest(BaseModel):
    spec_name:    str = Field(..., example="NACHA")
    file_content: str = Field(..., description="Raw file content to validate")
    delimiter:    str | None = Field(None, description="Field delimiter for delimited files")

class GenerateRequest(BaseModel):
    spec_name:   str         = Field(..., example="NACHA")
    num_records: int         = Field(5, ge=1, le=1000)
    seed:        int | None  = Field(None, description="Random seed for reproducibility")
    db_data:     list[dict]  = Field([], description="Source rows from DB (optional)")

class DBConnectRequest(BaseModel):
    connection_string: str = Field(..., description="SQLAlchemy connection string")

class DBQueryRequest(BaseModel):
    sql:      str = Field(..., description="SELECT query to execute")
    max_rows: int = Field(100, ge=1, le=5000)


# ════════════════════════════════════════════════════════════════════════
# Health
# ════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Health"])
def root():
    return {
        "service":  "Financial LLM Studio",
        "status":   "running",
        "specs":    _engine.list_specs(),
        "db":       _db.status,
    }

@app.get("/health", tags=["Health"])
def health():
    return {"ok": True, "spec_count": len(_engine.list_specs())}


# ════════════════════════════════════════════════════════════════════════
# Spec Management
# ════════════════════════════════════════════════════════════════════════
@app.get("/specs", tags=["Specs"])
def list_specs():
    """List all registered specs."""
    return {"specs": _engine.list_specs()}

@app.get("/specs/{spec_name}", tags=["Specs"])
def get_spec(spec_name: str):
    """Retrieve a single spec by name."""
    spec = _engine.get_spec(spec_name)
    if not spec:
        raise HTTPException(404, detail=f"Spec '{spec_name}' not found")
    return spec

@app.delete("/specs/{spec_name}", tags=["Specs"])
def delete_spec(spec_name: str):
    """Delete a spec."""
    ok = _engine.delete_spec(spec_name)
    if not ok:
        raise HTTPException(404, detail=f"Spec '{spec_name}' not found")
    _audit.record_delete(spec_name)
    return {"deleted": spec_name}


# ════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════
@app.post("/train", tags=["Training"])
def train_spec(req: TrainRequest):
    """
    Train the engine on a spec document (JSON body).
    Returns the learned spec with extracted field definitions.
    """
    try:
        result = _engine.train(
            spec_name   = req.spec_name,
            spec_text   = req.spec_text,
            format_type = req.format_type,
            description = req.description,
            metadata    = req.metadata,
        )
        _audit.record_training(req.spec_name, field_count=result["field_count"],
                               format_type=req.format_type)
        return {
            "spec_name":   result["name"],
            "field_count": result["field_count"],
            "token_count": result["token_count"],
            "format_type": result["format_type"],
            "fields":      result["fields"][:20],   # preview
        }
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

@app.post("/train/upload", tags=["Training"])
async def train_from_file(
    spec_name:   str        = Form(...),
    format_type: str        = Form("custom"),
    description: str        = Form(""),
    file:        UploadFile = File(...),
):
    """
    Train from an uploaded spec file (multipart/form-data).
    Supports .txt .csv .md .json
    """
    raw = await file.read()
    try:
        spec_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        spec_text = raw.decode("latin-1")

    try:
        result = _engine.train(
            spec_name   = spec_name,
            spec_text   = spec_text,
            format_type = format_type,
            description = description,
            metadata    = {"source_file": file.filename},
        )
        _audit.record_training(spec_name, field_count=result["field_count"],
                               format_type=format_type, source_file=file.filename or "")
        return {
            "spec_name":   result["name"],
            "field_count": result["field_count"],
            "source_file": file.filename,
        }
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


# ════════════════════════════════════════════════════════════════════════
# Identify / Auto-detect
# ════════════════════════════════════════════════════════════════════════
@app.post("/identify", tags=["Identify"])
def identify_format(file_content: str = Form(...)):
    """Auto-detect which spec best matches the provided file content."""
    matches = _engine.identify(file_content)
    return {"matches": matches}

@app.post("/identify/upload", tags=["Identify"])
async def identify_from_file(file: UploadFile = File(...)):
    """Upload a file and auto-detect its format."""
    raw  = await file.read()
    text = raw.decode("utf-8", errors="replace")
    matches = _engine.identify(text[:2000])
    return {"filename": file.filename, "matches": matches}


# ════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════
@app.post("/validate", tags=["Validation"])
def validate(req: ValidateRequest):
    """Validate file content against a spec."""
    try:
        result = _validator.validate(req.spec_name, req.file_content, req.delimiter)
        report = result.to_dict()
        _audit.record_validation(
            req.spec_name, "API call",
            is_valid=report["is_valid"], score=report["score"],
            records=report["total_records"], errors=report["failed"],
        )
        return report
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

@app.post("/validate/upload", tags=["Validation"])
async def validate_from_file(
    spec_name: str        = Form(...),
    delimiter: str        = Form(""),
    file:      UploadFile = File(...),
):
    """Upload a data file and validate it."""
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    result = _validator.validate(spec_name, content, delimiter or None)
    report = result.to_dict()
    _audit.record_validation(
        spec_name, file.filename or "upload",
        is_valid=report["is_valid"], score=report["score"],
        records=report["total_records"], errors=report["failed"],
    )
    return {**report, "filename": file.filename}


# ════════════════════════════════════════════════════════════════════════
# Generation
# ════════════════════════════════════════════════════════════════════════
@app.post("/generate", tags=["Generation"])
def generate(req: GenerateRequest):
    """Generate a test data file and return it as plain text."""
    if not _engine.get_spec(req.spec_name):
        raise HTTPException(404, detail=f"Spec '{req.spec_name}' not found")
    try:
        output = _generator.generate(
            spec_name   = req.spec_name,
            num_records = req.num_records,
            seed        = req.seed,
            db_data     = req.db_data or None,
        )
        _audit.record_generation(req.spec_name, rows=req.num_records,
                                 seed=req.seed, from_db=bool(req.db_data))
        return PlainTextResponse(content=output, media_type="text/plain")
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

@app.get("/generate/{spec_name}", tags=["Generation"])
def generate_quick(
    spec_name:   str,
    num_records: int       = Query(5, ge=1, le=1000),
    seed:        int | None = Query(None),
):
    """Quick GET endpoint to generate test data (no body needed)."""
    if not _engine.get_spec(spec_name):
        raise HTTPException(404, detail=f"Spec '{spec_name}' not found")
    output = _generator.generate(spec_name, num_records, seed)
    _audit.record_generation(spec_name, rows=num_records, seed=seed)
    ext_map = {"nacha": "ach", "oracle_gl": "txt", "visa_vcf": "dat",
               "swift_mt103": "txt", "custom": "txt"}
    spec    = _engine.get_spec(spec_name) or {}
    ext     = ext_map.get(spec.get("format_type", "custom"), "txt")
    return StreamingResponse(
        io.StringIO(output),
        media_type = "text/plain",
        headers    = {"Content-Disposition": f'attachment; filename="{spec_name}.{ext}"'},
    )


# ════════════════════════════════════════════════════════════════════════
# Database
# ════════════════════════════════════════════════════════════════════════
@app.post("/db/connect", tags=["Database"])
def db_connect(req: DBConnectRequest):
    """Connect to a database."""
    try:
        _db.connect(req.connection_string)
        _audit.record_db_connect("auto", _db.conn_str or "", True)
        return {"connected": True, "conn": _db.conn_str}
    except DBConnectionError as exc:
        _audit.record_db_connect("auto", "****", False)
        raise HTTPException(400, detail=str(exc))

@app.post("/db/disconnect", tags=["Database"])
def db_disconnect():
    """Disconnect from the current database."""
    _db.disconnect()
    return {"connected": False}

@app.get("/db/tables", tags=["Database"])
def db_tables():
    """List all tables in the connected database."""
    if not _db.connected:
        raise HTTPException(400, detail="Not connected. POST /db/connect first.")
    try:
        return {"tables": _db.list_tables()}
    except DBConnectionError as exc:
        raise HTTPException(500, detail=str(exc))

@app.post("/db/query", tags=["Database"])
def db_query(req: DBQueryRequest):
    """Execute a SELECT query and return rows."""
    if not _db.connected:
        raise HTTPException(400, detail="Not connected. POST /db/connect first.")
    try:
        rows = _db.fetch(req.sql, max_rows=req.max_rows)
        _audit.record_db_query(req.sql, len(rows))
        return {"rows": rows, "count": len(rows)}
    except DBConnectionError as exc:
        raise HTTPException(500, detail=str(exc))

@app.get("/db/mock/{format_type}", tags=["Database"])
def db_mock(format_type: str, n: int = Query(10, ge=1, le=100)):
    """Get mock database rows shaped for a given format (for testing)."""
    rows = _db.get_mock_data(format_type, n)
    return {"rows": rows, "count": len(rows)}


# ════════════════════════════════════════════════════════════════════════
# Audit Log
# ════════════════════════════════════════════════════════════════════════
@app.get("/audit", tags=["Audit"])
def get_audit(
    action: str | None = Query(None, description="Filter by action: TRAIN|VALIDATE|GENERATE|DB_CONNECT|DB_QUERY|DELETE"),
    spec:   str | None = Query(None, description="Filter by spec name"),
    limit:  int        = Query(100, ge=1, le=1000),
):
    """Retrieve audit log entries."""
    return {"entries": _audit.query(action=action, spec=spec, limit=limit)}

@app.get("/audit/stats", tags=["Audit"])
def audit_stats():
    """Aggregate statistics from the audit log."""
    return _audit.stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
