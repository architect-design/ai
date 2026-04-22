"""
SpecGenAI — FastAPI Application Entry Point
============================================
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.main_router import router
from app.core.config import settings
from app.core.exceptions import SpecGenAIError

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 %s v%s starting up", settings.APP_NAME, settings.APP_VERSION)
    settings.ensure_dirs()
    yield
    logger.info("🛑 %s shutting down", settings.APP_NAME)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Specification-driven Generative AI system that learns file structures "
            "from uploaded specs (VISA VCF, ACH/NACHA, custom JSON) and generates "
            "valid synthetic test data on demand. No external LLM required."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.1f}"
        return response

    # ── Global exception handler for domain errors ────────────────────────────
    @app.exception_handler(SpecGenAIError)
    async def domain_exception_handler(request: Request, exc: SpecGenAIError):
        logger.warning("Domain error [%s]: %s", type(exc).__name__, exc)
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc), "type": type(exc).__name__},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
