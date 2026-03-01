"""
Production FastAPI server with lifecycle management.

Handles:
- DSPy LLM configuration
- Pipeline initialization and retriever injection
- Graceful startup/shutdown
- CORS and middleware
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

import dspy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agents.autogen_consult import MedicalConsultTeam
from agents.orchestrator import AgentOrchestrator, OrchestratorConfig
from agents.medical_agent import MedicalQAAgent
from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from core.modules import MedicalQAPipeline
from retrieval.retriever import MedicalRetriever
from retrieval.vectorstore import PineconeManager
from optimization.feedback import FeedbackLoop

logger = get_logger("server")

# ── Global state (initialized during lifespan) ───────────────────────────────
_state: dict = {}


def _initialize_dspy() -> None:
    """Configure DSPy with the LLM backend."""
    settings = get_settings()
    lm = dspy.LM(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    dspy.configure(lm=lm)
    logger.info(f"DSPy configured with model={settings.llm_model}")


def _initialize_pipeline() -> tuple[MedicalQAPipeline, MedicalRetriever, AgentOrchestrator]:
    """Initialize the full pipeline stack."""
    # Retriever
    pinecone_mgr = PineconeManager()
    retriever = MedicalRetriever(pinecone_manager=pinecone_mgr)

    # Pipeline
    pipeline = MedicalQAPipeline(retriever_fn=retriever)

    # Medical QA Agent
    agent = MedicalQAAgent(pipeline=pipeline)

    # Orchestrator
    orchestrator = AgentOrchestrator(
        config=OrchestratorConfig(
            confidence_threshold=get_settings().eval_confidence_threshold,
            enable_self_reflection=True,
        )
    )
    orchestrator.set_pipeline(pipeline)
    orchestrator.set_retriever(retriever)
    orchestrator.register_agent("medical_qa", agent)

    return pipeline, retriever, orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — startup and shutdown."""
    settings = get_settings()
    setup_logging(level=settings.log_level, json_output=True)

    logger.info("Starting DSPy Medical AI System...")
    start = time.perf_counter()

    # Initialize DSPy
    _initialize_dspy()

    # Initialize pipeline
    pipeline, retriever, orchestrator = _initialize_pipeline()

    # Feedback loop
    feedback_loop = FeedbackLoop()
    feedback_loop.load_from_disk()

    # Store in app state
    _state["pipeline"] = pipeline
    _state["retriever"] = retriever
    _state["orchestrator"] = orchestrator
    _state["agent"] = orchestrator._agents.get("medical_qa")
    _state["feedback"] = feedback_loop

    # AutoGen Consult Team (shares the same retriever)
    consult_team = MedicalConsultTeam(retriever=retriever)
    _state["consult_team"] = consult_team

    elapsed = time.perf_counter() - start
    logger.info(f"System initialized in {elapsed:.1f}s")

    yield

    logger.info("Shutting down DSPy Medical AI System...")
    _state.clear()


def get_pipeline() -> MedicalQAPipeline:
    return _state["pipeline"]


def get_orchestrator() -> AgentOrchestrator:
    return _state["orchestrator"]


def get_agent() -> MedicalQAAgent:
    return _state["agent"]


def get_feedback() -> FeedbackLoop:
    return _state["feedback"]


def get_consult_team() -> MedicalConsultTeam:
    return _state["consult_team"]


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title="DSPy Medical AI System",
        description=(
            "Self-optimizing medical AI with declarative LLM pipelines, "
            "automatic prompt optimization, and multi-agent reasoning."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files and templates
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Register routes
    from api.routes import register_routes
    register_routes(app)

    return app
