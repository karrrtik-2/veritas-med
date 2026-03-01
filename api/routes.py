"""
API route definitions for the DSPy Medical AI System.

Routes:
- GET  /                → Chat UI
- POST /api/v1/chat     → Primary chat endpoint (JSON)
- POST /get             → Legacy chat endpoint (form data, backward compat)
- POST /api/v1/feedback → Submit user feedback
- GET  /api/v1/health   → Health check
- GET  /api/v1/stats    → System statistics
- POST /api/v1/evaluate → Run evaluation on a query
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config.logging_config import get_logger

logger = get_logger("routes")
templates = Jinja2Templates(directory="templates")


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Medical query")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation turns [{user, assistant}]"
    )
    include_reasoning: bool = Field(False, description="Include reasoning trace in response")
    include_sources: bool = Field(True, description="Include retrieved sources")


class ConsultRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Medical query for the consult team")
    include_chat_history: bool = Field(False, description="Include full agent chat history in response")


class ConsultResponse(BaseModel):
    final_answer: str
    rounds: int
    chat_history: Optional[list[dict[str, str]]] = None
    trace_id: str
    latency_ms: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    intent: str
    reasoning_trace: Optional[dict] = None
    sources: Optional[list[dict]] = None
    safety_level: str
    trace_id: str
    latency_ms: float
    structured_diagnosis: Optional[dict] = None


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: float = Field(ge=0.0, le=1.0)
    feedback_text: str = ""


class EvaluateRequest(BaseModel):
    query: str
    expected_answer: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    uptime_seconds: float


# ─── Route Registration ──────────────────────────────────────────────────────

_start_time = time.time()


def register_routes(app: FastAPI) -> None:
    """Register all API routes on the FastAPI app."""

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the chat UI."""
        return templates.TemplateResponse("chat.html", {"request": request})

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat_api(req: ChatRequest):
        """Primary JSON chat endpoint with full structured response."""
        from api.server import get_agent

        trace_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        agent = get_agent()

        if req.conversation_history:
            response = agent.answer_with_context(
                query=req.query,
                conversation_history=req.conversation_history,
            )
        else:
            response = agent.answer(query=req.query)

        latency_ms = (time.perf_counter() - start) * 1000

        result = ChatResponse(
            answer=response.answer,
            confidence=response.reasoning_trace.overall_confidence,
            intent=response.query_analysis.intent.value,
            safety_level=response.safety.level.value,
            trace_id=trace_id,
            latency_ms=round(latency_ms, 1),
        )

        if req.include_reasoning:
            result.reasoning_trace = response.reasoning_trace.model_dump()

        if req.include_sources:
            result.sources = [
                rc.model_dump() for rc in response.retrieved_contexts
            ]

        if response.structured_diagnosis:
            result.structured_diagnosis = response.structured_diagnosis.model_dump()

        logger.info(
            f"[{trace_id}] Chat response in {latency_ms:.0f}ms "
            f"(confidence={result.confidence:.2f})",
            extra={"trace_id": trace_id},
        )

        return result

    @app.post("/get")
    async def chat_legacy(msg: str = Form(...)):
        """Legacy form-based chat endpoint for backward compatibility with the original UI."""
        from api.server import get_agent

        agent = get_agent()
        response = agent.answer(query=msg)
        return response.answer

    @app.post("/api/v1/feedback")
    async def submit_feedback(req: FeedbackRequest):
        """Submit user feedback for continuous improvement."""
        from api.server import get_feedback

        feedback = get_feedback()
        feedback.add_user_feedback(
            query=req.query,
            response=req.response,
            rating=req.rating,
            text=req.feedback_text,
        )
        return {"status": "ok", "message": "Feedback recorded"}

    @app.post("/api/v1/evaluate")
    async def evaluate_query(req: EvaluateRequest):
        """Evaluate a query against the pipeline with metrics."""
        from api.server import get_agent

        agent = get_agent()
        response = agent.answer(query=req.query, use_cache=False)

        return {
            "query": req.query,
            "answer": response.answer,
            "confidence": response.reasoning_trace.overall_confidence,
            "safety_level": response.safety.level.value,
            "verification_status": response.verification.status.value,
            "verification_confidence": response.verification.confidence,
        }

    @app.post("/api/v1/consult", response_model=ConsultResponse)
    async def consult_api(req: ConsultRequest):
        """Multi-agent medical consultation endpoint (AutoGen-powered)."""
        from api.server import get_consult_team

        trace_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        team = get_consult_team()
        result = await team.arun(user_query=req.query)

        latency_ms = (time.perf_counter() - start) * 1000

        response = ConsultResponse(
            final_answer=result["final_answer"],
            rounds=result["rounds"],
            trace_id=trace_id,
            latency_ms=round(latency_ms, 1),
        )

        if req.include_chat_history:
            response.chat_history = result["chat_history"]

        logger.info(
            f"[{trace_id}] Consult response in {latency_ms:.0f}ms "
            f"({result['rounds']} rounds)",
            extra={"trace_id": trace_id},
        )

        return response

    @app.post("/consult")
    async def consult_legacy(msg: str = Form(...)):
        """Simple form-based consult endpoint for the chat UI."""
        from api.server import get_consult_team

        team = get_consult_team()
        result = await team.arun(user_query=msg)
        return result["final_answer"]

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        from config.settings import get_settings

        settings = get_settings()
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            model=settings.llm_model,
            uptime_seconds=round(time.time() - _start_time, 1),
        )

    @app.get("/api/v1/stats")
    async def stats():
        """System statistics."""
        from api.server import get_feedback

        try:
            feedback = get_feedback()
            feedback_stats = feedback.stats
        except Exception:
            feedback_stats = {}

        return {
            "uptime_seconds": round(time.time() - _start_time, 1),
            "feedback": feedback_stats,
        }
