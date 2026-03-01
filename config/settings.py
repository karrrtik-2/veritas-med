"""
Centralized settings management with Pydantic validation.
Supports environment-driven configuration for all pipeline components.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
OPTIMIZED_DIR = PROJECT_ROOT / "optimized_pipelines"
EVAL_DIR = PROJECT_ROOT / "evaluation_results"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class OptimizerStrategy(str, Enum):
    BOOTSTRAP_FEWSHOT = "bootstrap_fewshot"
    MIPRO_V2 = "mipro_v2"
    COPRO = "copro"
    NONE = "none"


class Settings(BaseSettings):
    """Global settings for the DSPy Medical AI System."""

    # ── API Keys ──────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key (optional)")

    # ── LLM Configuration ─────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "openai/gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Reasoning LLM (for complex CoT tasks) ────────────────────────────
    reasoning_model: str = "openai/gpt-4o"
    reasoning_temperature: float = 0.0

    # ── Embedding Configuration ───────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Vector Store ──────────────────────────────────────────────────────
    pinecone_index_name: str = "medical-chatbot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    retrieval_top_k: int = 5
    retrieval_rerank_top_k: int = 3

    # ── Optimization ──────────────────────────────────────────────────────
    optimizer_strategy: OptimizerStrategy = OptimizerStrategy.BOOTSTRAP_FEWSHOT
    optimization_max_bootstrapped_demos: int = 4
    optimization_max_labeled_demos: int = 8
    optimization_num_candidates: int = 10
    optimization_num_threads: int = 4
    optimization_teacher_model: str = "openai/gpt-4o"

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_sample_size: int = 50
    eval_confidence_threshold: float = 0.7
    eval_metric_weights: dict = Field(
        default_factory=lambda: {
            "factual_accuracy": 0.35,
            "relevance": 0.25,
            "completeness": 0.20,
            "safety": 0.20,
        }
    )

    # ── Document Processing ───────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Server ────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    server_workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Tracing / Observability ───────────────────────────────────────────
    enable_tracing: bool = True
    log_level: str = "INFO"

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _resolve_openai_key(cls, v: str) -> str:
        return v or os.getenv("OPENAI_API_KEY", "")

    @field_validator("pinecone_api_key", mode="before")
    @classmethod
    def _resolve_pinecone_key(cls, v: str) -> str:
        return v or os.getenv("PINECONE_API_KEY", "")

    def inject_env(self) -> None:
        """Push keys into os.environ for downstream SDKs."""
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.pinecone_api_key:
            os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for global settings."""
    settings = Settings()
    settings.inject_env()
    return settings
