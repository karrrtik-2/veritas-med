#  Veritas Med System 

> A production-grade, self-optimizing medical question-answering platform built on DSPy declarative LLM pipelines, LangChain retrieval, Pinecone vector search, and **AutoGen multi agent collaboration**. Designed for containerized deployment on AWS.(Latin veritas = truth; speaks to evidence-grounded responses)

**Core Stack:** DSPy · LangChain · Pinecone · FastAPI · **AutoGen** · OpenAI · Docker · AWS

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [System Components](#system-components)
- [Request Lifecycle](#request-lifecycle)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Optimization and Evaluation](#optimization-and-evaluation)
- [Docker](#docker)
- [AWS Deployment](#aws-deployment)

---

## Overview

The DSPy Medical AI System is a modular, evidence-grounded medical QA platform that combines retrieval-augmented generation (RAG) with declarative LLM programming. Its pipeline spans intent classification, multi-hop retrieval, chain-of-thought reasoning, fact verification, and safety enforcement — all with structured, Pydantic-validated outputs. Prompts are automatically optimized via DSPy compilers (BootstrapFewShot, MIPROv2, COPRO), so the system continuously improves as it accumulates feedback.

The platform exposes a FastAPI service with an interactive web chat UI, a versioned JSON API, and an optional AutoGen-powered multi-agent consultation mode staffed by specialized AI roles (Chief Medical Officer, Medical Researcher, and Safety Reviewer).

---

## Key Features

| Capability | Details |
|---|---|
| **Self-Optimizing Pipelines** | DSPy-native modules with automatic prompt/few-shot optimization via BootstrapFewShot, MIPROv2, and COPRO |
| **Retrieval-Augmented Generation** | LangChain + Pinecone vector store with sentence-transformer embeddings and context re-ranking |
| **Structured Reasoning** | Seven-stage pipeline: query analysis → retrieval → ranking → reasoning → verification → synthesis → safety |
| **Clinical Entity Extraction** | Automatic extraction of conditions, drugs, symptoms, and procedures from natural-language queries |
| **Multi-Agent Consultation** | AutoGen `SelectorGroupChat` team with CMO, Researcher, and Safety Reviewer roles |
| **Confidence-Based Refinement** | `AgentOrchestrator` iterates additional retrieval rounds when confidence falls below threshold |
| **Diagnosis Module** | Conditionally activated for diagnosis-intent queries; produces structured differential output |
| **Safety Enforcement** | Dedicated `SafetyGuard` module applies warnings, disclaimers, and escalation flags |
| **Evaluation Framework** | Composable DSPy-compatible metrics for factual accuracy, relevance, completeness, and safety |
| **Structured Observability** | Structured logging via `structlog`; health, stats, and feedback endpoints built-in |
| **Type-Safe Outputs** | All pipeline outputs validated against Pydantic v2 schemas |
| **Production-Ready Serving** | Uvicorn + FastAPI with configurable workers, CORS, and Docker health checks |

---

## Architecture

```text
┌─────────────────────┐        ┌──────────────────────────────┐
│   User  /  Client   │──────► │     FastAPI  API  Layer      │
└─────────────────────┘        └──────────────────────────────┘
                                          │              │
                                          ▼              ▼
                               ┌──────────────┐  ┌──────────────────┐
                               │ /api/v1/chat │  │ /api/v1/consult  │
                               └──────────────┘  └──────────────────┘
                                       │                  │
                                       ▼                  ▼
                             ┌────────────────┐  ┌──────────────────────┐
                             │ MedicalQAAgent │  │  MedicalConsultTeam  │
                             └────────────────┘  └──────────────────────┘
                                       │            	  │ CMO Agent
                                       │            	  │ Researcher Agent
                                       │            	  │ Safety Reviewer
                                       ▼                  │
                             ┌──────────────────┐         ▼
                             │ AgentOrchestrator│  ┌───────────────────────┐
                             └──────────────────┘  │search_medical_database│
                                       │           └───────────────────────┘
                                       ▼
                             ┌──────────────────┐
                             │ MedicalQAPipeline│
                             └──────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
             QueryAnalyzer     MedicalRetriever     ContextRanker
                    │                  │                   │
                    └──────────────────┴──────────────────►│
                                                           ▼
                                                   MedicalReasoner
                                                           │
                                                     FactVerifier
                                                           │
                                                  AnswerSynthesizer
                                                           │
                                                     SafetyGuard
                                                           │
                                              ┌────────────┴─────────┐
                                              ▼                      ▼
                                      Diagnosis Module        MedicalResponse
                                      (intent = diagnosis)        (JSON)
                                              │                       │
                                              └───────────┬───────────┘
                                                          ▼
                                                   Client Response
```

---

## System Components

```text
┌──────────────────────────────────────────────────────────────┐
│  Interface Layer                                             │
│    Web Chat UI  ·  REST Endpoints                            │
├──────────────────────────────────────────────────────────────┤
│  Application Layer                                           │
│    api/server.py → api/routes.py → agents/* → core/modules.py│
├──────────────────────────────────────────────────────────────┤
│  Knowledge Layer                                             │
│    retrieval/indexer.py                                      │
│        → retrieval/embeddings.py                             │
│        → retrieval/vectorstore.py  ←  retrieval/retriever.py│
├──────────────────────────────────────────────────────────────┤
│  Operations Layer                                            │
│    optimization/evaluators.py                                │
│    optimization/optimizers.py                                │
│    optimization/feedback.py                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Request Lifecycle

1. The client sends a natural-language query to `POST /api/v1/chat`.
2. `MedicalQAAgent` checks its response cache and conversation history, then delegates to `AgentOrchestrator`.
3. `QueryAnalyzer` classifies intent (diagnosis, treatment, symptoms, drug info, etc.), extracts clinical entities, scores complexity, and decomposes the query into sub-questions.
4. `MedicalRetriever` fetches top-K passages from Pinecone using sentence-transformer embeddings; `ContextRanker` re-scores and selects the highest-quality context chunks.
5. `MedicalReasoner` (chain-of-thought) generates evidence-grounded conclusions with an explicit reasoning trace.
6. `FactVerifier` cross-checks each claim against the retrieved evidence and assigns a verification status (verified, partially verified, unverified, or contradicted).
7. `AnswerSynthesizer` produces a patient-friendly, structured response.
8. `SafetyGuard` applies appropriate warnings, disclaimers, and emergency escalation flags based on query severity.
9. If intent is `diagnosis`, the `Diagnosis Module` generates a structured differential with severity and recommended next steps.
10. The API returns a structured `MedicalResponse` containing the answer, confidence score, safety level, optional reasoning trace, and cited sources.

---

## Project Structure

```text
app.py                          # CLI entry point: serve / optimize / evaluate / index
api/
  server.py                     # FastAPI app factory and middleware
  routes.py                     # Route definitions and request/response models
agents/
  medical_agent.py              # MedicalQAAgent: caching, history, query reformulation
  orchestrator.py               # AgentOrchestrator: confidence-based iterative refinement
  autogen_consult.py            # AutoGen multi-agent consult team (CMO, Researcher, Safety Reviewer)
  autogen_tools.py              # RAG tools exposed to AutoGen agents
  retrieval_agent.py            # Specialized retrieval agent
  synthesis_agent.py            # Specialized synthesis agent
  verification_agent.py         # Specialized verification agent
core/
  modules.py                    # Trainable DSPy modules (QueryAnalyzer, Reasoner, Verifier, etc.)
  signatures.py                 # DSPy Signature definitions
  schemas.py                    # Pydantic v2 output schemas
  reasoning.py                  # Reasoning helpers and chain-of-thought utilities
retrieval/
  embeddings.py                 # Sentence-transformer embedding wrapper
  vectorstore.py                # Pinecone vector store interface
  indexer.py                    # Document chunking and indexing pipeline
  retriever.py                  # Retriever bridge between LangChain and DSPy
optimization/
  evaluators.py                 # DSPy-compatible evaluation metrics
  optimizers.py                 # Optimizer wrappers (BootstrapFewShot, MIPROv2, COPRO)
  datasets.py                   # Training / evaluation dataset utilities
  feedback.py                   # User feedback collection and storage
scripts/
  index_documents.py            # CLI: index documents into Pinecone
  evaluate.py                   # CLI: run evaluation suite
  optimize.py                   # CLI: run optimization workflow
config/
  settings.py                   # Pydantic-Settings configuration with env-variable support
  logging_config.py             # Structured logging setup (structlog)
templates/                      # Jinja2 templates for the web chat UI
static/                         # CSS assets
Dockerfile                      # Multi-stage-ready container build
requirements.txt                # Python dependencies
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| OpenAI API key | GPT-4o access recommended |
| Pinecone account | Serverless index on AWS `us-east-1` |
| Docker (optional) | For containerized deployment |
| AWS account (optional) | For ECR + EC2 deployment |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp template.sh .env
```

Open `.env` and set the required values:

```bash
# Required
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Optional overrides (defaults shown)
LLM_MODEL=openai/gpt-4o
PINECONE_INDEX_NAME=medical-chatbot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

### 3. Index medical documents

Place source documents (PDF or text) in the `data/` directory, then run:

```bash
python app.py --index
```

### 4. Start the API server

```bash
python app.py
```

The server starts with the following endpoints available:

| Endpoint | URL |
|---|---|
| Web Chat UI | `http://localhost:8080/` |
| Interactive API Docs | `http://localhost:8080/docs` |
| Health Check | `http://localhost:8080/api/v1/health` |

---

## Configuration Reference

All settings are defined in `config/settings.py` and can be overridden via environment variables or a `.env` file.

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-4o` | Primary language model |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature for generation |
| `REASONING_MODEL` | `openai/gpt-4o` | Model used for chain-of-thought reasoning |
| `REASONING_TEMPERATURE` | `0.0` | Temperature for reasoning (deterministic) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for retrieval |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector dimensionality |
| `PINECONE_INDEX_NAME` | `medical-chatbot` | Pinecone index name |
| `RETRIEVAL_TOP_K` | `5` | Passages fetched per query |
| `RETRIEVAL_RERANK_TOP_K` | `3` | Passages kept after re-ranking |
| `OPTIMIZER_STRATEGY` | `bootstrap_fewshot` | DSPy optimizer: `bootstrap_fewshot`, `mipro_v2`, `copro`, `none` |
| `EVAL_SAMPLE_SIZE` | `50` | Number of examples per evaluation run |
| `EVAL_CONFIDENCE_THRESHOLD` | `0.7` | Minimum acceptable confidence score |
| `CHUNK_SIZE` | `512` | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks (tokens) |
| `SERVER_WORKERS` | `4` | Uvicorn worker processes |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web chat UI |
| `POST` | `/api/v1/chat` | Primary structured medical QA endpoint |
| `POST` | `/api/v1/consult` | Multi-agent AutoGen consultation endpoint |
| `POST` | `/api/v1/feedback` | Submit user feedback on a response |
| `POST` | `/api/v1/evaluate` | Evaluate a single query/response pair |
| `GET` | `/api/v1/health` | System health status |
| `GET` | `/api/v1/stats` | Runtime metrics and feedback statistics |
| `POST` | `/get` | Legacy form-based chat endpoint (backward-compatible) |
| `POST` | `/consult` | Legacy form-based consult endpoint (backward-compatible) |

### Example: Medical QA request

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the early signs of type 2 diabetes?",
    "include_reasoning": true,
    "include_sources": true
  }'
```

**Response fields**

| Field | Type | Description |
|---|---|---|
| `answer` | `string` | Patient-friendly evidence-grounded answer |
| `confidence` | `float` | Aggregate confidence score (0.0 – 1.0) |
| `intent` | `string` | Classified query intent |
| `safety_level` | `string` | `safe`, `caution`, `unsafe`, or `requires_professional` |
| `reasoning_trace` | `object` | Step-by-step reasoning trace (if requested) |
| `sources` | `array` | Retrieved source passages with metadata (if requested) |
| `structured_diagnosis` | `object` | Differential diagnosis output (diagnosis-intent queries only) |
| `trace_id` | `string` | Unique request identifier for observability |
| `latency_ms` | `float` | End-to-end response latency in milliseconds |

---

## Optimization and Evaluation

The system supports continuous improvement through DSPy's compiler-based prompt optimization and a composable evaluation framework.

```bash
# Run the full optimization workflow (uses configured optimizer strategy)
python app.py --optimize

# Run the evaluation suite against the default dataset
python app.py --evaluate

# Script-level execution with additional options
python -m scripts.optimize --strategy mipro_v2
python -m scripts.evaluate --dataset data/eval_set.json --sample-size 20
```

**Evaluation metrics** (weighted composite score):

| Metric | Default Weight | Description |
|---|---|---|
| Factual Accuracy | 35% | LLM-judged accuracy against ground truth |
| Relevance | 25% | How directly the response addresses the query |
| Completeness | 20% | Coverage of all medically relevant aspects |
| Safety | 20% | Presence of appropriate warnings and disclaimers |

Optimized pipeline weights are persisted to `optimized_pipelines/` and automatically loaded on server start.

---

## Docker

```bash
# Build the image
docker build -t medical-ai-system .

# Run with environment file
docker run -p 8080:8080 --env-file .env medical-ai-system
```

The container exposes port `8080` and includes a health check that polls `/api/v1/health` every 30 seconds.

---

## AWS Deployment

The recommended deployment target is an EC2 instance acting as a self-hosted GitHub Actions runner, with container images stored in Amazon ECR.

### Steps

1. **IAM permissions** — Create an IAM user or role with `AmazonEC2ContainerRegistryFullAccess` and the necessary EC2 permissions.
2. **ECR repository** — Create a private ECR repository to store Docker images.
3. **EC2 instance** — Provision an Ubuntu instance (t3.medium or larger recommended), install Docker, and register it as a GitHub Actions self-hosted runner.
4. **GitHub repository secrets** — Add the following secrets to your repository:

   | Secret | Description |
   |---|---|
   | `AWS_ACCESS_KEY_ID` | IAM access key ID |
   | `AWS_SECRET_ACCESS_KEY` | IAM secret access key |
   | `AWS_DEFAULT_REGION` | AWS region (e.g., `us-east-1`) |
   | `ECR_REPO` | Full ECR repository URI |
   | `OPENAI_API_KEY` | OpenAI API key |
   | `PINECONE_API_KEY` | Pinecone API key |

5. **CI/CD pipeline** — Push to the configured branch to trigger the GitHub Actions workflow, which will build the Docker image, push it to ECR, pull it onto the EC2 runner, and restart the container.

---

> **Disclaimer:** This system is intended for research and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`

## Notes

- This system is for medical information assistance, not a replacement for professional medical care.
- For emergencies, users should seek immediate help from licensed healthcare providers or local emergency services.
- Core pipeline modules are implemented with typed, composable LLM building blocks, and the project also includes AutoGen-based team consultation.
