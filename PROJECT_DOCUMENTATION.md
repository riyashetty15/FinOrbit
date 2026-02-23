# FinOrbit - Complete Project Documentation

**Version:** 1.2.1 (Routing Pre-emption Fix + Integration Test Suite)
**Status:** Active Development

This comprehensive documentation covers the complete architecture, implementation, evaluation results, and improvement history of the FinOrbit financial assistant system.

---

# 1. System Architecture Overview

This section provides a comprehensive overview of the FinOrbit financial assistant architecture, detailing the transition to a layered, service-oriented design.

## 1.1 High-Level Architecture

The system follows a **Layered Architecture** designed to separate orchestration, and decision logic.

```mermaid
graph TD
    User[User Client] --> API[FastAPI Server]

    subgraph "Layer 1: Orchestration & Safety"
        API --> Guardrails[Guardrail Pipeline]
        Guardrails --> Router[Semantic Router]
        Router --> Orchestrator[Multi-Agent Orchestrator]
    end

    subgraph "Layer 2: Services & Workflows"
        Orchestrator --> RetService[Retrieval Service]
        Orchestrator --> Workflows[Specialist Workflows]

        Workflows -->|Loan/Tax/Invest| Decision[Decision Engine]
        RetService -->|Evidence Pack| Decision
    end

    subgraph "Layer 3: Tooling (MCP Layer)"
        Workflows --> MathTools[Finance Math Tools]
        RetService --> RAGServer[RAG MCP Server]
    end

    subgraph "Layer 4: Data Infrastructure"
        RAGServer --> PGVector[(Postgres + PGVector)]
        RAGServer --> Ingestion[Ingestion Pipeline]
    end
```

## 1.2 Component Breakdown

### A. Orchestration Layer
- **Finance Orchestrator (`backend/finance_agent.py`)**: Conversational entry point. Manages session state and coordinates high-level user intent.
- **Multi-Agent Orchestrator (`backend/core/multi_agent_orchestrator.py`)**: Decomposes complex queries into sub-queries and executes them in dependency waves using `asyncio.gather()` for maximum parallelism. Independent sub-queries run concurrently; dependent ones run in order.
- **Router (`backend/core/router.py`)**: Determines query domain using a three-tier approach:
  1. **Domain pre-emption**: If investment terms (SIP, mutual funds, ETF, NAV, etc.) or retirement terms (EPF, NPS, VPF, gratuity, pension, etc.) are detected without specific regulatory document signals, routes directly to the appropriate specialist agent — bypassing both the RAG heuristic **and** the LLM classifier entirely. The `_preempted` flag prevents the `elif not needs_rag:` fallback from re-triggering `_simple_rag_heuristic`.
  2. **LLM-based RAG detection**: Classifies whether query needs knowledge-base lookup. Skipped when pre-emption fired.
  3. **Keyword pattern matching**: `INTENT_MAP` regex patterns covering 7 agent types with `MULTI_DOMAIN_PRIORITY` rules for overlapping domains.
- **LLM Provider (`backend/core/llm_provider.py`)**: Singleton factory supporting `openai` and `gemini` providers, selected via `LLM_PROVIDER` env var. Used by the router and main finance agent.
- **State Manager (`backend/core/conversation_context.py`)**: Tracks user profile (age, income, occupation) and last agent across conversation turns. Persists to PostgreSQL (24h TTL) when `DATABASE_URL` is set; falls back to in-memory if unavailable.

### B. Service Layer
- **Retrieval Service (`backend/services/retrieval_service.py`)**:
  - Converts natural language queries into verified evidence (`EvidencePack`).
  - Semantically routes to the correct RAG module, retrieves chunks, LLM-verifies relevance, and scores coverage.
  - Accepts `trace_id` for end-to-end request correlation.
- **Decision Engine (`backend/services/decision_engine.py`)**:
  - Enforces a strict `DecisionOutput` schema (Recommendations, Reasoning, Assumptions, Risks, Disclaimers).
  - Prevents unstructured "chatty" responses.

### C. Specialist Workflows (Domain Logic)
Structured workflows handling specific financial domains. All accept an `EvidencePack` for regulatory grounding.
- **Credits & Loans** (`backend/agents/specialist/credits_loans.py`): Debt profiles, EMI eligibility.
- **Investment Coach** (`backend/agents/specialist/investment_coach.py`): SIP planning, portfolio diversification.
- **Tax Planner** (`backend/agents/specialist/tax_planner.py`): 2024/2025 tax regime calculations and deduction logic.
- **Retirement Planner** (`backend/agents/specialist/retirement_planner.py`): Corpus planning, EPF/NPS/VPF rules, withdrawal assumptions.
- **Insurance Analyzer** (`backend/agents/specialist/insurance_analyzer.py`): Coverage gap analysis, suitability checks.

### D. Guardrails Layer (`backend/core/pipeline.py`)
A unified async pipeline ensuring safety at three stages. All checks are parallelized via `asyncio.gather()` + `asyncio.to_thread()`:

**Pre-Validation (input safety):**
1. PII Detection + Content Risk Filter — sequential blocking checks; early exit if either fires
2. Age/Category Guard + Mis-Selling Guard + Audit Logger — run in parallel
3. Compliance Engine — sequential (may be I/O-bound)

**Post-Validation (output quality), all parallel:**
1. Grounding Check (skipped for `intent=general`)
2. Regulatory Consistency Check
3. Suitability Check
4. Tone & Clarity Check

### E. Tooling Layer (MCP Architecture)

#### 1. RAG Server (`Finorbit_RAG/`)
- **Role**: Dedicated Context Provider (MCP).
- **Function**: Indexes PDFs/Docs into PGVector. Exposes `/query` API searching specific modules (Credit, Tax, etc.).
- **Key Feature**: Module siloing — loan queries never mix with mutual fund documents.
- `X-Trace-ID` HTTP header forwarded from LLM backend for cross-service log correlation.

#### 2. Finance Math Tools (`backend/tools/finance_math.py`)
- `calculate_emi`: Loan amortization.
- `estimate_tax_new_regime_2024`: Indian tax slab estimates.
- `calculate_sip_returns`: Future value of investments.
- Prevents LLM arithmetic hallucinations.

## 1.3 How the "MCP Server" Works

1. **Separation**: RAG logic lives in a separate process (`Finorbit_RAG/main.py`, port 8081) from the LLM logic (`Finorbit_LLM`, port 8000).
2. **Communication**: `RetrievalService` → `backend/tools/rag_tool.py` → HTTP POST to RAG Server, with `X-Trace-ID` header.
3. **Flow** (example: *"What are the NBFC rules?"*):
   - Router classifies: `module: credit`
   - Retrieval Service calls RAG Tool: `module=credit, query='NBFC rules'`
   - RAG Server vector-searches `credit_chunks` in Postgres
   - Retrieval Service verifies relevance → returns `EvidencePack`

## 1.4 Current Implementation Status

| Component | Status | Location |
| :--- | :--- | :--- |
| **FastAPI Server** | Active | `backend/server.py` |
| **Multi-Agent Orchestrator** | Active (parallel wave execution) | `backend/core/multi_agent_orchestrator.py` |
| **Router** | Active (domain pre-emption + expanded keywords) | `backend/core/router.py` |
| **LLM Provider Factory** | Active (OpenAI \| Gemini) | `backend/core/llm_provider.py` |
| **Validation Pipeline** | Active (fully async + parallel) | `backend/core/pipeline.py` |
| **Retrieval Service** | Active (trace_id aware) | `backend/services/retrieval_service.py` |
| **State Manager** | Active (PostgreSQL + in-memory fallback) | `backend/core/conversation_context.py` |
| **RAG Tool** | Active (X-Trace-ID forwarding) | `backend/tools/rag_tool.py` |
| **Math Tools** | Active | `backend/tools/finance_math.py` |
| **Decision Engine** | Active | `backend/services/decision_engine.py` |
| **Metrics Endpoint** | Active (`GET /metrics`) | `backend/server.py` |
| **Domain Specialists** | Active | `backend/agents/specialist/` |
| **DB Migration** | Ready to apply | `backend/migrations/001_conversation_contexts.sql` |

---

# 2. Production-Grade RAG Implementation

## 2.1 Components Implemented

### 2.1.1 EvidencePack Schema (`backend/services/retrieval_service.py`)

```python
@dataclass
class Citation:
    doc_id: str          # Document identifier
    source: str          # Source filename
    page: Optional[int]  # Page number
    chunk_id: str        # Chunk identifier
    text: str            # Citation text
    score: float         # Relevance score
    metadata: Dict       # Full metadata

@dataclass
class EvidencePack:
    module: str                     # Module queried
    query: str                      # Original query
    citations: List[Citation]       # Verified citations
    confidence: float               # Verification confidence
    coverage: str                   # "sufficient" | "partial" | "insufficient"
    filters: Dict[str, Any]        # Applied filters
    rejection_reason: Optional[str] # If coverage insufficient
```

**Coverage Scoring Rules:**
- `sufficient`: ≥3 verified citations OR ≥2 from same authoritative doc with confidence ≥0.7
- `partial`: 1-2 verified citations with confidence ≥0.5, or weak relevance
- `insufficient`: 0 verified citations

### 2.1.2 RouteIntent Schema (`backend/core/router.py`)

```python
@dataclass
class RouteIntent:
    module: str              # Target agent/module
    needs_evidence: bool     # Whether RAG grounding required
    jurisdiction: str        # Geographic context (default: "IN")
    time_sensitivity: str    # "high" | "low"
    query_type: str          # "info" | "advice" | "compliance"
```

**Evidence Need Detection triggers on:**
- Specific regulatory body mentions (RBI/SEBI/IRDAI) without investment domain context
- Regulatory document keywords: rules, circulars, master directions, guidelines
- Factual lookups: rates, limits, ratios
- Authority statements: "according to", "as per", "under section"

**Domain Pre-emption** (bypasses RAG heuristic AND LLM classifier):
- **Investment**: query contains strong investment terms (`_INVESTMENT_STRONG_TERMS`: SIP, mutual funds, ETF, NAV, PPF, ULIP, portfolio, stocks, etc.) AND does NOT contain specific regulatory document signals → routes directly to `investment_coach`.
- **Retirement**: query contains strong retirement terms (`_RETIREMENT_STRONG_TERMS`: EPF, NPS, VPF, pension, gratuity, superannuation, retirement, etc.) AND does NOT contain specific regulatory document signals → routes directly to `retirement_planner`.
- In both cases a `_preempted = True` flag prevents the `elif not needs_rag:` fallback path from overriding the decision via `_simple_rag_heuristic` (which would otherwise fire on "SEBI" or "RBI" mentions).

### 2.1.3 Optimistic Evidence Prefetch (`backend/server.py`)

Evidence retrieval now runs in parallel with LLM routing:

```python
# Start evidence fetch immediately if likely needed
_quick_needs_evidence = router_agent._detect_evidence_need(query)
if _quick_needs_evidence:
    _evidence_task = asyncio.create_task(retrieval_service.retrieve_evidence(...))

# Route query (overlaps with evidence fetch)
agent_type, intent = router_agent.route_with_context(...)
route_intent = router_agent.route_with_evidence_intent(...)

# Consume or cancel
if route_intent.needs_evidence and _evidence_task:
    evidence_pack = await _evidence_task   # already running
elif route_intent.needs_evidence:
    evidence_pack = await retrieval_service.retrieve_evidence(...)
else:
    if _evidence_task: _evidence_task.cancel()
```

**Expected gain:** 500ms–1.5s on evidence-needing queries.

### 2.1.4 RAG API Filters

**Filter Parameters (passed to RAG Server):**
- `module`: credit | investment | insurance | retirement | taxation
- `doc_type`, `year`, `issuer`, `regulator_tag`
- `year_min`, `jurisdiction`, `effective_date`, `version`
- `security`, `is_current`, `pii`
- `compliance_tags_any`: List[str] with IN operator

---

# 3. Deployment Guide

## 3.1 Quick Start

### Prerequisites
1. **Python 3.12+**
2. **PostgreSQL with pgvector** extension (or Docker)

## 3.2 Step 1: Database Setup

### Option A: Docker (Recommended)
```bash
cd Finorbit_RAG
docker-compose up -d postgres
```

### Option B: Local PostgreSQL
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE DATABASE financial_rag;
```

After the database is running, apply the conversation context migration:
```bash
psql $DATABASE_URL -f Finorbit_LLM/backend/migrations/001_conversation_contexts.sql
```

## 3.3 Step 2: Configure Environment Variables

### RAG Server (`Finorbit_RAG/.env`)
```dotenv
# Database
DB_HOST=localhost
DB_NAME=financial_rag
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432

# LLM Provider for RAG (openai or gemini)
LLM_PROVIDER=openai
GOOGLE_API_KEY=your_google_key   # if using gemini

# Server
API_PORT=8081
```

### LLM Backend (`Finorbit_LLM/.env`)
```dotenv
# LLM Provider (openai | gemini)
LLM_PROVIDER=openai               # or "gemini"
LLM_API_KEY=your_openai_key       # required when LLM_PROVIDER=openai
GOOGLE_API_KEY=your_google_key    # required when LLM_PROVIDER=gemini
CUSTOM_MODEL_NAME=gpt-4o-mini     # optional model override

# Database (conversation context persistence)
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/financial_rag

# RAG Service
RAG_API_URL=http://localhost:8081/query
RAG_HEALTH_CHECK_URL=http://localhost:8081/health
```

> **Note:** If `DATABASE_URL` is not set, conversation context falls back gracefully to in-memory storage. If `LLM_API_KEY` / `GOOGLE_API_KEY` is missing, the server raises a `RuntimeError` at startup.

## 3.4 Step 3: Ingest Documents

### Ingest sample documents (for development/testing)
```bash
cd Finorbit_LLM
python tests/ingest_sample_docs.py
```

### Ingest regulatory documents (batch CLI)
```bash
# Single file
python Finorbit_RAG/scripts/ingest_regulatory_docs.py \
  --file path/to/rbi_circular.pdf \
  --module credit --issuer RBI --doc-type circular --year 2024

# Entire directory
python Finorbit_RAG/scripts/ingest_regulatory_docs.py \
  --dir path/to/sebi_docs/ \
  --module investment --issuer SEBI --doc-type guideline

# Dry-run to preview without ingesting
python Finorbit_RAG/scripts/ingest_regulatory_docs.py \
  --dir path/to/docs/ --issuer IRDAI --dry-run

# With JSON report output
python Finorbit_RAG/scripts/ingest_regulatory_docs.py \
  --file my_doc.pdf --module taxation --issuer CBDT \
  --output-report report.json
```

## 3.5 Step 4: Start Services

```bash
# Terminal 1: RAG Server
cd Finorbit_RAG
uvicorn main:app --host 0.0.0.0 --port 8081

# Terminal 2: LLM Backend
cd Finorbit_LLM
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

## 3.6 Health & Metrics

```bash
# Health check
curl http://localhost:8000/health

# Metrics (latency P50/P95, agent counts, circuit breaker state)
curl http://localhost:8000/metrics | python3 -m json.tool

# RAG health
curl http://localhost:8081/health
```

---

### Step 2: Document Ingestion Scripts

**`tests/ingest_sample_docs.py`:**
- Removed `os.chdir()` (was mutating process CWD)
- Added `_REPO_ROOT` absolute path constant
- Different PDF per module (was ingesting same PDF into 3 modules → duplicates)
- Added upfront path validation + 3-attempt retry with exponential backoff

**New: `Finorbit_RAG/scripts/ingest_regulatory_docs.py`:**
- Full CLI with `--file`/`--dir`, `--issuer` (RBI/SEBI/IRDAI/CBDT auto-templates), `--module`, `--doc-type`, `--year`, `--dry-run`, `--output-report`
- 3-attempt retry, JSON summary report, exit code 0/1

---

### Step 3: Parallel Validation Pipeline (`backend/core/pipeline.py`)

All `.check()` and `.validate()` calls are now async via `asyncio.to_thread()`:

```
Pre-validation:
  Phase 1 (sequential, fast): PII + Content Risk → early exit if blocked
  Phase 2 (parallel): Age Guard + Mis-Selling + Audit Logger via asyncio.gather()
  Phase 3 (sequential): Compliance Engine

Post-validation (all parallel):
  asyncio.gather(Grounding, Regulatory, Suitability, Tone)
  If intent=general: skip Grounding, parallelize remaining 3
```

**Expected gain:** ~700ms–1.3s per request.

---

### Step 3: Parallel Routing + Evidence Retrieval (`backend/server.py`)

Optimistic evidence prefetch: `asyncio.create_task()` starts evidence retrieval while LLM routing runs. Task cancelled if routing determines evidence not needed. See Section 2.1.3.

**Expected gain:** ~500ms–1.5s on evidence-needing queries.

---

### Step 4: Multi-Agent Wave Execution (`backend/core/multi_agent_orchestrator.py`)

Sub-queries grouped by `depends_on` dependency into execution waves. Each wave runs concurrently:
```python
waves = self._build_execution_waves(sub_queries)
for wave in waves:
    wave_results = await asyncio.gather(
        *[self._execute_single_subquery(sq, profile, results, trace_id) for sq in wave],
        return_exceptions=True
    )
```

Supports both async and sync agents via `inspect.iscoroutinefunction`.
**Expected gain:** For 3 independent agents: T1+T2+T3 → max(T1,T2,T3) (~60–70% reduction).

---

### Step 5: `/metrics` Endpoint (`backend/server.py`)

```bash
GET /metrics
```

Returns:
```json
{
  "uptime_seconds": 3600.0,
  "router": {
    "cache_hits": 42, "cache_misses": 18, "llm_calls": 15,
    "p50_routing_ms": 120.0, "p95_routing_ms": 310.0,
    "circuit_breaker": {"is_open": false, "failures": 0}
  },
  "pipeline_validation": {"pii": {"pass": 59, "fail": 1}, ...},
  "agent_execution_counts": {"investment_coach": 23, "tax_planner": 14, ...},
  "specialist_agents_registered": 5
}
```

---

### Step 6: Trace ID Propagation

`trace_id` (from `TraceContext` in `server.py`) now flows to all sub-services:
- `retrieval_service.retrieve_evidence(..., trace_id=trace_id)`
- `orchestrator.process_complex_query(..., trace_id=trace_id)`
- RAG HTTP call: `headers["X-Trace-ID"] = trace_id`
- Compliance engine context: `{"trace_id": trace_id, ...}`

Enables end-to-end correlation across `backend.log` and `rag.log`.

---

### Step 7: LLM Provider Abstraction (`backend/core/llm_provider.py`)

New singleton factory supporting `openai` and `gemini`:

```python
from backend.core.llm_provider import get_llm_provider

provider = get_llm_provider()          # reads LLM_PROVIDER env var
text = await provider.async_complete(
    user_prompt="...", system_prompt="...", max_tokens=2048, temperature=0.3
)
```

- `LLM_PROVIDER=openai` (default): uses `LLM_API_KEY`
- `LLM_PROVIDER=gemini`: uses `GOOGLE_API_KEY`
- `CUSTOM_MODEL_NAME`: optional model override
- `_run_finance_agent()` in `server.py` uses this abstraction
- `RouterAgent` uses `provider.complete()` for RAG classification (replaced direct `OpenAI` client)
- `reset_provider()` available for testing

Config additions to `backend/config.py`:
```python
llm_provider: str = "openai"        # "openai" | "gemini"
google_api_key: Optional[str] = None
```

---

### Step 8: PostgreSQL Conversation Context (`backend/core/conversation_context.py`)

New async methods with lazy asyncpg pool + in-memory fallback:

```python
# Retrieve (24h TTL, falls back to in-memory)
ctx = await ConversationContext.get_context_async(conversation_id)

# Persist (INSERT … ON CONFLICT DO UPDATE, always updates in-memory too)
await ConversationContext.update_context_async(conversation_id, agent, profile_updates)
```

Apply migration before first run:
```bash
psql $DATABASE_URL -f backend/migrations/001_conversation_contexts.sql
```

If `DATABASE_URL` is not set: silently uses in-memory only (no startup error).

**Bug fixed (found during testing):** `update_context()` was ignoring `profile_updates` on first-time creation (profile initialized as `{}`). Fixed: first creation now uses `dict(profile_updates)`.

---

### Step 9: Expanded Routing Keywords

**`INTENT_MAP` additions (`backend/core/router.py`):**

| Agent | New Terms Added |
|---|---|
| `investment_coach` | ETF, NFO, small/mid/large/flexi-cap, index funds, FD, NCD, PPF, ULIP, NAV, AUM, SIP (+ plural `funds?` fix) |
| `retirement_planner` | EPF, employees' provident fund, VPF, gratuity, superannuation, PF withdrawal |
| `tax_planner` | TDS, GST, capital gains, HRA, Form 16, 80C, 80D |
| `fraud_shield` | scam, OTP fraud, cyber crime |

**New `MULTI_DOMAIN_PRIORITY` rules:**
- `{investment_coach, retirement_planner}` → `retirement_planner` (e.g., "EPF vs NPS")
- `{credits_loans, retirement_planner}` → `retirement_planner` (e.g., "loan against EPF")

**`FINORBIT_MODULE_KEYWORDS` expansion (`shared_config.py`):** Same new terms added to each module's keyword list for consistent RAG module routing and retrieval service classification.

---

## 5.3 Bug Fixes Found During v1.2 Testing

### Bug 1: `mutual funds?` plural not matched
`mutual fund` in `INTENT_MAP` and `_INVESTMENT_STRONG_TERMS` didn't match the plural "mutual funds" due to trailing `\b`. Fixed by adding `s?`: `mutual funds?`, `index funds?`, `exchange traded funds?`.

### Bug 2: `ConversationContext.update_context` first-creation data loss
On a brand-new conversation, `profile_updates` were silently dropped — profile initialized to `{}` instead of the provided values. Fixed: first-time creation now uses `dict(profile_updates)`.

---

### Integration Test Suite (`tests/test_citations.py`) — Complete Rewrite

The previous `test_citations.py` was a 3-case live test with no PASS/FAIL tracking. The rewritten file covers 47 test cases across 6 sections:

| Section | Cases | Coverage |
|---|---|---|
| 1 — Health & Metrics | 13 | `GET /health`, `GET /metrics` (all fields, p50/p95 latency) |
| 2 — Response Schema | 14 | All `QueryResponse` fields, types, pipeline steps |
| 3 — Routing Fixes | 8 | SEBI+investment, EPF/retirement, TDS, credit, insurance |
| 4 — Citation Extraction | 4+1 skip | RBI docs (5 citations confirmed), SEBI (skip if no docs) |
| 5 — Conversation Context | 4 | Two-turn context persistence, no crashes |
| 6 — Edge Cases | 3 | Whitespace rejection (422), missing field (422), profileHint (200) |

**Final result: 46 passed / 0 failed / 1 skipped.**

Usage:
```bash
# Full run (with RAG server and ingested documents)
python tests/test_citations.py

# Skip citation tests (when RAG store is empty)
python tests/test_citations.py --skip-citations
```

---

### Bug 3: `elif not needs_rag:` Fallback Overriding Pre-emption

**Root cause** (discovered in live testing):

The `route()` method had this structure:

```python
_preempted = False
if self._is_domain_investment_query(query):
    needs_rag = False
    _preempted = True
elif self._is_domain_retirement_query(query):
    needs_rag = False
    _preempted = True
else:
    needs_rag = self._simple_rag_heuristic(query)

# BUG: elif fires even when _preempted=True
if not needs_rag and not _preempted and self._gemini_configured:
    needs_rag, ... = self._ask_llm_needs_rag(query)   # correctly skipped
elif not needs_rag:   # ← FIRES when _preempted=True + LLM not called!
    needs_rag = self._simple_rag_heuristic(query)     # re-fires heuristic
```

When `_preempted=True`, the LLM block was correctly skipped. But the `elif not needs_rag:` fallback (intended for "no LLM available" cases) fired unconditionally, re-calling `_simple_rag_heuristic`. Since SEBI/RBI keywords appear in investment queries, the heuristic returned `True` → override to `rag_agent`.

**Fix:**
```python
elif not needs_rag and not _preempted:   # ← added `and not _preempted`
    needs_rag = self._simple_rag_heuristic(query)
```

**Effect:** "What are SEBI mutual fund regulations?" now correctly routes to `investment_coach`.

**Verified by direct router test (no API key required):**
```
[PASS] How does SEBI regulate mutual funds?    → investment_coach
[PASS] What are SEBI mutual fund regulations? → investment_coach
[PASS] What is EPF and how much can I withdraw? → retirement_planner
[PASS] SEBI/HO/IMD/2024 circular text?        → rag_agent (still RAG for doc lookups)
```

---

### New: `_is_domain_retirement_query()` Pre-emption

**Problem:** "What is EPF and how much can I withdraw?" was routing to `rag_agent` because the LLM classified "EPF withdrawal" as a regulatory document lookup. No retirement pre-emption existed.

**Fix:** Added `_RETIREMENT_STRONG_TERMS` and `_is_domain_retirement_query()` (mirrors the investment pattern):

```python
_RETIREMENT_STRONG_TERMS = re.compile(
    r"\b(epf|employees?\s+provident\s+fund|vpf|voluntary\s+provident\s+fund|"
    r"nps|national\s+pension|ppf|public\s+provident\s+fund|"
    r"retir(e|ement|ing)|pension|gratuity|superannuation|pf\s+withdrawal)\b",
    re.IGNORECASE,
)

def _is_domain_retirement_query(self, query: str) -> bool:
    has_retirement_terms = bool(self._RETIREMENT_STRONG_TERMS.search(query))
    has_specific_regulatory_lookup = bool(self._REGULATORY_LOOKUP_SIGNALS.search(query))
    return has_retirement_terms and not has_specific_regulatory_lookup
```

Note: Queries with the word " and " (e.g., "What is EPF **and** how much can I withdraw?") may still trigger the multi-agent orchestrator's `needs_decomposition()` heuristic, resulting in a multi-agent response. This is correct behavior for compound questions.

---

## 5.5 Confirmed Passing Tests (Post v1.2.1 Live Run)

**Integration test suite (47 cases, live servers):** 46 passed / 0 failed / 1 skipped

| Test ID | Description | Status |
|---|---|---|
| health_001 | `GET /health` → status: healthy | ✅ Confirmed live |
| metrics_001–008 | `GET /metrics` all fields + p50/p95 | ✅ Confirmed live |
| schema_001–014 | `QueryResponse` field types and pipeline steps | ✅ Confirmed live |
| routing_001 | "What is a mutual fund?" → investment_coach | ✅ Confirmed live |
| routing_002 | "What is SIP?" → investment_coach | ✅ Confirmed live |
| routing_003 | "How does SEBI regulate mutual funds?" → investment_coach | ✅ Fixed (was rag_agent) |
| routing_004 | "What are SEBI mutual fund regulations?" → investment_coach | ✅ Fixed (was rag_agent) |
| routing_005 | EPF + compound question → multi-agent (accepted) | ✅ |
| routing_006 | "How does TDS work on salary?" → tax_planner | ✅ Confirmed live |
| routing_007 | "What is my credit score?" → multi-agent (accepted) | ✅ |
| routing_008 | "What term life insurance?" → insurance_analyzer | ✅ Confirmed live |
| citation_001 | SEBI investment docs → skip (evidence gate, no docs) | ⏭ Skip |
| citation_002 | RBI credit guidelines → 5 citations | ✅ Confirmed live |
| citation_003 | General mutual fund → no citations | ✅ Confirmed live |
| context_001–004 | Two-turn conversation, no crashes, follow-up handled | ✅ Confirmed live |
| edge_001 | Whitespace-only query → 422 | ✅ |
| edge_002 | Missing `query` field → 422 | ✅ |
| edge_003 | profileHint forwarded → 200 | ✅ |
| agent_001 | Personal loans → credits_loans | ✅ |
| agent_002 | SEBI SIP regulations → investment_coach | ✅ (was failing) |
| agent_003 | Health insurance → insurance_analyzer | ✅ |
| agent_004 | Retirement savings → retirement_planner | ✅ |
| agent_005 | Tax exemptions | ⚠️ Needs tax docs in RAG |
| compliance_001 | Guaranteed returns → BLOCKED | ✅ |
| compliance_002 | Loan approval → BLOCKED | ✅ |
| compliance_004 | Valid query → ALLOWED | ✅ |
| Evidence/grounding tests | Require SEBI/IRDAI docs ingested | ⚠️ |

---

## 5.6 Remaining Work

### High Priority
1. **Ingest regulatory documents** — Unlock all evidence/grounding/citation tests
   - Source: SEBI mutual fund regulations, IRDAI guidelines, RBI master directions, CBDT circulars
   - Use: `python Finorbit_RAG/scripts/ingest_regulatory_docs.py`

2. **Fix tax_planner content generation** — Either create `taxDeductions.xml`/`taxExemptions.xml` or update tax_planner to provide general info without XML.

### Medium Priority
3. Full re-evaluation run to confirm actual pass rate (target: 80%+ / 13+/16)
4. PII detection test (compliance_003)
5. P95 latency validation with parallelization applied

### Low Priority
6. Improve citation extraction for complex multi-part queries
7. Additional evaluation test cases (target: 30+ for statistical significance)

---

## Conclusion

FinOrbit is a production-grade multi-agent financial assistant with:
- **Layered architecture**: Safety → Routing → Specialists → RAG
- **Async-first**: All validation, evidence retrieval, and sub-agent execution parallelized
- **Provider-agnostic LLM**: Switch between OpenAI and Gemini via env var
- **Persistent context**: PostgreSQL-backed conversation memory with graceful fallback
- **Observable**: End-to-end trace IDs, `/metrics` endpoint, structured JSON logs
- **Regulatory-grade guardrails**: PII detection, compliance blocking, grounding verification, suitability checks
