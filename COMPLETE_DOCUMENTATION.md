# FinOrbit Complete Documentation

**Last Updated:** February 17, 2026

This comprehensive guide covers the architecture, implementation, and deployment of the FinOrbit financial assistant system with production-grade RAG capabilities.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Production-Grade RAG Implementation](#2-production-grade-rag-implementation)
3. [Deployment Guide](#3-deployment-guide)

---

# 1. System Architecture Overview

This section provides a comprehensive overview of the FinOrbit financial assistant architecture, detailing the transition to a layered, service-oriented design.

## 1.1 High-Level Architecture

The system follows a **Layered Architecture** designed to separate orchestration, decision logic, and low-level tooling.

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
*   **Finance Orchestrator (`backend/finance_agent.py`)**: The conversational entry point. It manages session state and coordinates high-level user intent.
*   **Multi-Agent Orchestrator (`backend/core/multi_agent_orchestrator.py`)**: Routes queries to domain specialists or services based on router signals.
*   **Router (`backend/core/router.py`)**: Determines the nature of the query (e.g., "Credit" vs "Investment") using a hybrid approach (LLM semantic classification with keyword fallback).
*   **State Manager (`backend/core/conversation_context.py`)**: Maintains user context (Age, Income, Risk Profile) across the session.

### B. Service Layer
This layer replaces standalone "agents" with deterministic services.
*   **Retrieval Service (`backend/services/retrieval_service.py`)**: 
    *   Replaces the old `RAGAgent`.
    *   **Responsibility**: Converts natural language queries into verified evidence.
    *   **Workflow**: 
        1.  **Semantically Routes** query to the correct document module (e.g., "RBI/NBFC" -> "Credit").
        2.  **Retrieves** chunks from the RAG Server.
        3.  **Verifies** chunks using an LLM to ensure they actually answer the question.
        4.  **Returns** an `EvidencePack` (proven citations).
*   **Decision Engine (`backend/services/decision_engine.py`)**:
    *   Enforces a strict output schema for all financial advice.
    *   **Output Format**: `DecisionOutput` (Recommendations, Reasoning, Assumptions, Risks, Disclaimers).
    *   Prevents unstructured "chatty" responses.

### C. Specialist Workflows (Domain Logic)
Instead of free-form agents, these are structured workflows handling specific domains.
*   **Credits & Loans** (`backend/agents/specialist/credits_loans.py`): Analyzes debt profiles, calculates EMI eligibility.
*   **Investment Coach** (`backend/agents/specialist/investment_coach.py`): SIP planning, portfolio diversification logic.
*   **Tax Planner** (`backend/agents/specialist/tax_planner.py`): 2024/2025 tax regime calculations and deduction logic.
*   **Retirement Planner** (`backend/agents/specialist/retirement_planner.py`): Retirement corpus planning and withdrawal assumptions.
*   **Insurance Analyzer** (`backend/agents/specialist/insurance_analyzer.py`): Coverage gap analysis and suitability checks.
*   *Note: These specialists call the `RetrievalService` for regulatory data.*

### D. Guardrails Layer (`backend/guardrails/`)
A unified pipeline ensuring safety at three stages:
1.  **Pre-Guardrails**: Input filtering (PII checks, jailbreak detection).
2.  **Advice-Guardrails**: In-flight checks (Mis-selling prevention, Suitability checks against user profile).
3.  **Post-Guardrails**: Output verification (Hallucination checks, Tone & Clarity).

### E. Tooling Layer (MCP Architecture)
This layer abstracts capabilities into "servers" or "toolkits".

#### 1. RAG Server (`Finorbit_RAG/`)
*   **Role**: Dedicated Context Provider (MCP).
*   **Function**: Indexes PDFs/Docs into `PGVector`. Exposes an API to search specific modules (Credit, Tax, etc.).
*   **Key Feature**: "Module Siloing" - ensures queries about Loans don't get mixed with Mutual Fund documents.

#### 2. Finance Math Tools (`backend/tools/finance_math.py`)
*   **Role**: Computational Tool Provider.
*   **Function**: Performs deterministic calculations.
    *   `calculate_emi`: Loan amortization.
    *   `estimate_tax_new_regime_2024`: Indian tax slab estimates.
    *   `calculate_sip_returns`: Future value of investments.
*   **Benefit**: preventing the LLM from doing arithmetic (hallucination risk).

## 1.3 How the "MCP Server" Works
The Model Context Protocol (MCP) pattern is implemented via the **RAG Server**.

1.  **Separation**: The RAG logic lives in a completely separate process (`Finorbit_RAG/main.py`) from the LLM logic (`Finorbit_LLM`).
2.  **Communication**: 
    *   The `RetrievalService` calls `backend/tools/rag_tool.py`.
    *   `rag_tool.py` makes an HTTP request to the running RAG Server.
3.  **Flow**:
    *   User asks: *"What are the NBFC rules?"*
    *   **LLM Router**: Classifies as `module: credit`.
    *   **Retrieval Service**: Calls RAG Tool with `module=credit, query='NBFC rules'`.
    *   **RAG Server**: Vector searches the `credit_chunks` table in Postgres.
    *   **Retrieval Service**: Receives chunks -> Verifies relevance -> Returns Evidence.

## 1.4 Current Implementation Status

| Component | Status | Location | 
| :--- | :--- | :--- | 
| **Orchestrator** | Active | `backend/server.py` |
| **Multi-Agent Orchestrator** | Active | `backend/core/multi_agent_orchestrator.py` |
| **Retrieval Service** | Active | `backend/services/retrieval_service.py` |
| **Logic Routing** | Active | `backend/core/router.py` |
| **Math Tools** | Active | `backend/tools/finance_math.py` |
| **Decision Engine** | Active | `backend/services/decision_engine.py` |
| **Guardrails** | Active | `backend/guardrails/pipeline.py` |
| **Domain Specialists** | Active (Pending Workflow Migration) | `backend/agents/specialist/` |
| **Workflow Layer** | Planned | `backend/workflows/` |

---

# 2. Production-Grade RAG Implementation

This section describes the production-grade RAG system implementation with evidence contracts, routing signals, metadata filters, coverage scoring, and refusal logic.

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
- `partial`: 1-2 verified citations OR weak relevance
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

**Evidence Need Detection:**
- Triggers on: RBI, SEBI, IRDAI, regulations, compliance keywords
- Regulatory questions: rules, circulars, master directions
- Factual lookups: rates, limits, ratios
- Authority statements: "according to", "as per", "under section"

**Methods:**
- `route_with_evidence_intent(query, user_context)` → RouteIntent
- `_detect_evidence_need(query)` → bool
- `_detect_time_sensitivity(query)` → "high" | "low"
- `_detect_query_type(query)` → "info" | "advice" | "compliance"

### 2.1.3 RAG API Filters (`Finorbit_RAG/main.py` + `retrieval_pipeline.py`)

**New Filter Parameters:**
- `year_min`: Minimum year (inclusive, uses GTE operator)
- `jurisdiction`: Geographic filter (e.g., "IN", "US", "UK")
- `effective_date`: Document effective date (ISO format)
- `version`: Document version string

**Existing Filters:**
- `doc_type`, `year`, `issuer`, `regulator_tag`
- `security`, `is_current`, `pii`
- `compliance_tags_any`: List[str] with IN operator

### 2.1.4 DecisionEngine Evidence Gating (`backend/services/decision_engine.py`)

**Key Methods:**

```python
DecisionEngine.check_evidence_gate(
    needs_evidence: bool,
    evidence_pack: Optional[EvidencePack],
    query: str,
    module: str
) → Optional[DecisionOutput]
```

**Gating Logic:**
1. If `needs_evidence=False` → pass gate (return None)
2. If `needs_evidence=True` and no evidence_pack → refuse
3. If `needs_evidence=True` and coverage != "sufficient" → refuse with follow-ups

**Refusal Output:**
- Contextual follow-up questions (2-4 questions based on module)
- Clear reasoning about insufficient evidence
- Risk warnings about compliance accuracy
- Module-specific clarification requests

**Example Follow-ups:**
- Credit: "Are you asking about NBFC, banks, or specific lending institutions?"
- Taxation: "Which assessment year or financial year does your query relate to?"
- Insurance: "Which type of insurance (life, health, vehicle, property)?"

### 2.1.5 Post-Guardrail Citation Validation (`backend/agents/validation/grounding_check.py`)

**Enhanced GroundingCheckAgent:**

```python
def _validate_with_evidence_pack(response, evidence_pack, context) → ValidationCheck
```

**Checks:**
1. **Regulatory Claim Extraction:** Detects sentences with RBI/SEBI/IRDAI + requires/mandates/specifies
2. **Citation Requirement:** Regulatory claims must have backing citations
3. **Citation Mapping:** Verifies claims map to actual doc_ids from EvidencePack
4. **Coverage Validation:** Ensures coverage is "sufficient" for regulatory queries

**Failure Modes:**
- CRITICAL: Regulatory claims with no citations
- WARNING: Coverage insufficient for regulatory query
- WARNING: Unbacked regulatory claims detected

**Regulatory Claim Patterns:**
- "RBI requires/mandates..."
- "SEBI guidelines state..."
- "According to IRDAI regulations..."
- "Under Section X..."
- "The regulation mandates..."

## 2.2 Integration Steps for server.py

### Step A: Import New Components (Top of File)

```python
from backend.core.router import RouterAgent, RouteIntent
from backend.services.retrieval_service import RetrievalService, EvidencePack
from backend.services.decision_engine import DecisionEngine
```

### Step B: After Routing, Get RouteIntent

```python
# STEP 3A: Get base routing decision
agent_type, intent = router_agent.route_with_context(
    query=user_input,
    conversation_context=conversation_context
)

# STEP 3B: Get enriched routing intent with evidence requirements
route_intent = router_agent.route_with_evidence_intent(
    query=user_input,
    user_context=profile
)

log_event(logger, "route_intent_enriched", "router", {
    "module": route_intent.module,
    "needs_evidence": route_intent.needs_evidence,
    "jurisdiction": route_intent.jurisdiction,
    "time_sensitivity": route_intent.time_sensitivity,
    "query_type": route_intent.query_type
})
```

### Step C: Retrieve Evidence if Needed

```python
# STEP 4A: Evidence Retrieval (if required)
evidence_pack = None
if route_intent.needs_evidence:
    log_event(logger, "evidence_retrieval_start", "rag")
    
    # Build filters from route intent
    filters = {
        "jurisdiction": route_intent.jurisdiction,
    }
    
    # Add time sensitivity filters
    if route_intent.time_sensitivity == "high":
        filters["is_current"] = True
        filters["year_min"] = 2023  # Last 3 years
    
    # Retrieve evidence
    evidence_pack = await retrieval_service.retrieve_evidence(
        query=user_input,
        module=route_intent.module,
        top_k=8,
        filters=filters
    )
    
    log_event(logger, "evidence_retrieval_complete", "rag", {
        "coverage": evidence_pack.coverage,
        "citations_count": len(evidence_pack.citations),
        "confidence": evidence_pack.confidence
    })
```

### Step D: Evidence Gate Check

```python
# STEP 4B: Evidence Gate - Refuse if insufficient evidence for regulatory query
evidence_refusal = DecisionEngine.check_evidence_gate(
    needs_evidence=route_intent.needs_evidence,
    evidence_pack=evidence_pack,
    query=user_input,
    module=route_intent.module
)

if evidence_refusal:
    log_event(logger, "evidence_gate_failed", "decision_engine", {
        "coverage": evidence_pack.coverage if evidence_pack else "none",
        "follow_ups_count": len(evidence_refusal.follow_ups)
    })
    
    # Convert DecisionOutput to response format
    response_text = evidence_refusal.recommendations[0] if evidence_refusal.recommendations else "Insufficient evidence to answer query."
    
    # Add follow-up questions to response
    if evidence_refusal.follow_ups:
        response_text += "\n\n**To help me provide accurate information, please clarify:**\n"
        for i, follow_up in enumerate(evidence_refusal.follow_ups, 1):
            response_text += f"{i}. {follow_up}\n"
    
    # Skip agent execution, return refusal
    # ... (set appropriate metadata and return)
```

### Step E: Pass EvidencePack to Agent

```python
# STEP 4C: Execute Agent with Evidence
initial_state = {
    "query": full_context,
    "profile": profile,
    "intent": intent,
    "evidence_pack": evidence_pack  # NEW: Pass evidence to agent
}

result = await agent.run(initial_state)
```

### Step F: Pass EvidencePack to Grounding Check

```python
# STEP 6: Post-Validation
validation_context = {
    "query": user_input,
    "profile": profile,
    "retrieved_passages": retrieved_passages,  # Legacy
    "evidence_pack": evidence_pack,            # NEW: Production-grade
    "agent_type": agent_type
}

post_validation_result = validation_pipeline.validate_response(
    response=response_text,
    context=validation_context
)
```

### Step G: Include Citations in Response

```python
# STEP 9: Build Response
citations_list = []
if evidence_pack and evidence_pack.citations:
    for citation in evidence_pack.citations:
        citations_list.append({
            "doc_id": citation.doc_id,
            "source": citation.source,
            "page": citation.page,
            "chunk_id": citation.chunk_id,
            "text": citation.text[:200],  # Truncate for response size
            "score": citation.score,
            "metadata": citation.metadata
        })

return QueryResponse(
    response=final_response_text,
    confidence=confidence,
    agent_type=agent_type,
    intent=intent,
    citations=citations_list,  # NEW: Production-grade citations
    # ... other fields
)
```

## 2.3 Testing Checklist

### Evidence Retrieval Tests
- [ ] Returns correct module docs only
- [ ] Returns correct metadata and page refs
- [ ] Filters work (year_min, jurisdiction, issuer, doc_type)
- [ ] Coverage scoring works: sufficient, partial, insufficient
- [ ] Citation extraction includes doc_id, page, chunk_id, source

### Grounding Tests
- [ ] No regulatory claims without citations
- [ ] Refusal occurs when coverage insufficient
- [ ] Follow-up questions are contextual to module
- [ ] Citation-to-claim mapping validation works

### Safety Tests
- [ ] PII blocked
- [ ] Mis-selling prevented without user profile
- [ ] Compliance claims gated by evidence

### Integration Tests
- [ ] RouteIntent correctly detects needs_evidence
- [ ] Evidence gate refusal works end-to-end
- [ ] EvidencePack flows through agent → validation → response
- [ ] Citations appear in final response JSON

## 2.4 Example Queries

### Should Trigger Evidence Need + Sufficient Coverage
```
"What are the NBFC NPA classification rules according to RBI?"
```
Expected:
- `needs_evidence=True`
- `time_sensitivity="high"` (rules change)
- `module="credit"`
- Coverage: sufficient (if docs available)
- Response includes citations from RBI master direction

### Should Trigger Evidence Need + Insufficient Coverage
```
"What are the latest SEBI regulations for cryptocurrency trading?"
```
Expected:
- `needs_evidence=True`
- `time_sensitivity="high"` (latest)
- `module="investment"`
- Coverage: insufficient (likely no docs)
- Refusal with follow-ups: "Which specific SEBI circular are you referring to?"

### Should NOT Trigger Evidence Need
```
"Should I invest in mutual funds or stocks for retirement?"
```
Expected:
- `needs_evidence=False`
- `query_type="advice"`
- Standard agent execution without RAG gating

## 2.5 Performance Considerations

1. **Caching:** Route intent detection uses regex (fast), no LLM needed
2. **Parallel Retrieval:** Evidence retrieval can run in parallel with other pre-processing
3. **Early Refusal:** Evidence gate happens BEFORE expensive agent execution
4. **Verification:** LLM-based chunk verification is optional (falls back to heuristics)

## 2.6 Monitoring & Observability

**Key Events to Log:**
- `route_intent_enriched`: Log needs_evidence, jurisdiction, time_sensitivity
- `evidence_retrieval_complete`: Log coverage, citations_count, confidence
- `evidence_gate_failed`: Log refusal reason and follow-ups sent
- `grounding_validation`: Log regulatory_claims_count, unbacked_claims_count

**Metrics to Track:**
- Evidence gate refusal rate (by module)
- Average coverage score (by module, time_sensitivity)
- Citation count per query
- Grounding check pass rate

## 2.7 Files Modified

1. `/backend/services/retrieval_service.py` - EvidencePack schema, coverage scoring
2. `/backend/core/router.py` - RouteIntent schema, evidence detection
3. `/Finorbit_RAG/main.py` - Filter parameters in /query endpoint
4. `/Finorbit_RAG/retrieval/retrieval_pipeline.py` - Metadata filters implementation
5. `/backend/services/decision_engine.py` - Evidence gating, refusal logic
6. `/backend/agents/validation/grounding_check.py` - Citation validation
7. `/backend/server.py` - Full integration (evidence retrieval, gating, context passing)

---

# 3. Deployment Guide

## 3.1 Quick Start

### Prerequisites

1. **Python 3.11+** installed
2. **PostgreSQL with pgvector** extension (or use Docker)
3. **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))

## 3.2 Step 1: Database Setup (PostgreSQL + pgvector)

### Option A: Use Docker (Recommended)

```bash
cd Finorbit_RAG
docker-compose up -d postgres
```

This starts PostgreSQL with pgvector on `localhost:5432`.

### Option B: Local PostgreSQL

If you have PostgreSQL installed:

```bash
# Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Create database
CREATE DATABASE financial_rag;
```

## 3.3 Step 2: Configure Environment Variables

### RAG Server Configuration

```bash
cd Finorbit_RAG
cp .env.example .env
```

Edit `.env` and set:

```dotenv
# Database
DB_HOST=localhost
DB_NAME=financial_rag
DB_USER=postgres
DB_PASSWORD=your_actual_password
DB_PORT=5432

# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key_here

# Server Port
API_PORT=8081
```

### LLM Backend Configuration

```bash
cd ../Finorbit_LLM
cp .env.example .env
```

Edit `.env` and set:

```dotenv
# Google Gemini API
LLM_API_KEY=your_gemini_api_key_here
CUSTOM_MODEL_NAME=gemini-1.5-flash

# Database (same as RAG server)
DATABASE_URL=postgresql://postgres:your_actual_password@localhost:5432/financial_rag
```

## 3.4 Step 3: Install Dependencies

### RAG Server Dependencies

```bash
cd Finorbit_RAG
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### LLM Backend Dependencies

```bash
cd ../Finorbit_LLM
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## 3.5 Step 4: Initialize Database Schema

Run this **once** to create the database tables:

```bash
cd Finorbit_RAG
source .venv/bin/activate
python3 -c "
from stores import get_vector_store_for_module
from config import MODULES
for module in MODULES:
    print(f'Initializing {module}...')
    get_vector_store_for_module(module)
print('Database initialized successfully!')
"
```

## 3.6 Step 5: Start the Servers

### Terminal 1: Start RAG Server (Port 8081)

```bash
cd Finorbit_RAG
source .venv/bin/activate
python3 main.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8081
```

### Terminal 2: Start LLM Backend (Port 8000)

```bash
cd Finorbit_LLM
source .venv/bin/activate
python3 -m backend.server
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 3.7 Step 6: Verify the Servers

### Check RAG Server Health

```bash
curl http://localhost:8081/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Financial RAG API",
  "version": "1.0.0"
}
```

### Check LLM Backend Health

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "pipeline": "initialized",
  "specialist_agents": 5
}
```

## 3.8 Step 7: Access the UI

Open your browser and navigate to:

**http://localhost:8000/ui**

You should see the FinOrbit Assistant chat interface.

## 3.9 Step 8: Ingest Documents (Optional)

To enable RAG retrieval, upload regulatory PDFs:

```bash
curl -X POST "http://localhost:8081/ingest" \
  -F "file=@/path/to/your/document.pdf" \
  -F "module=credit" \
  -F "doc_type=regulation" \
  -F "year=2024" \
  -F "issuer=RBI"
```

**Supported Modules:**
- `credit` - Loans, EMI, CIBIL
- `investment` - Mutual funds, stocks, SIPs
- `insurance` - Health, life, term insurance
- `retirement` - NPS, pension, retirement planning
- `taxation` - Income tax, deductions

## 3.10 Architecture Flow

```
┌─────────────┐
│   Browser   │
│  localhost: │──┐
│   8000/ui   │  │
└─────────────┘  │
                 │
                 ▼
        ┌────────────────────┐
        │  LLM Backend       │  Port 8000
        │  (Finorbit_LLM)    │
        │  ┌──────────────┐  │
        │  │ Orchestrator │  │
        │  │   Router     │  │
        │  │  Specialists │  │
        │  └──────┬───────┘  │
        └─────────┼──────────┘
                  │
                  │ HTTP /retrieve
                  ▼
        ┌────────────────────┐
        │  RAG Server        │  Port 8081
        │  (Finorbit_RAG)    │
        │  ┌──────────────┐  │
        │  │ LlamaIndex   │  │
        │  │ Retrieval    │  │
        │  └──────┬───────┘  │
        └─────────┼──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │  PostgreSQL +      │  Port 5432
        │  pgvector          │
        └────────────────────┘
```

## 3.11 Common Issues

### 1. "Connection refused" errors

**Cause:** Servers not running or wrong ports.

**Fix:**
```bash
# Check if processes are listening
lsof -i :8000  # LLM Backend
lsof -i :8081  # RAG Server
lsof -i :5432  # PostgreSQL
```

### 2. "Module 'psycopg2' not found"

**Fix:**
```bash
pip install psycopg2-binary
```

### 3. "API key not configured"

**Fix:** Ensure `.env` files have valid `GOOGLE_API_KEY` or `LLM_API_KEY`.

### 4. Exit Code 137 (Out of Memory)

**Cause:** Insufficient RAM for embedding models.

**Fix:**
- Close unused applications
- Reduce `LLAMAINDEX_CHUNK_SIZE` in RAG `.env`
- Use smaller embedding models

## 3.12 Production Deployment

For production, use:

1. **Reverse Proxy (nginx)** to handle SSL/TLS
2. **Process Manager (systemd or PM2)** to auto-restart servers
3. **Environment Variables** via secrets management (AWS Secrets Manager, etc.)
4. **Docker Compose** for orchestration:

```bash
cd Finorbit_RAG
docker-compose up -d
```

## 3.13 Agent Invocation Flow

When a user query is received:

1. **Query hits `/query` endpoint** (LLM Backend)
2. **Router Agent** classifies intent (credit, investment, insurance, etc.)
3. **Orchestrator** selects specialist agent based on route
4. **Specialist Agent** (e.g., `CreditsLoanAgent`) executes:
   - Calls `RetrievalService` → which calls RAG Server → returns evidence
   - Processes user profile (age, income, CIBIL)
   - Runs LangGraph workflow (search or analysis mode)
5. **Validation Pipeline** checks output (hallucination, compliance)
6. **Response** returned to UI

**All agents are in-memory** and invoked on-demand. No separate agent processes needed.

## 3.14 Stopping the Servers

Press `Ctrl+C` in each terminal to stop the servers gracefully.

To stop Docker services:

```bash
cd Finorbit_RAG
docker-compose down
```

## 3.15 Next Steps

- **Upload Documents**: Use `/ingest` endpoint to add regulatory PDFs
- **Test Agents**: Try queries like:
  - "Show me home loans under 10% interest"
  - "What are tax deductions for salaried employees?"
  - "I need health insurance with 5 lakh coverage"
- **Monitor Logs**: Check terminal output for trace IDs and debug info
- **Customize Specialists**: Edit `backend/agents/specialist/*.py` to adjust logic

---

## Conclusion

This implementation provides:
- Layered architecture with clear separation of concerns
- Production-grade RAG with evidence contracts and coverage scoring
- Intelligent routing with conversation context preservation
- Multi-agent orchestration with specialist workflows
- Comprehensive guardrails pipeline (pre/in-flight/post validation)
- Rich metadata filters and citation validation
- Evidence gating with contextual refusal logic
- Scalable deployment with Docker support

**For issues, check server logs in terminals for detailed errors.**

**Last Updated:** February 17, 2026
