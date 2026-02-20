# FinOrbit

**A Production-Grade Multi-Agent Financial Assistant with RAG-Powered Regulatory Intelligence**

FinOrbit is an intelligent financial advisory system that combines specialized AI agents with a production-grade Retrieval-Augmented Generation (RAG) system to deliver accurate, compliant, and contextual financial guidance across credit, investment, insurance, taxation, and retirement planning.

---

## Key Features

### Multi-Agent Orchestration
- **5 Specialized Agents**: Credit & Loans, Investment Coach, Tax Planner, Retirement Planner, Insurance Analyzer
- **Intelligent Routing**: Semantic query classification with conversation context preservation
- **Parallel Processing**: Multi-agent orchestration for complex financial queries

### Production-Grade RAG System
- **Evidence Contracts**: Citation-backed responses with doc_id, source, page references
- **Coverage Scoring**: Automatic assessment of evidence sufficiency (sufficient/partial/insufficient)
- **Metadata Filtering**: Advanced filters by year, jurisdiction, document type, issuer, effective date
- **Evidence Gating**: Automatic refusal with contextual follow-ups when evidence is insufficient
- **Module Siloing**: Separate vector stores for credit, investment, insurance, taxation, retirement

### Comprehensive Guardrails
- **Pre-Guardrails**: Input validation, PII detection, jailbreak prevention
- **In-Flight Guardrails**: Mis-selling prevention, suitability checks
- **Post-Guardrails**: Hallucination detection, citation validation, tone & clarity checks

### Domain-Specific Tools
- **Finance Math Tools**: EMI calculation, SIP returns, tax estimation (Indian tax regime 2024/2025)
- **Regulatory Compliance**: Automatic grounding of RBI, SEBI, IRDAI regulatory claims
- **Decision Engine**: Structured output format (Recommendations, Reasoning, Risks, Disclaimers)

---

## Architecture

```
┌─────────────┐
│   Browser   │
│  localhost: │──────────────────┐
│   8000/ui   │                  │
└─────────────┘                  │
                                 ▼
                    ┌────────────────────────┐
                    │  LLM Backend           │  Port 8000
                    │  (Finorbit_LLM)        │
                    │  ┌──────────────────┐  │
                    │  │  Orchestrator    │  │
                    │  │  Router          │  │
                    │  │  5 Specialists   │  │
                    │  │  Guardrails      │  │
                    │  └────────┬─────────┘  │
                    └───────────┼────────────┘
                                │ HTTP /retrieve
                                ▼
                    ┌────────────────────────┐
                    │  RAG Server (MCP)      │  Port 8081
                    │  (Finorbit_RAG)        │
                    │  ┌──────────────────┐  │
                    │  │  LlamaIndex      │  │
                    │  │  Evidence Packs  │  │
                    │  │  5 Module Stores │  │
                    │  └────────┬─────────┘  │
                    └───────────┼────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  PostgreSQL + pgvector │  Port 5432
                    │  Vector Embeddings     │
                    └────────────────────────┘
```

### Layered Architecture

1. **Orchestration Layer**: Multi-agent coordinator, semantic router, conversation context manager
2. **Service Layer**: Retrieval service (RAG), decision engine, validation pipeline
3. **Specialist Layer**: Domain-specific workflows (credit, investment, tax, insurance, retirement)
4. **Tooling Layer (MCP)**: RAG server, finance math tools, regulatory compliance tools
5. **Infrastructure Layer**: PostgreSQL with pgvector, LlamaIndex, embedding models

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** with `pgvector` extension (or use Docker)
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/FinOrbit.git
cd FinOrbit
```

### 2. Database Setup

**Option A: Docker (Recommended)**

```bash
cd Finorbit_RAG
docker-compose up -d postgres
```

**Option B: Local PostgreSQL**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE DATABASE financial_rag;
```

### 3. Configure Environment Variables

**RAG Server (`Finorbit_RAG/.env`)**

```bash
cd Finorbit_RAG
cp .env.example .env
```

Edit `.env`:

```dotenv
DB_HOST=localhost
DB_NAME=financial_rag
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432

GOOGLE_API_KEY=your_gemini_api_key_here
API_PORT=8081
```

**LLM Backend (`Finorbit_LLM/.env`)**

```bash
cd ../Finorbit_LLM
cp .env.example .env
```

Edit `.env`:

```dotenv
LLM_API_KEY=your_gemini_api_key_here
CUSTOM_MODEL_NAME=gemini-1.5-flash
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/financial_rag
```

### 4. Install Dependencies

**RAG Server**

```bash
cd Finorbit_RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**LLM Backend**

```bash
cd ../Finorbit_LLM
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 5. Initialize Database

```bash
cd Finorbit_RAG
source .venv/bin/activate
python3 -c "
from stores import get_vector_store_for_module
from config import MODULES
for module in MODULES:
    print(f'Initializing {module}...')
    get_vector_store_for_module(module)
print('Database initialized!')
"
```

### 6. Start the Servers

**Terminal 1: RAG Server**

```bash
cd Finorbit_RAG
source .venv/bin/activate
python3 main.py
```

**Terminal 2: LLM Backend**

```bash
cd Finorbit_LLM
source .venv/bin/activate
python3 -m backend.server
```

### 7. Access the UI

Open your browser: **http://localhost:8000/ui**

---

## Usage Examples

### Health Check

```bash
# Check RAG Server
curl http://localhost:8081/health

# Check LLM Backend
curl http://localhost:8000/health
```

### Query the Assistant

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the NBFC NPA classification rules according to RBI?",
    "conversationId": "user123"
  }'
```

### Ingest Regulatory Documents

```bash
curl -X POST "http://localhost:8081/ingest" \
  -F "file=@rbi_nbfc_guidelines.pdf" \
  -F "module=credit" \
  -F "doc_type=regulation" \
  -F "year=2024" \
  -F "issuer=RBI"
```

**Supported Modules:**
- `credit` - Loans, EMI, CIBIL, NBFC regulations
- `investment` - Mutual funds, stocks, SIPs, SEBI guidelines
- `insurance` - Life, health, vehicle insurance (IRDAI)
- `retirement` - NPS, pension planning
- `taxation` - Income tax, deductions (IT Act)

---

## Testing

Run the full test suite:

```bash
cd Finorbit_LLM
source .venv/bin/activate
pytest tests/ -v
```

**Test Coverage:**
- Evidence retrieval and coverage scoring
- Routing with conversation context
- Guardrails enforcement (PII, hallucination, compliance)
- Grounding validation (regulatory claims require citations)
- Multi-agent orchestration

---

## Project Structure

```
FinOrbit/
├── Finorbit_LLM/              # LLM Backend (Port 8000)
│   ├── backend/
│   │   ├── agents/           # Specialist agents
│   │   │   ├── specialist/   # Domain experts
│   │   │   │   ├── credits_loans.py
│   │   │   │   ├── investment_coach.py
│   │   │   │   ├── tax_planner.py
│   │   │   │   ├── insurance_analyzer.py
│   │   │   │   └── retirement_planner.py
│   │   │   └── validation/   # Post-guardrails
│   │   ├── core/             # Orchestration & routing
│   │   │   ├── router.py     # RouteIntent with evidence detection
│   │   │   ├── multi_agent_orchestrator.py
│   │   │   └── conversation_context.py
│   │   ├── services/         # Business logic
│   │   │   ├── retrieval_service.py  # RAG with EvidencePack
│   │   │   └── decision_engine.py    # Evidence gating
│   │   ├── tools/            # Finance math, RAG client
│   │   └── server.py         # FastAPI app
│   └── tests/                # Test suite
│
├── Finorbit_RAG/              # RAG Server (MCP) (Port 8081)
│   ├── core/                 # LlamaIndex setup
│   ├── ingestion/            # PDF processing pipeline
│   ├── retrieval/            # Retrieval with metadata filters
│   ├── stores/               # Vector store (pgvector)
│   ├── mcp_server/           # MCP protocol handlers
│   └── main.py               # FastAPI RAG server
│
├── docs/                     # Documentation
└── README.md                 # This file
```

---

## Production-Grade RAG Features

### Evidence Contracts

Every RAG response includes:

```python
{
  "citations": [
    {
      "doc_id": "rbi_nbfc_2024_001",
      "source": "RBI_NBFC_Master_Direction_2024.pdf",
      "page": 12,
      "chunk_id": "chunk_45",
      "text": "NBFCs shall classify NPA as per RBI guidelines...",
      "score": 0.89,
      "metadata": {
        "year": 2024,
        "issuer": "RBI",
        "doc_type": "regulation",
        "jurisdiction": "IN"
      }
    }
  ],
  "coverage": "sufficient",  // or "partial" or "insufficient"
  "confidence": 0.87
}
```

### Evidence Gating

The system **refuses** to answer regulatory queries when evidence is insufficient:

**Query:** "What are the latest SEBI regulations for cryptocurrency trading?"

**Response (No Docs Available):**
```
I cannot provide accurate information about SEBI cryptocurrency regulations 
without verified regulatory documents.

To help me provide accurate information, please clarify:
1. Which specific SEBI circular or notification are you referring to?
2. Are you asking about trading rules, disclosure requirements, or investor protection?
3. What timeframe are you interested in (specific year or latest)?
```

### Metadata Filtering

Advanced filters ensure precision:

```python
# Example: Get latest RBI regulations from 2023-2025
filters = {
    "year_min": 2023,        # GTE operator
    "issuer": "RBI",
    "jurisdiction": "IN",
    "is_current": True,
    "doc_type": "regulation"
}
```

---

## Safety & Compliance

### Guardrails Pipeline

1. **Pre-Guardrails**
   - PII masking (Aadhaar, PAN, phone numbers)
   - Jailbreak attempt detection
   - Input sanitization

2. **In-Flight Guardrails**
   - Mis-selling prevention (requires user profile for advice)
   - Suitability checks (age, income, risk profile)
   - Financial product compliance

3. **Post-Guardrails**
   - Hallucination detection (claim-to-citation mapping)
   - Regulatory claim validation (RBI/SEBI/IRDAI statements require proof)
   - Tone & clarity checks

### Grounding Validation

Every regulatory claim is validated:

```python
# Valid: Backed by citation
"RBI requires NBFCs to classify NPAs after 90 days of default 
(Source: RBI_NBFC_Master_Direction_2024.pdf, Page 12)"

# Invalid: No citation provided
"RBI requires NBFCs to classify NPAs after 90 days of default"
# → CRITICAL validation failure
```

---

## Advanced Features

### Conversation Context Preservation

The system maintains conversation state across turns:

```
User: "What are NBFC NPA rules?"
Bot: [Evidence gate refuses - no docs]

User: "I'm looking into RBI regulations for 2025"
Bot: [Stays in RBI/credit context, doesn't route to unrelated agent]
```

### Multi-Agent Orchestration

Complex queries trigger multiple specialists:

**Query:** "I'm 28, earning ₹8L/year. Plan my taxes, investments, and insurance."

**Orchestrator Response:**
- Tax Planner: Old vs New regime comparison
- Investment Coach: SIP recommendations with risk profile
- Insurance Analyzer: Term + health coverage gap analysis

### Deterministic Finance Math

Calculations never rely on LLM estimation:

```python
# EMI calculation (deterministic)
calculate_emi(principal=500000, rate=8.5, tenure=120)
# → ₹6,158/month

# Tax estimation (IT Act 2024)
estimate_tax_new_regime_2024(income=800000)
# → ₹25,000 (after standard deduction)
```

---

## Common Issues

### Port Already in Use

```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9
lsof -ti:8081 | xargs kill -9
```

### Module Import Errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Database Connection Refused

```bash
# Check PostgreSQL status
lsof -i :5432

# Restart Docker container
cd Finorbit_RAG
docker-compose restart postgres
```

---

## Monitoring & Observability

Key events logged:

```json
{
  "event": "evidence_gate_failed",
  "coverage": "insufficient",
  "module": "credit",
  "follow_ups_count": 3
}

{
  "event": "grounding_validation",
  "regulatory_claims_count": 2,
  "unbacked_claims_count": 0,
  "result": "PASS"
}

{
  "event": "orchestrator_skipped",
  "reason": "follow_up_query",
  "last_agent": "rag_agent"
}
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=backend tests/

# Format code
black backend/ tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **LlamaIndex** - RAG framework
- **FastAPI** - API framework
- **PostgreSQL + pgvector** - Vector database
- **Google Gemini** - LLM provider
- **LangGraph** - Agent workflow orchestration

---

## Contact

For questions, issues, or collaboration:

- **Issues**: [GitHub Issues](https://github.com/yourusername/FinOrbit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FinOrbit/discussions)

---

**Built for production-grade financial intelligence**

*Last Updated: February 17, 2026*
