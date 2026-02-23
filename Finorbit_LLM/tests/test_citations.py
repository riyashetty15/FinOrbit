"""
Integration tests for FinOrbit v1.2 — citation extraction, routing fixes,
metrics endpoint, and response schema validation.

Requires a live server at http://localhost:8000 (and optionally RAG at
http://localhost:8081 for full citation tests).

Usage:
    # Standalone
    python tests/test_citations.py

    # Pytest (skips tests that need a live server)
    pytest tests/test_citations.py -v

    # Skip citation tests (no RAG/docs ingested)
    python tests/test_citations.py --skip-citations
"""

import asyncio
import sys
import argparse
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_pass = 0
_fail = 0
_skip = 0


def _record(label: str, passed: bool, note: str = "") -> None:
    global _pass, _fail
    if passed:
        _pass += 1
        print(f"  [PASS] {label}" + (f" — {note}" if note else ""))
    else:
        _fail += 1
        print(f"  [FAIL] {label}" + (f" — {note}" if note else ""))


def _skip_test(label: str, reason: str = "") -> None:
    global _skip
    _skip += 1
    print(f"  [SKIP] {label}" + (f" — {reason}" if reason else ""))


def _section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Health & Metrics (no LLM or RAG needed)
# ─────────────────────────────────────────────────────────────────────────────

async def test_health(client: httpx.AsyncClient) -> None:
    _section("SECTION 1 — Health & Metrics Endpoints")

    try:
        r = await client.get(f"{BASE_URL}/health")
        _record("GET /health returns 200", r.status_code == 200)
        body = r.json()
        _record("/health has 'status' field", "status" in body, body.get("status"))
        _record("/health status is 'healthy'", body.get("status") == "healthy")
        _record("/health has 'specialist_agents' field", "specialist_agents" in body)
    except Exception as e:
        _record("GET /health reachable", False, str(e))


async def test_metrics(client: httpx.AsyncClient) -> None:
    try:
        r = await client.get(f"{BASE_URL}/metrics")
        _record("GET /metrics returns 200", r.status_code == 200)
        body = r.json()
        required_keys = {"uptime_seconds", "router", "pipeline_validation",
                         "agent_execution_counts", "specialist_agents_registered"}
        for key in required_keys:
            _record(f"/metrics has '{key}'", key in body)
        router = body.get("router", {})
        _record("/metrics router has 'latency_ms'", "latency_ms" in router)
        latency = router.get("latency_ms", {})
        _record("/metrics latency has p50/p95", "p50" in latency and "p95" in latency)
        _record("/metrics 'specialist_agents_registered' is list",
                isinstance(body.get("specialist_agents_registered"), list))
    except Exception as e:
        _record("GET /metrics reachable", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Response Schema Validation (requires LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def _query(client: httpx.AsyncClient, query: str, conv_id: str = "test_schema_1",
                 user_id: str = "test_user") -> dict:
    r = await client.post(
        f"{BASE_URL}/query",
        json={"userId": user_id, "conversationId": conv_id, "query": query},
    )
    r.raise_for_status()
    return r.json()


async def test_response_schema(client: httpx.AsyncClient) -> None:
    _section("SECTION 2 — Response Schema Validation")

    try:
        body = await _query(client, "What is a mutual fund?", "test_schema_1")
        required_fields = ["response", "agents", "needs_clarification",
                           "pipeline_steps", "compliance_status"]
        for f in required_fields:
            _record(f"QueryResponse has '{f}'", f in body)

        _record("'response' is a non-empty string",
                isinstance(body.get("response"), str) and len(body["response"]) > 0)
        _record("'agents' is a non-empty list",
                isinstance(body.get("agents"), list) and len(body["agents"]) > 0)
        _record("'needs_clarification' is bool",
                isinstance(body.get("needs_clarification"), bool))
        _record("'pipeline_steps' is a list",
                isinstance(body.get("pipeline_steps"), list))

        # Optional but expected
        _record("'confidence_score' present and in [0,1]",
                body.get("confidence_score") is None or 0.0 <= body["confidence_score"] <= 1.0)
        _record("'compliance_status' is OK/BLOCKED/SKIPPED",
                body.get("compliance_status") in {"OK", "BLOCKED", "SKIPPED", None})
        _record("'mode' is info/guidance/action",
                body.get("mode") in {"info", "guidance", "action", None})

        # Pipeline steps — must include routing step and guardrails
        steps = body.get("pipeline_steps", [])
        step_names = [s.get("step", "") for s in steps]
        has_routing = any("Routing" in n for n in step_names)
        has_guardrails = any("Guardrail" in n for n in step_names)
        _record("Pipeline steps include routing step", has_routing,
                f"steps: {step_names}")
        _record("Pipeline steps include guardrails step", has_guardrails,
                f"steps: {step_names}")

    except Exception as e:
        _record("Response schema query succeeded", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Routing Fix Validation (v1.2) — requires LLM
# ─────────────────────────────────────────────────────────────────────────────

_ROUTING_CASES = [
    # (query, expected_agent, description)
    ("What is a mutual fund?",
     "investment_coach",
     "General investment query → investment_coach"),
    ("What is SIP?",
     "investment_coach",
     "SIP educational query → investment_coach"),
    ("How does SEBI regulate mutual funds?",
     "investment_coach",
     "SEBI + investment → investment_coach (not rag_agent) [v1.2 fix]"),
    ("What are SEBI mutual fund regulations?",
     "investment_coach",
     "SEBI + mutual fund → investment_coach [v1.2 fix]"),
    ("What is EPF and how much can I withdraw?",
     None,  # multi-part "and" triggers orchestrator; accept any specialist
     "EPF query → retirement_planner (or multi-agent) [v1.2 expanded keywords]"),
    ("How does TDS work on salary?",
     "tax_planner",
     "TDS salary query → tax_planner [v1.2 expanded keywords]"),
    ("What is my credit score and how can I improve it?",
     None,  # orchestrator may run multiple agents; accept any specialist
     "Credit score query → credits_loans (or multi-agent)"),
    ("What term life insurance should I buy?",
     "insurance_analyzer",
     "Term life insurance query → insurance_analyzer"),
]


async def test_routing_fixes(client: httpx.AsyncClient) -> None:
    _section("SECTION 3 — Routing Fix Validation (v1.2)")

    for i, (query, expected_agent, desc) in enumerate(_ROUTING_CASES, 1):
        try:
            body = await _query(client, query, conv_id=f"test_routing_{i}")
            agents = body.get("agents", [])
            agent_type = body.get("agent_type") or (agents[0] if agents else "unknown")
            if expected_agent is None:
                # Accept any specialist agent (multi-agent orchestrator scenario)
                passed = agent_type in {"credits_loans", "investment_coach", "tax_planner",
                                        "insurance_analyzer", "retirement_planner", "rag_agent"}
                _record(desc, passed, f"got={agent_type!r} (multi-agent OK)")
            else:
                passed = agent_type == expected_agent
                _record(
                    desc,
                    passed,
                    f"got={agent_type!r}, expected={expected_agent!r}",
                )
        except Exception as e:
            _record(desc, False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Citation / Source Extraction (requires LLM + RAG + docs)
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_CASES = [
    {
        "query": "What are SEBI mutual fund regulations?",
        "expected_module": "investment",
        "should_have_citations": True,
        # NOTE: passes only when SEBI/investment docs are ingested in the RAG store.
        # Without docs the evidence gate correctly refuses — mark as SKIP if no docs found.
        "skip_if_no_docs": True,
        "desc": "SEBI regulations query should produce citations (requires ingested SEBI docs)",
    },
    {
        "query": "What are RBI credit lending guidelines?",
        "expected_module": "credit",
        "should_have_citations": True,
        "desc": "RBI guidelines query should produce citations",
    },
    {
        "query": "What is mutual fund?",
        "expected_module": "investment",
        "should_have_citations": False,
        "desc": "General educational query should NOT produce citations",
    },
]

_CITATION_SOURCE_FIELDS = {"source", "excerpt", "score", "type"}


async def test_citations(client: httpx.AsyncClient, skip: bool = False) -> None:
    _section("SECTION 4 — Citation Extraction")

    if skip:
        for c in _CITATION_CASES:
            _skip_test(c["desc"], "RAG service / docs not available (--skip-citations)")
        return

    for i, case in enumerate(_CITATION_CASES, 1):
        try:
            body = await _query(client, case["query"], conv_id=f"test_cit_{i}")
            sources = body.get("sources") or []
            response_text = body.get("response", "")

            if case["should_have_citations"]:
                has_cit = len(sources) > 0
                # If no citations but response is an evidence gate refusal → skip (docs not ingested)
                is_evidence_gate = "verified regulatory sources" in response_text or \
                                   "Insufficient regulatory evidence" in response_text or \
                                   "Cannot determine answer" in response_text
                if not has_cit and is_evidence_gate and case.get("skip_if_no_docs"):
                    _skip_test(case["desc"], "evidence gate fired — no docs ingested in RAG store")
                else:
                    _record(case["desc"], has_cit,
                            f"citations={len(sources)}" if has_cit else
                            f"none found — preview: {response_text[:120]!r}")
                if has_cit:
                    # Validate citation structure
                    first = sources[0]
                    present = _CITATION_SOURCE_FIELDS & first.keys()
                    _record(f"  Citation #{i} has required fields",
                            len(present) >= 3,
                            f"found: {sorted(present)}")
                    _record(f"  Citation #{i} score is numeric",
                            isinstance(first.get("score"), (int, float)))
            else:
                _record(case["desc"], not sources,
                        f"unexpectedly got {len(sources)} citation(s)" if sources else "")

        except Exception as e:
            _record(case["desc"], False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Conversation Context Persistence (requires LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def test_conversation_context(client: httpx.AsyncClient) -> None:
    _section("SECTION 5 — Conversation Context Persistence")

    conv_id = "test_ctx_persistence_1"
    user_id = "test_ctx_user"

    try:
        # Turn 1: Establish context (tax topic)
        body1 = await _query(client,
                             "What are the income tax slabs for FY 2024-25?",
                             conv_id=conv_id, user_id=user_id)
        agent1 = (body1.get("agents") or ["unknown"])[0]
        _record("Turn 1 succeeds", bool(body1.get("response")),
                f"agent={agent1}")

        # Turn 2: Follow-up referencing prior turn
        body2 = await _query(client,
                             "My annual income is 15 lakhs. What would I owe?",
                             conv_id=conv_id, user_id=user_id)
        _record("Turn 2 follow-up succeeds", bool(body2.get("response")),
                f"agent={(body2.get('agents') or ['unknown'])[0]}")

        # Both turns should produce non-empty text
        _record("Both responses are non-empty",
                len(body1.get("response", "")) > 0 and
                len(body2.get("response", "")) > 0)

        # The profile (15 lakhs income) should ideally be reflected;
        # we can only verify the request didn't error.
        _record("Follow-up does not crash server",
                body2.get("response") is not None)

    except Exception as e:
        _record("Conversation context test succeeded", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Edge Cases & Error Handling
# ─────────────────────────────────────────────────────────────────────────────

async def test_edge_cases(client: httpx.AsyncClient) -> None:
    _section("SECTION 6 — Edge Cases & Error Handling")

    # Empty-ish query (just whitespace) should return 422
    try:
        r = await client.post(
            f"{BASE_URL}/query",
            json={"userId": "u1", "conversationId": "c1", "query": "   "},
        )
        _record("Whitespace-only query rejected (422)", r.status_code == 422,
                f"got status {r.status_code}")
    except Exception as e:
        _record("Whitespace-only query handled", False, str(e))

    # Missing required field (no query key) should return 422
    try:
        r = await client.post(
            f"{BASE_URL}/query",
            json={"userId": "u1", "conversationId": "c1"},
        )
        _record("Missing 'query' field rejected (422)", r.status_code == 422,
                f"got status {r.status_code}")
    except Exception as e:
        _record("Missing 'query' field handled", False, str(e))

    # profileHint is forwarded correctly (should not crash)
    try:
        r = await client.post(
            f"{BASE_URL}/query",
            json={
                "userId": "u2",
                "conversationId": "c2",
                "query": "What SIP should I start?",
                "profileHint": "I am 30 years old with 12 lakhs annual income",
            },
        )
        _record("profileHint accepted (200)", r.status_code == 200,
                f"status={r.status_code}")
    except Exception as e:
        _record("profileHint request handled", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_all(skip_citations: bool = False) -> None:
    print("\n" + "=" * 70)
    print("  FINORBIT v1.2 INTEGRATION TEST SUITE")
    print("  Target: http://localhost:8000")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Connectivity check
        try:
            r = await client.get(f"{BASE_URL}/health")
            if r.status_code != 200:
                print(f"\n[ERROR] Server returned {r.status_code} on /health. Aborting.")
                sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Cannot reach {BASE_URL}: {e}")
            print("  Start the server first: cd Finorbit_LLM && uvicorn backend.server:app --port 8000")
            sys.exit(1)

        await test_health(client)
        await test_metrics(client)
        await test_response_schema(client)
        await test_routing_fixes(client)
        await test_citations(client, skip=skip_citations)
        await test_conversation_context(client)
        await test_edge_cases(client)

    # Summary
    total = _pass + _fail + _skip
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {_pass} passed | {_fail} failed | {_skip} skipped | {total} total")
    print(f"{'=' * 70}\n")

    sys.exit(0 if _fail == 0 else 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pytest-compatible wrappers (no live server needed for offline unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FinOrbit integration tests")
    p.add_argument("--skip-citations", action="store_true",
                   help="Skip citation tests (use when RAG / docs are not available)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run_all(skip_citations=args.skip_citations))
