import types

from backend.core.multi_agent_orchestrator import MultiAgentOrchestrator


class CaptureAgent:
    def __init__(self):
        self.seen_queries = []

    def run(self, state):
        self.seen_queries.append(state.get("query"))
        return {"summary": "ok", "sources": [], "confidence": 1.0}


async def test_rag_agent_receives_unmodified_subquery(monkeypatch):
    rag = CaptureAgent()
    inv = CaptureAgent()

    orch = MultiAgentOrchestrator({"rag_agent": rag, "investment_coach": inv})

    # Force decomposition without calling OpenAI
    monkeypatch.setattr(orch.decomposer, "needs_decomposition", lambda q: True)
    monkeypatch.setattr(
        orch.decomposer,
        "decompose",
        lambda q: [
            {"sub_query": "Which investment gives guaranteed maximum payout?", "agent": "rag_agent", "depends_on": []},
            {"sub_query": "Summarize the answer", "agent": "investment_coach", "depends_on": [0]},
        ],
    )

    res = await orch.process_complex_query(
        query="Which investment gives guaranteed maximum payout?",
        profile={},
        session_id="s1",
    )

    assert res is not None
    assert rag.seen_queries == ["Which investment gives guaranteed maximum payout?"]
    # investment_coach can be augmented; we only care that rag query isn't mutated
