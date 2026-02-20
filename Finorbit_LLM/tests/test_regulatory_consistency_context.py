from backend.agents.validation.regulatory_consistency import RegulatoryConsistencyAgent
from backend.core.validation_models import Severity


def test_guaranteed_returns_is_critical_for_market_linked_investing():
    agent = RegulatoryConsistencyAgent()
    res = agent.validate(
        response="This mutual fund will give you guaranteed returns.",
        context={"query": "Which mutual funds guarantee returns?", "module": "SIP_INVESTMENT"},
    )
    assert res.severity == Severity.CRITICAL
    assert res.passed is False


def test_guaranteed_returns_is_warning_for_fixed_deposit_context():
    agent = RegulatoryConsistencyAgent()
    res = agent.validate(
        response="Fixed deposits offer guaranteed returns at a fixed rate.",
        context={"query": "What is the guaranteed payout for fixed deposits?", "module": "GENERIC"},
    )
    assert res.severity in (Severity.WARNING, Severity.INFO)
    assert res.passed is False
