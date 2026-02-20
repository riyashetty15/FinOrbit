from backend.core.guardrails import enforce_input_guardrail
from backend.models import GuardrailResult


def test_enforce_input_guardrail_allows_when_allowed_true():
    gr = GuardrailResult(allowed=True, blocked_category=None, blocked_reason=None)
    out = enforce_input_guardrail(gr)
    assert out.tripwire_triggered is False


def test_enforce_input_guardrail_blocks_when_reason_present():
    gr = GuardrailResult(allowed=False, blocked_category="illegal", blocked_reason="tax evasion")
    out = enforce_input_guardrail(gr)
    assert out.tripwire_triggered is True


def test_enforce_input_guardrail_fails_open_on_empty_block():
    gr = GuardrailResult(allowed=False, blocked_category=None, blocked_reason="")
    out = enforce_input_guardrail(gr)
    assert out.tripwire_triggered is False
