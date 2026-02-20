from backend.core.validation_models import PostValidationResult, ValidationCheck, Severity
from backend.server import _should_hallucination_fallback


def _pv(crit_types: list[str]) -> PostValidationResult:
    checks = [
        ValidationCheck(
            agent_name=t,
            check_type=t,
            passed=False,
            confidence=0.0,
            severity=Severity.CRITICAL,
            issues=[f"{t} failed"],
            recommendations=[],
            metadata={},
        )
        for t in crit_types
    ]
    return PostValidationResult(
        grounding_ok=False,
        grounding_score=0.0,
        reg_ok=False,
        reg_score=0.0,
        numeric_ok=False,
        numeric_score=0.0,
        suitability_ok=True,
        suitability_score=1.0,
        tone_ok=True,
        tone_score=1.0,
        validation_checks=checks,
        blocking_issues=["blocked"],
        warnings=[],
        should_block=True,
    )


def test_hallucination_fallback_true_for_grounding_only():
    assert _should_hallucination_fallback(_pv(["grounding"])) is True


def test_hallucination_fallback_true_for_numeric_only():
    assert _should_hallucination_fallback(_pv(["numeric"])) is True


def test_hallucination_fallback_true_for_grounding_and_numeric():
    assert _should_hallucination_fallback(_pv(["grounding", "numeric"])) is True


def test_hallucination_fallback_false_for_regulatory():
    assert _should_hallucination_fallback(_pv(["regulatory"])) is False


def test_hallucination_fallback_false_for_mixed_critical_types():
    assert _should_hallucination_fallback(_pv(["grounding", "regulatory"])) is False
