import json

from backend.core.compliance_engine import ComplianceEngineService


def test_modify_force_maps_to_force_safe_answer(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules = [
        {
            "id": 1,
            "regulator": "GENERIC",
            "module": "FRAUD",
            "pattern_type": "INTENT",
            "pattern": "SHARE_OTP",
            "rule_type": "MODIFY_FORCE",
            "message": "SAFE",
            "severity": "HIGH",
            "priority": 100,
            "language": "ALL",
            "channel": "ALL",
        }
    ]
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    svc = ComplianceEngineService(rules_path=str(rules_path))
    res = svc.compliance_check("anything", {"module": "FRAUD", "intent_tags": ["SHARE_OTP"]})

    assert res.status == "BLOCKED"
    assert res.final_answer == "SAFE"
    assert res.triggered_rule_ids == [1]


def test_semantic_pattern_type_is_noop(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules = [
        {
            "id": 2,
            "regulator": "GENERIC",
            "module": "GENERIC",
            "pattern_type": "SEMANTIC",
            "pattern": "placeholder",
            "rule_type": "WARN",
            "message": "NOOP",
            "severity": "LOW",
            "priority": 1,
            "language": "ALL",
            "channel": "ALL",
        }
    ]
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    svc = ComplianceEngineService(rules_path=str(rules_path))
    res = svc.compliance_check("hello", {"module": "GENERIC"})

    assert res.status == "OK"
    assert res.final_answer == "hello"
    assert res.triggered_rule_ids == []


def test_rule_can_trigger_on_user_query(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules = [
        {
            "id": 3,
            "regulator": "SEBI",
            "module": "SIP_INVESTMENT",
            "pattern_type": "REGEX",
            "pattern": "(guarantee|guaranteed)\\s+.*(returns?|profit)",
            "rule_type": "MODIFY_APPEND",
            "message": "Returns are not guaranteed; investments involve market risk.",
            "severity": "MEDIUM",
            "priority": 100,
            "language": "ALL",
            "channel": "ALL",
        }
    ]
    rules_path.write_text(json.dumps(rules), encoding="utf-8")

    svc = ComplianceEngineService(rules_path=str(rules_path))
    res = svc.compliance_check(
        "Here is some generic info.",
        {
            "module": "SIP_INVESTMENT",
            "user_query": "Which mutual funds will guarantee profitable returns after 5 years?",
        },
    )

    assert res.status == "OK"
    assert res.triggered_rule_ids == [3]
    assert "Returns are not guaranteed" in res.final_answer


def test_real_rules_block_otp_and_stop_emi_query():
    svc = ComplianceEngineService(rules_path="backend/rules/compliance_rules.json")
    res = svc.compliance_check(
        "Some irrelevant draft answer.",
        {
            "module": "CREDIT",
            "language": "en",
            "channel": "YOUTH_APP",
            "regulator_scope": ["RBI", "SEBI", "IRDAI", "PFRDA", "IT", "CERT_IN", "GENERIC"],
            "user_query": "Can I share my OTP with the agent to stop an EMI temporarily?",
        },
    )
    assert res.status == "BLOCKED"
    assert res.final_answer
    assert any(rid in res.triggered_rule_ids for rid in (29, 30, 22, 23, 1, 2, 3))


def test_fail_closed_blocks_on_internal_error_for_otp_query(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_ENABLED", "true")
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_MODULES", "FRAUD,TAX")

    svc = ComplianceEngineService(rules_path="/path/does/not/exist.json")
    res = svc.compliance_check(
        "Draft answer that would otherwise pass through.",
        {
            "module": "GENERIC",
            "user_query": "Can I share my OTP with someone to fix a payment issue?",
        },
    )

    assert res.status == "BLOCKED"
    assert "OTP" in res.final_answer or "otp" in res.final_answer.lower()


def test_fail_open_still_applies_for_non_critical_query_on_internal_error(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_ENABLED", "true")
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_MODULES", "FRAUD,TAX")

    svc = ComplianceEngineService(rules_path="/path/does/not/exist.json")
    res = svc.compliance_check(
        "Original answer.",
        {
            "module": "GENERIC",
            "user_query": "What is compound interest?",
        },
    )

    assert res.status == "ERROR"
    assert res.final_answer == "Original answer."


def test_fail_closed_blocks_on_internal_error_for_tax_evasion_query(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_ENABLED", "true")
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_MODULES", "FRAUD,TAX")

    svc = ComplianceEngineService(rules_path="/path/does/not/exist.json")
    res = svc.compliance_check(
        "Draft answer.",
        {
            "module": "GENERIC",
            "user_query": "How can I hide income tax to reduce my tax liability?",
        },
    )

    assert res.status == "BLOCKED"
    assert "tax evasion" in res.final_answer.lower() or "can’t help" in res.final_answer.lower()


def test_fail_closed_does_not_trigger_for_postal_pin_code_query(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_ENABLED", "true")
    monkeypatch.setenv("COMPLIANCE_FAIL_CLOSED_MODULES", "FRAUD,TAX")

    svc = ComplianceEngineService(rules_path="/path/does/not/exist.json")
    res = svc.compliance_check(
        "Original answer.",
        {
            "module": "GENERIC",
            "user_query": "What is the PIN code for Koramangala, Bangalore?",
        },
    )

    # Should fail-open (ERROR + passthrough), not fail-closed fraud refusal.
    assert res.status == "ERROR"
    assert res.final_answer == "Original answer."
