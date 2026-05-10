import json
import pytest
from unittest.mock import MagicMock, patch
from backend.core.compliance_engine import ComplianceEngineService

@pytest.fixture
def rules_file(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules = [
        {
            "id": 99,
            "regulator": "TEST",
            "module": "TEST",
            "pattern_type": "SEMANTIC",
            "pattern": "Hate speech or offensive language",
            "rule_type": "BLOCK",
            "message": "Offensive content detected.",
            "severity": "HIGH",
            "priority": 100,
            "language": "ALL",
            "channel": "ALL",
        }
    ]
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    return str(rules_path)


def _make_openai_mock(reply: str) -> MagicMock:
    """Build a mock OpenAI client whose chat.completions.create returns `reply`."""
    mock_choice = MagicMock()
    mock_choice.message.content = reply
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_semantic_compliance_violation_detected(rules_file):
    """Test that a semantic rule violation returns BLOCKED status."""
    mock_client = _make_openai_mock("YES")

    with patch("backend.core.compliance_engine.OpenAI", return_value=mock_client):
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            svc = ComplianceEngineService(rules_path=rules_file)

            res = svc.compliance_check("You are an idiot.", {"module": "TEST"})

            assert res.status == "BLOCKED"
            assert res.final_answer == "Offensive content detected."
            assert res.triggered_rule_ids == [99]

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args
            prompt_text = call_kwargs[1]["messages"][0]["content"]
            assert "Policy Violation Concept: \"Hate speech or offensive language\"" in prompt_text
            assert "You are an idiot." in prompt_text


def test_semantic_compliance_no_violation(rules_file):
    """Test that safe text passes the semantic check."""
    mock_client = _make_openai_mock("NO")

    with patch("backend.core.compliance_engine.OpenAI", return_value=mock_client):
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            svc = ComplianceEngineService(rules_path=rules_file)

            res = svc.compliance_check("Hello, how are you?", {"module": "TEST"})

            assert res.status == "OK"
            assert res.triggered_rule_ids == []


def test_semantic_compliance_api_failure_fails_open(rules_file):
    """Test that if the LLM API fails, we fail open (return OK)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    with patch("backend.core.compliance_engine.OpenAI", return_value=mock_client):
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            svc = ComplianceEngineService(rules_path=rules_file)

            res = svc.compliance_check("Some text", {"module": "TEST"})

            assert res.status == "OK"
