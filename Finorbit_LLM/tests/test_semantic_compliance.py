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

def test_semantic_compliance_violation_detected(rules_file):
    """Test that a semantic rule violation returns BLOCKED status."""
    
    # Mock the genai module
    with patch("backend.core.compliance_engine.genai") as mock_genai:
        # Configure the mock model response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "YES"  # Simulate Gemini saying "YES" (violation found)
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Instantiate service (will see LLM_API_KEY from env or defaults, but we mocked genai anyway)
        # We need to ensure _gemini_configured is True. 
        # The service checks os.environ.get("LLM_API_KEY"). 
        # We can force it by patching os.environ or just ensuring the mocked init works.
        
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            svc = ComplianceEngineService(rules_path=rules_file)
            
            # Run check on "offensive" text
            res = svc.compliance_check("You are an idiot.", {"module": "TEST"})
            
            # Verify results
            assert res.status == "BLOCKED"
            assert res.final_answer == "Offensive content detected."
            assert res.triggered_rule_ids == [99]
            
            # Verify prompt construction
            mock_model.generate_content.assert_called_once()
            call_args = mock_model.generate_content.call_args
            prompt_text = call_args[0][0]
            assert "Policy Violation Concept: \"Hate speech or offensive language\"" in prompt_text
            assert "You are an idiot." in prompt_text

def test_semantic_compliance_no_violation(rules_file):
    """Test that safe text passes the semantic check."""
    
    with patch("backend.core.compliance_engine.genai") as mock_genai:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "NO"  # Simulate Gemini saying "NO"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            svc = ComplianceEngineService(rules_path=rules_file)
            
            res = svc.compliance_check("Hello, how are you?", {"module": "TEST"})
            
            assert res.status == "OK"
            assert res.triggered_rule_ids == []

def test_semantic_compliance_api_failure_fails_open(rules_file):
    """Test that if Gemini API fails, we fail open (return False/OK) unless configured otherwise."""
    
    with patch("backend.core.compliance_engine.genai") as mock_genai:
        mock_model = MagicMock()
        # Simulate API error
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model
        
        with patch.dict("os.environ", {"LLM_API_KEY": "fake_key"}):
            svc = ComplianceEngineService(rules_path=rules_file)
            
            # Should log warning and return False (no violation) -> OK status
            res = svc.compliance_check("Some text", {"module": "TEST"})
            
            assert res.status == "OK"
