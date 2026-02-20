from backend.server import _infer_compliance_module


def test_infers_investment_module_from_multi_agent_execution_order():
    module = _infer_compliance_module(
        user_input="Which mutual funds will guarantee profitable returns after 5 years?",
        agent_type="rag_agent",
        agents_used=["rag_agent", "investment_coach", "tax_planner"],
        execution_order=["rag_agent", "investment_coach", "tax_planner"],
    )
    assert module == "SIP_INVESTMENT"


def test_infers_tax_module_from_keywords_when_unknown_agent():
    module = _infer_compliance_module(
        user_input="How do I claim 80C deduction in ITR?",
        agent_type="rag_agent",
        agents_used=["rag_agent"],
        execution_order=["rag_agent"],
    )
    assert module == "TAX"
