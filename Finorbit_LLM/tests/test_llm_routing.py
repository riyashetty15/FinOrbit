"""
Test script for LLM-based RAG routing
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.router import RouterAgent
import logging

logging.basicConfig(level=logging.INFO)

def test_routing():
    """Test various queries to verify LLM routing"""
    router = RouterAgent()
    
    test_cases = [
        # Should route to RAG (specific data queries)
        ("what is the trail commission of parag parekh flexi cap fund?", "rag_agent"),
        ("tell me about SEBI guidelines for mutual funds 2025", "rag_agent"),
        ("what is the NAV of HDFC Balanced Advantage Fund?", "rag_agent"),
        ("show me the factsheet of ICICI Prudential Equity Fund", "rag_agent"),
        
        # Should NOT route to RAG (general queries)
        ("what is a mutual fund?", "investment_coach"),
        ("how do I start investing?", "investment_coach"),
        ("should I invest in equity or debt?", "investment_coach"),
        ("what is my tax liability?", "tax_planner"),
        ("how to calculate retirement corpus?", "retirement_planner"),
    ]
    
    print("\n" + "="*80)
    print("TESTING LLM-BASED RAG ROUTING")
    print("="*80 + "\n")
    
    for query, expected_agent in test_cases:
        result = router.route(query)
        status = "✓" if result == expected_agent else "✗"
        
        print(f"{status} Query: {query}")
        print(f"  Expected: {expected_agent}")
        print(f"  Got: {result}")
        print()

if __name__ == "__main__":
    test_routing()
