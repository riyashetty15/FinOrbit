"""
Pytest configuration for FinOrbit LLM test suite.

Tests in LIVE_TEST_FILES require a live OpenAI API key and are automatically
skipped in CI when OPENAI_API_KEY is not set.
"""

import os
import pytest

# Test files that make real network/API calls and need OPENAI_API_KEY
LIVE_TEST_FILES = {
    "test_citations.py",
    "test_llm_routing.py",
    "test_orchestrator_query_integrity.py",
    "test_rag_agent.py",
    "test_rag_tool.py",
    "test_router_load.py",
    "test_router_regression.py",
}


def pytest_collection_modifyitems(config, items):
    """Skip live tests when OPENAI_API_KEY is not configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return  # Key present — run everything

    skip_live = pytest.mark.skip(reason="OPENAI_API_KEY not set — skipping live API tests")
    for item in items:
        filename = os.path.basename(item.fspath)
        if filename in LIVE_TEST_FILES:
            item.add_marker(skip_live)
