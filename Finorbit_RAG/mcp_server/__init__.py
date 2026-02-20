"""
MCP Server for module-specific document retrieval.

Exposes retrieval capabilities for each of the 5 business modules:
- Credit & Loans
- Investments
- Insurance
- Retirement/NPS
- Taxation
"""

from .retrieval_server import RetrievalMCPServer
from .tools import ALL_TOOLS
from .handlers import (
    handle_query_credit,
    handle_query_investment,
    handle_query_insurance,
    handle_query_retirement,
    handle_query_taxation,
)

__all__ = [
    "RetrievalMCPServer",
    "ALL_TOOLS",
    "handle_query_credit",
    "handle_query_investment",
    "handle_query_insurance",
    "handle_query_retirement",
    "handle_query_taxation",
]

