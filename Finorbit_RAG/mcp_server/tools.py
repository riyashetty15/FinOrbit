"""
MCP Tool definitions for module-specific document retrieval.

Defines 5 tools, one for each business module:
- query_credit_documents
- query_investment_documents
- query_insurance_documents
- query_retirement_documents
- query_taxation_documents
"""

from mcp.types import Tool


def create_query_tool(module_name: str, module_display_name: str) -> Tool:
    """
    Create a query tool for a specific module.
    
    Args:
        module_name: Module identifier (credit, investment, etc.)
        module_display_name: Human-readable module name
        
    Returns:
        Tool definition for the module
    """
    return Tool(
        name=f"query_{module_name}_documents",
        description=f"Search and retrieve relevant documents from the {module_display_name} module",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                },
                "use_hyde": {
                    "type": "boolean",
                    "description": "Use HyDE (Hypothetical Document Embeddings) for enhanced retrieval",
                    "default": False
                },
                "use_hybrid": {
                    "type": "boolean",
                    "description": "Use hybrid search (vector + full-text search)",
                    "default": False
                },
                "use_rerank": {
                    "type": "boolean",
                    "description": "Use reranking for improved relevance",
                    "default": False
                },
                "doc_type": {
                    "type": "string",
                    "description": "Filter by document type (optional)"
                },
                "year": {
                    "type": "integer",
                    "description": "Filter by year (optional)"
                },
                "filename": {
                    "type": "string",
                    "description": "Filter by filename (optional)"
                },
                "module": {"type": "string"},
                "issuer": {"type": "string"},
                "language": {"type": "string"},
                "regulator_tag": {"type": "string"},
                "version_id": {"type": "string"},
                "security": {"type": "string"},
                "is_current": {"type": "boolean"},
                "pii": {"type": "boolean"},
                "compliance_tags_any": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
            "required": ["query"]
        }
    )


# Module-specific tool definitions
QUERY_CREDIT_TOOL = create_query_tool("credit", "Credit & Loans")
QUERY_INVESTMENT_TOOL = create_query_tool("investment", "Investments / Mutual Funds & SIP")
QUERY_INSURANCE_TOOL = create_query_tool("insurance", "Insurance")
QUERY_RETIREMENT_TOOL = create_query_tool("retirement", "Retirement / NPS")
QUERY_TAXATION_TOOL = create_query_tool("taxation", "Taxation")


# List of all tools for easy registration
ALL_TOOLS = [
    QUERY_CREDIT_TOOL,
    QUERY_INVESTMENT_TOOL,
    QUERY_INSURANCE_TOOL,
    QUERY_RETIREMENT_TOOL,
    QUERY_TAXATION_TOOL,
]

