"""
MCP Server for module-specific document retrieval.

Exposes retrieval capabilities for each of the 5 business modules:
- Credit & Loans
- Investments
- Insurance
- Retirement/NPS
- Taxation
"""

import logging
from typing import Dict, Any, Optional, List

from mcp.server import Server
from mcp.types import TextContent

from ..stores.vector_store_setup import VectorStoreManager, get_vector_store_for_module
from ..retrieval import RetrievalPipeline
from ..config import MODULES
from .tools import ALL_TOOLS
from .handlers import (
    handle_query_credit,
    handle_query_investment,
    handle_query_insurance,
    handle_query_retirement,
    handle_query_taxation,
)

logger = logging.getLogger(__name__)


class RetrievalMCPServer:
    """
    MCP Server for module-specific document retrieval.
    
    Exposes 5 tools, one for each business module:
    - query_credit_documents
    - query_investment_documents
    - query_insurance_documents
    - query_retirement_documents
    - query_taxation_documents
    """
    
    def __init__(self):
        """Initialize the MCP server with retrieval pipelines for each module."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize VectorStoreManager (singleton)
        self.vector_store_manager = VectorStoreManager()
        
        # Initialize RetrievalPipeline for each module
        self.pipelines: Dict[str, RetrievalPipeline] = {}
        for module in MODULES:
            try:
                store = self.vector_store_manager.get_store(module)
                self.pipelines[module] = RetrievalPipeline(
                    module_name=module,
                    store=store
                )
                self.logger.info(f"✓ Initialized retrieval pipeline for module '{module}'")
            except Exception as e:
                self.logger.error(f"✗ Failed to initialize retrieval pipeline for module '{module}': {e}")
                raise
        
        # Create MCP server instance
        self.server = Server("financial_rag_retrieval")
        
        # Map tool names to their handlers
        self.tool_handlers = {
            "query_credit_documents": handle_query_credit,
            "query_investment_documents": handle_query_investment,
            "query_insurance_documents": handle_query_insurance,
            "query_retirement_documents": handle_query_retirement,
            "query_taxation_documents": handle_query_taxation,
        }
        
        # Register tools and handlers
        self._register_tools()
        
        self.logger.info("MCP Server initialized successfully")
    
    def _register_tools(self):
        """Register all module-specific tools with the MCP server."""
        # Register list_tools handler
        @self.server.list_tools()
        async def list_tools() -> list:
            """List all available tools."""
            return ALL_TOOLS
        
        # Register call_tool handler
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool execution."""
            # Get the handler for this tool
            handler = self.tool_handlers.get(name)
            if not handler:
                return [TextContent(
                    type="text",
                    text=f"Error: Unknown tool '{name}'"
                )]
            
            # Inject server instance into handler arguments
            arguments['server_instance'] = self
            
            # Call the handler
            try:
                return await handler(**arguments)
            except Exception as e:
                self.logger.error(f"Error executing tool '{name}': {e}", exc_info=True)
                module_name = name.replace("query_", "").replace("_documents", "")
                return self._handle_error(e, module_name)
    
    def get_server(self):
        """
        Get the underlying MCP Server instance.
        
        Returns:
            Server instance
        """
        return self.server
    
    def get_pipeline_for_module(self, module: str) -> RetrievalPipeline:
        """
        Get RetrievalPipeline instance for a specific module.
        
        Args:
            module: Module identifier (credit, investment, insurance, retirement, taxation)
        
        Returns:
            RetrievalPipeline instance for the module
        
        Raises:
            ValueError: If module is not recognized
        """
        if module not in self.pipelines:
            raise ValueError(
                f"Unknown module: {module}. "
                f"Valid modules: {', '.join(MODULES)}"
            )
        return self.pipelines[module]
    
    def _format_mcp_response(self, results: Dict[str, Any]) -> List[TextContent]:
        """
        Format retrieval results for MCP TextContent response.
        
        Args:
            results: Results dictionary from RetrievalPipeline.query()
        
        Returns:
            List of TextContent objects for MCP response
        """
        chunks = results.get("chunks", [])
        
        # Build formatted response
        response_parts = []
        
        # Summary
        response_parts.append(
            f"Found {results['total_results']} results for query: '{results['query']}'\n"
            f"Module: {results['module']}\n"
            f"HyDE used: {results.get('hyde_used', False)}\n"
            f"Hybrid search used: {results.get('hybrid_used', False)}\n"
            f"Reranking used: {results.get('rerank_used', False)}\n"
            f"{'='*80}\n"
        )
        
        # Chunks
        if chunks:
            for idx, chunk in enumerate(chunks, 1):
                chunk_text = f"\n[Result {idx}] (Score: {chunk.get('score', 0):.4f})\n"
                chunk_text += f"Document: {chunk.get('document_filename', 'N/A')}\n"
                chunk_text += f"Page: {chunk.get('page_number', 'N/A')}\n"
                chunk_text += f"Chunk Index: {chunk.get('chunk_index', 'N/A')}\n"
                chunk_text += f"\nContent:\n{chunk.get('text', 'N/A')}\n"
                chunk_text = f"\nContent (with local context):\n{chunk.get("expanded_text") or chunk.get("text", "N/A")}\n"
                chunk_text += f"{'-'*80}\n"
                response_parts.append(chunk_text)
        else:
            response_parts.append("\nNo results found.\n")
        
        return [TextContent(type="text", text="".join(response_parts))]
    
    def _handle_error(self, error: Exception, module: str) -> List[TextContent]:
        """
        Format error for MCP response.
        
        Args:
            error: Exception that occurred
            module: Module name where error occurred
        
        Returns:
            List of TextContent objects with error message
        """
        error_msg = (
            f"Error querying {module} module: {str(error)}\n"
            f"Please check your query and try again."
        )
        self.logger.error(f"Error in {module} module: {error}", exc_info=True)
        return [TextContent(type="text", text=error_msg)]

