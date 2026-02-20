"""
Tool handlers for module-specific document retrieval.

Each handler processes MCP tool calls and routes them to the appropriate
RetrievalPipeline instance.
"""

import logging
from typing import Dict, Any, Optional
from mcp.types import TextContent

logger = logging.getLogger(__name__)


def create_query_handler(module_name: str):
    """
    Create a query handler function for a specific module.
    
    Args:
        module_name: Module identifier (credit, investment, insurance, retirement, taxation)
    
    Returns:
        Handler function that processes tool calls
    """
    async def handler(
        query: str,
        top_k: int = 5,
        use_hyde: bool = False,
        use_hybrid: bool = False,
        use_rerank: bool = False,
        doc_type: Optional[str] = None,
        year: Optional[int] = None,
        filename: Optional[str] = None,
        module: Optional[str] = None,
        issuer: Optional[str] = None,
        language: Optional[str] = None,
        regulator_tag: Optional[str] = None,
        version_id: Optional[str] = None,
        security: Optional[str] = None,
        is_current: Optional[bool] = None,
        pii: Optional[bool] = None,
        compliance_tags_any: Optional[list] = None,
        server_instance=None
    ) -> list[TextContent]:
        """
        Handle query for a specific module.
        
        Args:
            query: The search query
            top_k: Number of results to return
            use_hyde: Use HyDE for enhanced retrieval
            use_hybrid: Use hybrid search (vector + FTS)
            use_rerank: Use reranking for improved relevance
            doc_type: Filter by document type (optional)
            year: Filter by year (optional)
            filename: Filter by filename (optional)
            server_instance: RetrievalMCPServer instance (injected)
        
        Returns:
            List of TextContent objects with results
        """
        if server_instance is None:
            return [TextContent(
                type="text",
                text=f"Error: Server instance not available for {module_name} module"
            )]
        
        try:
            # Get pipeline for this module
            pipeline = server_instance.get_pipeline_for_module(module_name)
            
            # Build kwargs for metadata filtering
            kwargs = {}
            if doc_type: kwargs['doc_type'] = doc_type
            if year: kwargs['year'] = year
            if filename: kwargs['filename'] = filename
            if module: kwargs['module'] = module
            if issuer: kwargs['issuer'] = issuer
            if language: kwargs['language'] = language
            if regulator_tag: kwargs['regulator_tag'] = regulator_tag
            if version_id: kwargs['version_id'] = version_id
            if security: kwargs['security'] = security

            # booleans
            if is_current is not None: kwargs['is_current'] = is_current
            if pii is not None: kwargs['pii'] = pii

            # list membership
            if compliance_tags_any: kwargs['compliance_tags_any'] = compliance_tags_any
            
            # Execute query
            results = pipeline.query(
                query_text=query,
                top_k=top_k,
                use_hyde=use_hyde,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                **kwargs
            )
            
            # Format results for MCP response
            return server_instance._format_mcp_response(results)
            
        except Exception as e:
            logger.error(f"Error in {module_name} handler: {e}", exc_info=True)
            return server_instance._handle_error(e, module_name)
    
    return handler


# Create handlers for each module
handle_query_credit = create_query_handler("credit")
handle_query_investment = create_query_handler("investment")
handle_query_insurance = create_query_handler("insurance")
handle_query_retirement = create_query_handler("retirement")
handle_query_taxation = create_query_handler("taxation")

