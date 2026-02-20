#!/usr/bin/env python3
"""
Entry point script to run the Financial RAG Retrieval MCP Server.

Usage:
    python run_mcp_server.py

Or make it executable:
    chmod +x run_mcp_server.py
    ./run_mcp_server.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.stdio import stdio_server
from mcp_server.retrieval_server import RetrievalMCPServer

# Configure logging to stderr (stdout is for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the MCP server."""
    logger.info("=" * 80)
    logger.info("Financial RAG Retrieval MCP Server")
    logger.info("=" * 80)
    
    try:
        # Initialize the MCP server
        logger.info("Initializing server...")
        retrieval_server = RetrievalMCPServer()
        server = retrieval_server.server
        
        logger.info("Server initialized successfully")
        logger.info("Available modules: credit, investment, insurance, retirement, taxation")
        logger.info("Ready to accept connections via stdio")
        logger.info("=" * 80)
        
        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown complete")
        sys.exit(0)

