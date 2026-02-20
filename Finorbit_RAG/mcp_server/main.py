"""
Main entry point for the Financial RAG Retrieval MCP Server.

This script initializes and runs the MCP server using stdio transport,
exposing retrieval capabilities for all 5 business modules.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.stdio import stdio_server
from mcp import types

from .retrieval_server import RetrievalMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # Log to stderr so stdout is for MCP protocol
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main entry point for the MCP server.
    
    Initializes the RetrievalMCPServer and runs it with stdio transport.
    """
    logger.info("Starting Financial RAG Retrieval MCP Server...")
    
    try:
        # Initialize the MCP server
        retrieval_server = RetrievalMCPServer()
        server = retrieval_server.server
        
        logger.info("MCP Server initialized successfully")
        logger.info("Available modules: credit, investment, insurance, retirement, taxation")
        
        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

