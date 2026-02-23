"""
Knowledge Lookup Tool for Finorbit Backend
- Token safe
- Integrated RAG Service
- Summary-first design
"""

import os
import logging
from typing import Dict, Any, Optional, List
import httpx
from dotenv import load_dotenv

load_dotenv()


# Setup RAG-specific logger (initialize only once, avoid duplicate handlers)
rag_logger = logging.getLogger("rag_logger")
rag_logger.setLevel(logging.INFO)
rag_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs", "rag.log"))


def _attach_rag_file_handler(target_logger: logging.Logger) -> None:
    """Ensure the target logger writes to rag.log without adding duplicate handlers."""
    if any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == rag_log_path for handler in target_logger.handlers):
        return

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(rag_log_path), exist_ok=True)

    file_handler = logging.FileHandler(rag_log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    target_logger.addHandler(file_handler)


_attach_rag_file_handler(rag_logger)
rag_logger.propagate = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_attach_rag_file_handler(logger)
logger.propagate = False

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8081/query")
RAG_HEALTH_CHECK_URL = os.getenv("RAG_HEALTH_CHECK_URL", "http://localhost:8081/health")

VALID_MODULES = ["credit", "investment", "insurance", "retirement", "taxation"]

# Safety limits (relaxed for backend usage)
MAX_CHUNKS = 8
MAX_CONTENT_CHARS = 4000   # 400 tokens per chunk approx
MAX_TOTAL_CHARS = 16000    # absolute safety net

# RAG Service Health Check
RAG_HEALTH_CHECK_TIMEOUT = 5  # seconds
RAG_QUERY_TIMEOUT = 20  # seconds


def _check_rag_config():
    """Warn if using localhost in production settings."""
    env = os.getenv("ENV", "development").lower()
    if env == "production" and "localhost" in RAG_API_URL:
        logger.warning(f"[WARNING]  PRODUCTION WARNING: RAG_API_URL is set to localhost ({RAG_API_URL}). Ensure this is intended.")
    elif "localhost" in RAG_API_URL or "127.0.0.1" in RAG_API_URL:
        logger.info(f"ℹ️  RAG Service configured for local development: {RAG_API_URL}")

_check_rag_config()


async def check_rag_health() -> bool:
    """
    Check if RAG service is available.
    
    Returns:
        bool: True if RAG service is healthy, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=RAG_HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(RAG_HEALTH_CHECK_URL)
            is_healthy = response.status_code == 200
            if is_healthy:
                logger.debug("[OK] RAG Service health check: OK")
            else:
                logger.warning(f"[WARNING]  RAG Service health check failed with status {response.status_code}")
            return is_healthy
    except httpx.TimeoutException:
        logger.error(f"[ERROR] RAG Service health check timed out ({RAG_HEALTH_CHECK_TIMEOUT}s)")
        return False
    except Exception as e:
        logger.error(f"[ERROR] RAG Service health check failed: {type(e).__name__}: {e}")
        return False

async def knowledge_lookup(
    query: str,
    module: str,
    top_k: int = 5,
    doc_type: Optional[str] = None,
    year: Optional[str] = None,
    issuer: Optional[str] = None,
    regulator_tag: Optional[str] = None,
    security: Optional[str] = None,
    is_current: Optional[bool] = None,
    pii: Optional[bool] = None,
    compliance_tags_any: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the RAG knowledge base for financial information.
    
    This tool connects to the RAG system to retrieve relevant financial documents
    based on the query and filters provided. Returns summarized, truncated results
    optimized for agent context.
    
    Args:
        query: The search query text
        module: Domain module (credit, investment, insurance, retirement, taxation)
        top_k: Number of chunks to retrieve (default 5)
        doc_type: Filter by document type
        year: Filter by document year
        issuer: Filter by document issuer
        regulator_tag: Filter by regulatory tag
        security: Filter by security level
        is_current: Filter for current documents only
        pii: Filter documents containing PII
        compliance_tags_any: Filter by compliance tags (any match)
        
    Returns:
        Dict containing:
        - found: Whether relevant documents were found
        - results: List of relevant document chunks with metadata
        - error: Error message if request failed
    """
    logger.info("[CONFIG] RAG SERVICE CALL: knowledge_lookup")
    rag_logger.info(f"[RAG] knowledge_lookup called | query='{query[:80]}' | module='{module}' | top_k={top_k}")

    module = module.strip().lower()
    if module not in VALID_MODULES:
        rag_logger.warning(f"[RAG] Invalid module: {module}")
        return {
            "error": f"Invalid module: {module}. Valid modules: {', '.join(VALID_MODULES)}",
            "found": False,
            "results": []
        }

    if not query or not query.strip():
        rag_logger.warning("[RAG] Query text cannot be empty")
        return {
            "error": "Query text cannot be empty",
            "found": False,
            "results": []
        }

    # Construct complete payload matching the RAG endpoint schema
    payload = {
        "query": query,
        "module": module,
        "top_k": min(top_k, MAX_CHUNKS),
        "doc_type": doc_type,
        "year": year,
        "issuer": issuer,
        "regulator_tag": regulator_tag,
        "security": security,
        "is_current": is_current,
        "pii": pii,
        "compliance_tags_any": compliance_tags_any,
    }

    rag_logger.info(f"[RAG] Sending Full Payload: {str(payload)}")

    try:
        headers = {}
        if trace_id:
            headers["X-Trace-ID"] = str(trace_id)
        async with httpx.AsyncClient(timeout=RAG_QUERY_TIMEOUT) as client:
            response = await client.post(RAG_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        chunks = data.get("chunks", [])[:MAX_CHUNKS]

        rag_logger.info(f"[RAG] Retrieved {len(chunks)} chunks for query='{query[:80]}' | module='{module}'")
        for idx, c in enumerate(chunks):
            full_text = c.get('text', '')
            rag_logger.info(f"[RAG] Chunk {idx}: doc='{c.get('document_filename')}', idx={c.get('chunk_index')}, score={c.get('score')}, text='{full_text}'")

        results = []
        total_chars = 0

        for c in chunks:
            # Prefer contextual_text if present, else fallback object to text
            text = c.get("contextual_text")
            if not text:
                text = c.get("text", "")
            text = text[:MAX_CONTENT_CHARS]

            total_chars += len(text)
            if total_chars > MAX_TOTAL_CHARS:
                break

            # Return in format expected by retrieval_service
            results.append({
                "text": text,  # Changed from "content" to "text"
                "score": c.get("score"),  # Add score at top level
                "document_id": c.get("metadata", {}).get("document_id"),  # Extract from metadata
                "id": c.get("id"),  # Keep chunk ID
                "similarity_score": c.get("score"),  # Keep for backward compatibility
                "document_filename": c.get("metadata", {}).get("filename"),  # Extract from metadata
                "chunk_index": c.get("chunk_index"),
                "metadata": c.get("metadata", {})  # Include full metadata
            })

        rag_logger.info(f"[RAG] Final results count: {len(results)} | Used in output: {len(results) > 0}")
        if len(results) > 0:
            for idx, r in enumerate(results):
                rag_logger.info(f"[RAG] Used in output: Chunk {idx}: doc='{r['document_filename']}', idx={r['chunk_index']}, score={r['similarity_score']}, text='{r['text'] [:100]}'...")
        else:
            rag_logger.info(f"[RAG] No chunks used in output for query='{query[:80]}'")

        return {
            "found": len(results) > 0,
            "results": results,
            "total_chunks": len(results)
        }

    except httpx.HTTPStatusError as e:
        error_msg = f"RAG API error: {e.response.status_code} - {e.response.text}"
        logger.exception("knowledge_lookup HTTP error")
        rag_logger.error(f"[RAG] HTTP error: {error_msg}")
        return {
            "error": error_msg,
            "found": False,
            "results": []
        }
    except httpx.RequestError as e:
        error_msg = f"RAG API connection error: {str(e)}"
        logger.exception("knowledge_lookup connection error")
        rag_logger.error(f"[RAG] Connection error: {error_msg}")
        return {
            "error": error_msg,
            "found": False,
            "results": []
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception("knowledge_lookup failed")
        rag_logger.error(f"[RAG] Unexpected error: {error_msg}")
        return {
            "error": error_msg,
            "found": False,
            "results": []
        }
