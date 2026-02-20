"""
FastAPI application for Financial RAG Pipeline.

Provides REST API endpoints for:
- Document ingestion (PDF upload)
- Document querying (semantic search)
"""

import os
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import pipeline classes and utilities using absolute imports
from stores import get_vector_store_for_module, clear_manager_cache
from ingestion import TextIngestionPipeline
from retrieval import RetrievalPipeline
from config import MODULES, validate_module
from services import VectorStoreCleanupService
from core.llm_setup import configure_global_llm

# Import background job management
from ingestion.db_ingestion_models import create_job, get_job, list_jobs
from ingestion.background_worker import create_background_task_wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pipeline Instance Cache
# ============================================================================

_ingestion_pipelines: Dict[str, TextIngestionPipeline] = {}
_retrieval_pipelines: Dict[str, RetrievalPipeline] = {}
_cleanup_service: VectorStoreCleanupService = None


def get_cleanup_service() -> VectorStoreCleanupService:
    """Get or create cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = VectorStoreCleanupService()
    return _cleanup_service


def get_ingestion_pipeline(module: str) -> TextIngestionPipeline:
    """
    Get or create cached ingestion pipeline for a module.
    
    Args:
        module: Module name (credit, investment, insurance, retirement, taxation)
    
    Returns:
        TextIngestionPipeline instance for the module
    
    Raises:
        ValueError: If module is invalid
    """
    module = module.strip().lower()
    if not validate_module(module):
        raise ValueError(f"Invalid module: {module}. Valid modules: {MODULES}")
    
    if module not in _ingestion_pipelines:
        logger.info(f"Initializing ingestion pipeline for module '{module}'")
        try:
            store = get_vector_store_for_module(module)
            pipeline = TextIngestionPipeline(module_name=module, store=store)
            _ingestion_pipelines[module] = pipeline
            logger.info(f"✓ Ingestion pipeline initialized for module '{module}'")
        except Exception as e:
            logger.error(f"Failed to initialize ingestion pipeline for '{module}': {e}")
            raise
    
    return _ingestion_pipelines[module]


def get_retrieval_pipeline(module: str) -> RetrievalPipeline:
    """
    Get or create cached retrieval pipeline for a module.
    
    Args:
        module: Module name (credit, investment, insurance, retirement, taxation)
    
    Returns:
        RetrievalPipeline instance for the module
    
    Raises:
        ValueError: If module is invalid
    """
    module = module.strip().lower()
    if not validate_module(module):
        raise ValueError(f"Invalid module: {module}. Valid modules: {MODULES}")
    
    if module not in _retrieval_pipelines:
        logger.info(f"Initializing retrieval pipeline for module '{module}'")
        try:
            store = get_vector_store_for_module(module)
            pipeline = RetrievalPipeline(module_name=module, store=store)
            _retrieval_pipelines[module] = pipeline
            logger.info(f"✓ Retrieval pipeline initialized for module '{module}'")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval pipeline for '{module}': {e}")
            raise
    
    return _retrieval_pipelines[module]


def clear_pipeline_cache(module: str):
    """
    Clear cached pipeline instances for a module and reinitialize.
    
    This ensures LlamaIndex indexes are cleared and reinitialized after cleanup.
    """
    module = module.strip().lower()
    # Remove from caches
    if module in _ingestion_pipelines:
        del _ingestion_pipelines[module]
        logger.info(f"Cleared ingestion pipeline cache for module '{module}'")
    
    if module in _retrieval_pipelines:
        del _retrieval_pipelines[module]
        logger.info(f"Cleared retrieval pipeline cache for module '{module}'")
    
    # Clear vector store manager cache to force reinitialization
    clear_manager_cache()
    logger.info(f"Cleared vector store manager cache for module '{module}'")
    
    # Reinitialize pipelines (will create fresh indexes)
    try:
        get_ingestion_pipeline(module)
        get_retrieval_pipeline(module)
        logger.info(f"Reinitialized pipelines for module '{module}'")
    except Exception as e:
        logger.warning(f"Failed to reinitialize pipelines for module '{module}': {e}")


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Financial RAG API starting up...")
    
    # Configure LLM (OpenAI vs Gemini)
    configure_global_llm()
    
    logger.info(f"Available modules: {MODULES}")
    yield
    # Shutdown
    logger.info("Financial RAG API shutting down...")


app = FastAPI(
    title="Financial RAG API",
    description="REST API for document ingestion and querying in modular RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Request/Response Models
# ============================================================================

class IngestMetadata(BaseModel):
    """Metadata for document ingestion"""
    module: str = Field(..., description="Module name (credit, investment, insurance, retirement, taxation)")
    doc_type: Optional[str] = Field(None, description="Document type")
    year: Optional[str] = Field(None, description="Year")
    tags: Optional[Dict[str, Any]] = Field(None, description="Additional tags")
    additional_modules: Optional[List[str]] = Field(None, description="Additional modules for multi-module tagging")
    source_type: Optional[str] = Field("upload", description="Source type")


class QueryRequest(BaseModel):
    """Request model for document querying"""
    query: str = Field(..., description="Search query text")
    module: str = Field(..., description="Module name (credit, investment, insurance, retirement, taxation)")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Financial RAG API",
        "version": "1.0.0"
    }


@app.get("/modules")
async def list_modules():
    """List all available modules"""
    return {
        "modules": MODULES,
        "count": len(MODULES)
    }


@app.get("/modules/{module}/stats")
async def get_module_stats(module: str):
    """
    Get statistics about data in a module (chunks, documents count).
    
    Parameters:
    - module: Module name (credit, investment, insurance, retirement, taxation)
    """
    try:
        # Validate module
        module = module.strip().lower()
        if not validate_module(module):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid module: {module}. Valid modules: {MODULES}"
            )
        
        cleanup_service = get_cleanup_service()
        stats = cleanup_service.get_module_statistics(module)
        
        return stats
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting statistics for module '{module}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/modules/{module}/clear")
async def clear_module(
    module: str,
    confirm: bool = False,
    scope: str = "all"
):
    """
    Clear vector store data for a specific module.
    
    WARNING: This is a destructive operation that permanently deletes data!
    
    Parameters:
    - module: Module name (credit, investment, insurance, retirement, taxation) (required)
    - confirm: Must be true to proceed with deletion (required, query parameter)
    - scope: What to clear - "chunks", "documents", or "all" (default: "all", query parameter)
        - "chunks": Delete only chunks, keep documents
        - "documents": Delete documents (cascades to chunks via FK)
        - "all": Delete both chunks and documents
    
    Returns:
    - Statistics about what was deleted
    - Status message
    
    Example:
    POST /modules/credit/clear?confirm=true&scope=all
    """
    try:
        # Validate module
        module = module.strip().lower()
        if not validate_module(module):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid module: {module}. Valid modules: {MODULES}"
            )
        
        # Validate scope
        if scope not in ["chunks", "documents", "all"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scope: {scope}. Must be 'chunks', 'documents', or 'all'"
            )
        
        # Require confirmation
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required. Set confirm=true to proceed with deletion."
            )
        
        logger.warning(
        f"CLEAR OPERATION REQUESTED for module '{module}' "
        )
        
        # Perform cleanup
        cleanup_service = get_cleanup_service()
        result = cleanup_service.clear_module(
            module=module,
            scope=scope,  # type: ignore
            confirm=confirm
        )
        
        # Clear and reinitialize LlamaIndex indexes
        logger.info(f"Clearing and reinitializing LlamaIndex indexes for module '{module}'")
        clear_pipeline_cache(module)
        
        logger.warning(
            f"Successfully cleared module '{module}' (scope: {scope}). "
            f"Deleted {result['statistics']}"
        )
        
        return JSONResponse(
            content=result,
            status_code=200
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during cleanup for module '{module}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/documents/id/{document_id}")
async def delete_document_by_id(
    document_id: int,
    confirm: bool = False
):
    """
    Delete document and associated chunks by document ID.
    
    WARNING: This is a destructive operation that permanently deletes data!
    
    Parameters:
    - document_id: Document ID from the documents table (required, path parameter)
    - confirm: Must be true to proceed with deletion (required, query parameter)
    
    Returns:
    - Statistics about what was deleted (documents, chunks per module, modules affected)
    - Status message
    
    Example:
    DELETE /documents/id/5?confirm=true
    """
    try:
        # Require confirmation
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required. Set confirm=true to proceed with deletion."
            )
        
        logger.warning(
            f"DELETE DOCUMENT REQUESTED for document_id={document_id} "
            f"(CONFIRMED: {confirm})"
        )
        
        # Perform deletion
        cleanup_service = get_cleanup_service()
        result = cleanup_service.delete_document_by_id(
            document_id=document_id,
            confirm=confirm
        )
        
        # Clear and reinitialize LlamaIndex indexes for all affected modules
        modules_affected = result["statistics"]["modules_affected"]
        for module in modules_affected:
            logger.info(f"Clearing and reinitializing LlamaIndex indexes for module '{module}'")
            clear_pipeline_cache(module)
        
        logger.warning(
            f"Successfully deleted document with document_id={document_id}. "
            f"Deleted {result['statistics']['documents_deleted']} document(s) and "
            f"{sum(result['statistics']['chunks_deleted_by_module'].values())} chunk(s) "
            f"across {len(modules_affected)} module(s)"
        )
        
        return JSONResponse(
            content=result,
            status_code=200
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        # Document not found or confirmation missing
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error during deletion for document_id={document_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete("/documents/{filename:path}")
async def delete_document_by_filename(
    filename: str,
    confirm: bool = False
):
    """
    Delete document(s) and associated chunks by filename.
    
    WARNING: This is a destructive operation that permanently deletes data!
    
    The filename is matched case-insensitively (exact match). If multiple documents
    have the same filename, all will be deleted.
    
    Parameters:
    - filename: Document filename (URL-encoded if contains special characters) (required, path parameter)
    - confirm: Must be true to proceed with deletion (required, query parameter)
    
    Returns:
    - Statistics about what was deleted (documents, chunks per module, modules affected)
    - Status message
    
    Example:
    DELETE /documents/my-document.pdf?confirm=true
    DELETE /documents/my%20document.pdf?confirm=true  (URL-encoded space)
    """
    try:
        # URL decode the filename to handle special characters
        from urllib.parse import unquote
        decoded_filename = unquote(filename)
        
        # Require confirmation
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation required. Set confirm=true to proceed with deletion."
            )
        
        logger.warning(
            f"DELETE DOCUMENT REQUESTED for filename '{decoded_filename}' "
            f"(CONFIRMED: {confirm})"
        )
        
        # Perform deletion
        cleanup_service = get_cleanup_service()
        result = cleanup_service.delete_document_by_filename(
            filename=decoded_filename,
            confirm=confirm
        )
        
        # Clear and reinitialize LlamaIndex indexes for all affected modules
        modules_affected = result["statistics"]["modules_affected"]
        for module in modules_affected:
            logger.info(f"Clearing and reinitializing LlamaIndex indexes for module '{module}'")
            clear_pipeline_cache(module)
        
        logger.warning(
            f"Successfully deleted document(s) with filename '{decoded_filename}'. "
            f"Deleted {result['statistics']['documents_deleted']} document(s) and "
            f"{sum(result['statistics']['chunks_deleted_by_module'].values())} chunk(s) "
            f"across {len(modules_affected)} module(s)"
        )
        
        return JSONResponse(
            content=result,
            status_code=200
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        # Document not found or confirmation missing
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error during deletion for filename '{filename}': {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest (PDF, DOCX, PPTX, images, etc.)"),
    module: str = Form(..., description="Module name (credit, investment, insurance, retirement, taxation)"),
    doc_type: Optional[str] = Form(None, description="Document type"),
    year: Optional[str] = Form(None, description="Year"),
    tags: Optional[str] = Form(None, description="Additional tags as JSON string"),
    issuer: Optional[str] = Form(None, description="Issuer / organization name"),
    regulator_tag: Optional[str] = Form(None, description="Regulator tag (RBI, SEBI, IRDAI, etc.)"),
    language: Optional[str] = Form("EN", description="Primary language (e.g., EN, HI, MR)"),
    jurisdiction: Optional[str] = Form("India", description="Jurisdiction / country"),

    version: Optional[str] = Form(None, description="Human-readable version string"),
    version_id: Optional[str] = Form(None, description="Canonical version ID"),
    effective_from: Optional[str] = Form(None, description="Effective from date (YYYY-MM-DD)"),
    valid_from: Optional[str] = Form(None, description="Valid from date (YYYY-MM-DD)"),
    valid_to: Optional[str] = Form(None, description="Valid to date (YYYY-MM-DD)"),

    security: Optional[str] = Form("Public", description="Security level (Public / Internal / Restricted)"),
    pii: Optional[bool] = Form(False, description="Contains PII"),
    is_current: Optional[bool] = Form(True, description="Is this the current version?"),

    compliance_tags: Optional[str] = Form(None, description='Compliance tags as JSON array, e.g. ["govt.rbi","banking.savings"]'),
    additional_modules: Optional[str] = Form(None,description='Additional modules as JSON array, e.g. ["credit","investment"]'),
):
    """
    Ingest a document into the vector store.
    
    Supports multiple file formats via LlamaParse integration:
    - Documents: PDF, DOCX, PPTX, XLSX, RTF, TXT
    - Images: JPG, JPEG, PNG, GIF, BMP, TIFF
    - E-books: EPUB
    - Apple formats: Pages, Keynote
    
    Falls back to PyMuPDF for PDFs when LlamaParse is unavailable.
    
    Parameters:
    - file: Document file to ingest (required)
    - module: Module name - credit, investment, insurance, retirement, or taxation (required)
    - doc_type: Document type (optional)
    - year: Year (optional)
    - tags: Additional tags as JSON string, e.g., '{"key": "value"}'. If not valid JSON, will be skipped (optional)
    """
    import json
    from config import get_llamaparse_config
    
    temp_file_path: Optional[str] = None
    
    try:
        # Get file extension
        if not file.filename or '.' not in file.filename:
            raise HTTPException(
                status_code=400,
                detail="File must have a valid extension"
            )
        
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        
        # Get LlamaParse config to check supported formats
        llamaparse_config = get_llamaparse_config()
        
        # Validate file type
        supported_formats = list(llamaparse_config.supported_formats) if llamaparse_config.enabled else ["pdf"]
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"File type '.{file_ext}' is not supported. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Validate module
        module = module.strip().lower()
        if not validate_module(module):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid module: {module}. Valid modules: {MODULES}"
            )
        
        # Parse optional JSON fields
        tags_dict = None
        if tags and tags.strip():
            try:
                tags_dict = json.loads(tags)
            except json.JSONDecodeError:
                # If tags is not valid JSON, skip it (don't raise error)
                logger.warning(f"Tags provided but not valid JSON, skipping: {tags}")
                tags_dict = None
        
        # Get original filename from uploaded file
        original_filename = file.filename if file.filename else f"unknown.{file_ext}"
        
        # Build metadata dictionary
        # ingest_metadata = {}
        # if doc_type:
        #     ingest_metadata["doc_type"] = doc_type
        # if year:
        #     ingest_metadata["year"] = year
        # if tags_dict:
        #     ingest_metadata["tags"] = tags_dict
        # # Store original filename in metadata so it can be used instead of temp filename
        # ingest_metadata["original_filename"] = original_filename

        # Enhanced Metadata Retrieval
        ingest_metadata: Dict[str, Any] = {}

        # core business metadata
        ingest_metadata["module"] = module  # helpful to keep here too
        if doc_type:
            ingest_metadata["doc_type"] = doc_type
        if year:
            ingest_metadata["year"] = year
        if tags_dict:
            ingest_metadata["tags"] = tags_dict
        if issuer:
            ingest_metadata["issuer"] = issuer
        if regulator_tag:
            ingest_metadata["regulator_tag"] = regulator_tag
        if language:
            ingest_metadata["language"] = language
        if jurisdiction:
            ingest_metadata["jurisdiction"] = jurisdiction
        if version:
            ingest_metadata["version"] = version
        if version_id:
            ingest_metadata["version_id"] = version_id
        if effective_from:
            ingest_metadata["effective_from"] = effective_from
        if valid_from:
            ingest_metadata["valid_from"] = valid_from
        if valid_to:
            ingest_metadata["valid_to"] = valid_to
        if security:
            ingest_metadata["security"] = security
        if pii is not None:
            ingest_metadata["pii"] = pii
        if is_current is not None:
            ingest_metadata["is_current"] = is_current

        # parse JSON-style arrays
        compliance_tags_list = None
        if compliance_tags and compliance_tags.strip():
            try:
                compliance_tags_list = json.loads(compliance_tags)
            except json.JSONDecodeError:
                logger.warning(f"Invalid compliance_tags JSON, skipping: {compliance_tags}")
        if compliance_tags_list:
            ingest_metadata["compliance_tags"] = compliance_tags_list

        additional_modules_list = None
        if additional_modules and additional_modules.strip():
            try:
                additional_modules_list = json.loads(additional_modules)
            except json.JSONDecodeError:
                logger.warning(f"Invalid additional_modules JSON, skipping: {additional_modules}")
        if additional_modules_list:
            additional_modules_list = [m.strip().lower() for m in additional_modules_list]
            ingest_metadata["additional_modules"] = additional_modules_list

        # store original filename so ingestion pipeline can use it
        ingest_metadata["original_filename"] = original_filename
        
        # Save uploaded file to temporary location (preserve extension)
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"rag_upload_{os.urandom(8).hex()}.{file_ext}")
        
        logger.info(f"Saving uploaded file to temporary location: {temp_file_path} (original: {original_filename}, type: {file_ext})")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved ({len(content)} bytes). Starting ingestion for module '{module}'")
        
        # Get ingestion pipeline
        pipeline = get_ingestion_pipeline(module)
        
        # Ingest document
        result = pipeline.ingest(
            document_path=temp_file_path,
            metadata=ingest_metadata
        )
        
        # Map result status to HTTP status code
        status_code = 200
        if result["status"] == "failed":
            status_code = 500
        elif result["status"] == "duplicate":
            status_code = 200  # Still 200, but indicate duplicate in response
        
        return JSONResponse(
            content=result,
            status_code=status_code
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")


@app.post("/query")
async def query_documents(
    query: str = Body(..., description="Search query text"),
    module: str = Body(..., description="Module name (credit, investment, insurance, retirement, taxation)"),
    top_k: int = Body(5, ge=1, le=50, description="Number of results to return"),

    # optional metadata filters
    doc_type: Optional[str] = Body(None, description="Filter by document type"),
    year: Optional[str] = Body(None, description="Filter by exact year"),
    year_min: Optional[int] = Body(None, description="Filter by minimum year (inclusive)"),
    issuer: Optional[str] = Body(None, description="Filter by issuer"),
    jurisdiction: Optional[str] = Body(None, description="Filter by jurisdiction (e.g., IN, US, UK)"),
    regulator_tag: Optional[str] = Body(None, description="Filter by regulator tag"),
    security: Optional[str] = Body(None, description="Filter by security level"),
    is_current: Optional[bool] = Body(None, description="Filter by current/archived"),
    pii: Optional[bool] = Body(None, description="Filter by PII flag"),
    effective_date: Optional[str] = Body(None, description="Filter by effective date (ISO format)"),
    version: Optional[str] = Body(None, description="Filter by document version"),
    compliance_tags_any: Optional[List[str]] = Body(
        None,
        description="Return chunks that have ANY of these compliance tags"
    )
):
    """
    Query documents from the vector store with production-grade metadata filtering.
    
    Returns relevant document chunks based on semantic similarity with full provenance.
    
    Parameters:
    - query: Search query text (required)
    - module: Module name - credit, investment, insurance, retirement, or taxation (required)
    - top_k: Number of results to return (default: 5, max: 50)
    - Filters: doc_type, year, year_min, issuer, jurisdiction, effective_date, version, etc.
    
    Returns:
    - results: List of citations with doc_id, source, page, chunk_id, text, score, metadata
    - found: Whether any results were retrieved
    - total_results: Count of results
    """
    module = module.strip().lower()
    try:
        # Validate module
        if not validate_module(module):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid module: {module}. Valid modules: {MODULES}"
            )
        
        # Validate query text
        if not query or not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query text cannot be empty"
            )
        
        # Validate top_k
        if top_k < 1 or top_k > 50:
            raise HTTPException(
                status_code=400,
                detail="top_k must be between 1 and 50"
            )
        
        logger.info(f"Querying module '{module}' with query: '{query[:50]}...'")
        
        # Get retrieval pipeline
        pipeline = get_retrieval_pipeline(module)
        
        # Query documents with all filters
        result = pipeline.query(
            query_text=query,
            top_k=top_k,
            doc_type=doc_type,
            year=year,
            year_min=year_min,
            issuer=issuer,
            jurisdiction=jurisdiction,
            regulator_tag=regulator_tag,
            security=security,
            is_current=is_current,
            pii=pii,
            effective_date=effective_date,
            version=version,
            compliance_tags_any=compliance_tags_any,
        )

        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/ingest_background")
async def ingest_document_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to ingest (PDF, DOCX, PPTX, images, etc.)"),
    module: str = Form(..., description="Module name (credit, investment, insurance, retirement, taxation)"),
    doc_type: Optional[str] = Form(None, description="Document type"),
    year: Optional[str] = Form(None, description="Year"),
    tags: Optional[str] = Form(None, description="Additional tags as JSON string"),
    issuer: Optional[str] = Form(None, description="Issuer / organization name"),
    regulator_tag: Optional[str] = Form(None, description="Regulator tag (RBI, SEBI, IRDAI, etc.)"),
    language: Optional[str] = Form("EN", description="Primary language (e.g., EN, HI, MR)"),
    jurisdiction: Optional[str] = Form("India", description="Jurisdiction / country"),

    version: Optional[str] = Form(None, description="Human-readable version string"),
    version_id: Optional[str] = Form(None, description="Canonical version ID"),
    effective_from: Optional[str] = Form(None, description="Effective from date (YYYY-MM-DD)"),
    valid_from: Optional[str] = Form(None, description="Valid from date (YYYY-MM-DD)"),
    valid_to: Optional[str] = Form(None, description="Valid to date (YYYY-MM-DD)"),

    security: Optional[str] = Form("Public", description="Security level (Public / Internal / Restricted)"),
    pii: Optional[bool] = Form(False, description="Contains PII"),
    is_current: Optional[bool] = Form(True, description="Is this the current version?"),

    compliance_tags: Optional[str] = Form(None, description='Compliance tags as JSON array, e.g. ["govt.rbi","banking.savings"]'),
    additional_modules: Optional[str] = Form(None, description='Additional modules as JSON array, e.g. ["credit","investment"]'),
):
    """
    Ingest a document into the vector store (ASYNC - returns immediately).
    
    This endpoint accepts file upload and returns immediately with a job_id.
    The actual processing happens in the background. Use GET /ingest/status/{job_id}
    to check the status and get results.
    
    Supports multiple file formats via LlamaParse integration:
    - Documents: PDF, DOCX, PPTX, XLSX, RTF, TXT
    - Images: JPG, JPEG, PNG, GIF, BMP, TIFF
    - E-books: EPUB
    - Apple formats: Pages, Keynote
    
    Falls back to PyMuPDF for PDFs when LlamaParse is unavailable.
    
    Parameters:
    - file: Document file to ingest (required)
    - module: Module name - credit, investment, insurance, retirement, or taxation (required)
    - doc_type: Document type (optional)
    - year: Year (optional)
    - tags: Additional tags as JSON string, e.g., '{"key": "value"}'. If not valid JSON, will be skipped (optional)
    
    Returns:
    - HTTP 202 Accepted
    - job_id: UUID for tracking the job
    - status: "pending"
    - Use GET /ingest/status/{job_id} to check progress
    """
    import json
    from config import get_llamaparse_config
    
    temp_file_path: Optional[str] = None
    
    try:
        # Get file extension
        if not file.filename or '.' not in file.filename:
            raise HTTPException(
                status_code=400,
                detail="File must have a valid extension"
            )
        
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        
        # Get LlamaParse config to check supported formats
        llamaparse_config = get_llamaparse_config()
        
        # Validate file type
        supported_formats = list(llamaparse_config.supported_formats) if llamaparse_config.enabled else ["pdf"]
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"File type '.{file_ext}' is not supported. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Validate module
        module = module.strip().lower()
        if not validate_module(module):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid module: {module}. Valid modules: {MODULES}"
            )
        
        # Parse optional JSON fields
        tags_dict = None
        if tags and tags.strip():
            try:
                tags_dict = json.loads(tags)
            except json.JSONDecodeError:
                logger.warning(f"Tags provided but not valid JSON, skipping: {tags}")
                tags_dict = None
        
        # Get original filename from uploaded file
        original_filename = file.filename if file.filename else f"unknown.{file_ext}"
        
        # Build metadata dictionary
        ingest_metadata: Dict[str, Any] = {}

        # core business metadata
        ingest_metadata["module"] = module
        if doc_type:
            ingest_metadata["doc_type"] = doc_type
        if year:
            ingest_metadata["year"] = year
        if tags_dict:
            ingest_metadata["tags"] = tags_dict
        if issuer:
            ingest_metadata["issuer"] = issuer
        if regulator_tag:
            ingest_metadata["regulator_tag"] = regulator_tag
        if language:
            ingest_metadata["language"] = language
        if jurisdiction:
            ingest_metadata["jurisdiction"] = jurisdiction
        if version:
            ingest_metadata["version"] = version
        if version_id:
            ingest_metadata["version_id"] = version_id
        if effective_from:
            ingest_metadata["effective_from"] = effective_from
        if valid_from:
            ingest_metadata["valid_from"] = valid_from
        if valid_to:
            ingest_metadata["valid_to"] = valid_to
        if security:
            ingest_metadata["security"] = security
        if pii is not None:
            ingest_metadata["pii"] = pii
        if is_current is not None:
            ingest_metadata["is_current"] = is_current

        # parse JSON-style arrays
        compliance_tags_list = None
        if compliance_tags and compliance_tags.strip():
            try:
                compliance_tags_list = json.loads(compliance_tags)
            except json.JSONDecodeError:
                logger.warning(f"Invalid compliance_tags JSON, skipping: {compliance_tags}")
        if compliance_tags_list:
            ingest_metadata["compliance_tags"] = compliance_tags_list

        additional_modules_list = None
        if additional_modules and additional_modules.strip():
            try:
                additional_modules_list = json.loads(additional_modules)
            except json.JSONDecodeError:
                logger.warning(f"Invalid additional_modules JSON, skipping: {additional_modules}")
        if additional_modules_list:
            additional_modules_list = [m.strip().lower() for m in additional_modules_list]
            ingest_metadata["additional_modules"] = additional_modules_list

        # store original filename
        ingest_metadata["original_filename"] = original_filename
        
        # Save uploaded file to temporary location (preserve extension)
        # NOTE: We do NOT delete this file here - the background worker will delete it
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"rag_upload_async_{os.urandom(8).hex()}.{file_ext}")
        
        logger.info(f"Saving uploaded file to temporary location: {temp_file_path} (original: {original_filename}, type: {file_ext})")
        
        # Read and save file
        content = await file.read()
        file_size = len(content)
        
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved ({file_size} bytes). Creating background job for module '{module}'")
        
        # Create job record in database
        job_id = create_job(
            module=module,
            filename=original_filename,
            file_path=temp_file_path,
            metadata=ingest_metadata,
            file_size=file_size
        )
        
        logger.info(f"Created job {job_id} for file '{original_filename}' (module: {module})")
        
        # Get ingestion pipeline for this module
        pipeline = get_ingestion_pipeline(module)
        
        # Submit to background processing
        task = create_background_task_wrapper(job_id, pipeline, worker_id=f"fastapi-{os.getpid()}")
        background_tasks.add_task(task)
        
        logger.info(f"Submitted job {job_id} to background processing")
        
        # Return immediately with job_id (HTTP 202 Accepted)
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": "pending",
                "message": "Document upload accepted. Processing in background.",
                "filename": original_filename,
                "module": module,
                "check_status_url": f"/ingest/status/{job_id}"
            },
            status_code=202  # 202 Accepted
        )
        
    except HTTPException:
        # Clean up temp file on HTTP exceptions
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up temporary file after error: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
        raise
    except ValueError as e:
        # Clean up temp file on validation errors
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Clean up temp file on unexpected errors
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        logger.error(f"Error during async ingestion setup: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/ingest/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """
    Get the status of an async ingestion job.
    
    Parameters:
    - job_id: UUID of the job (returned from POST /ingest_background)
    
    Returns:
    - job_id: UUID of the job
    - status: pending, processing, completed, failed, cancelled
    - progress_percent: 0-100
    - progress_message: Optional progress message
    - result: Ingestion result (if completed)
    - error_message: Error details (if failed)
    - created_at: Job creation timestamp
    - started_at: Processing start timestamp
    - completed_at: Completion timestamp
    
    Example:
    GET /ingest/status/550e8400-e29b-41d4-a716-446655440000
    """
    try:
        job = get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        # Build response
        response = {
            "job_id": job["job_id"],
            "status": job["status"],
            "module": job["module"],
            "filename": job["filename"],
            "progress_percent": job.get("progress_percent", 0),
            "progress_message": job.get("progress_message"),
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at")
        }
        
        # Include result if completed
        if job["status"] == "completed" and job.get("result"):
            response["result"] = job["result"]
            response["document_id"] = job.get("document_id")
        
        # Include error if failed
        if job["status"] == "failed":
            response["error_message"] = job.get("error_message")
            # Don't expose full traceback in API response (security)
            # response["error_traceback"] = job.get("error_traceback")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/ingest/jobs")
async def list_ingestion_jobs(
    module: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List ingestion jobs with optional filters.
    
    Parameters:
    - module: Filter by module name (optional)
    - status: Filter by status (pending, processing, completed, failed, cancelled) (optional)
    - limit: Maximum number of jobs to return (default: 50, max: 200)
    - offset: Offset for pagination (default: 0)
    
    Returns:
    - jobs: List of job objects
    - count: Number of jobs returned
    - limit: Limit used
    - offset: Offset used
    
    Example:
    GET /ingest/jobs?module=insurance&status=completed&limit=10
    """
    try:
        # Validate module if provided
        if module:
            module = module.strip().lower()
            if not validate_module(module):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid module: {module}. Valid modules: {MODULES}"
                )
        
        # Validate status if provided
        if status:
            valid_statuses = ['pending', 'processing', 'completed', 'failed', 'cancelled']
            if status not in valid_statuses:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid statuses: {valid_statuses}"
                )
        
        # Validate limit
        if limit < 1 or limit > 200:
            raise HTTPException(
                status_code=400,
                detail="limit must be between 1 and 200"
            )
        
        # Get jobs
        jobs = list_jobs(
            module=module,
            status=status,
            limit=limit,
            offset=offset
        )
        
        # Don't expose sensitive fields in list view
        for job in jobs:
            # Remove large fields
            if "error_traceback" in job:
                del job["error_traceback"]
            if "metadata" in job:
                del job["metadata"]  # Can be large
        
        return {
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit,
            "offset": offset,
            "filters": {
                "module": module,
                "status": status
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

