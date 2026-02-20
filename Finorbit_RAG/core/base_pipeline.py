"""
Abstract base class for RAG pipelines.

Defines the common interface for ingestion and retrieval pipelines,
ensuring consistency across all module-specific implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from config import validate_module, PIPELINE_VERSION

logger = logging.getLogger(__name__)


class BaseRAGPipeline(ABC):
    """
    Abstract base class for RAG pipelines.
    
    All ingestion and retrieval pipelines should inherit from this class
    and implement the abstract methods.
    """
    
    def __init__(self, module_name: str, store):
        """
        Initialize the base pipeline.
        
        Args:
            module_name: Module identifier (credit, investment, insurance, retirement, taxation)
            store: PGVectorStore instance for this module
        
        Raises:
            ValueError: If module_name is not recognized
        """
        if not validate_module(module_name):
            raise ValueError(
                f"Invalid module name: {module_name}. "
                f"Must be one of: credit, investment, insurance, retirement, taxation"
            )
        
        self.module_name = module_name
        self.store = store
        self.pipeline_version = PIPELINE_VERSION
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(
            f"Initialized {self.__class__.__name__} for module '{module_name}' "
            f"(pipeline version: {self.pipeline_version})"
        )
    
    @abstractmethod
    def ingest(self, document_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a document into the vector store.
        
        This method should:
        1. Extract text from the document
        2. Chunk the text appropriately
        3. Generate embeddings
        4. Store in the module's vector table
        5. Track metadata (filename, module, document_id, etc.)
        
        Args:
            document_path: Path to the document to ingest
            metadata: Document metadata (filename, year, doc_type, etc.)
        
        Returns:
            Dictionary with ingestion results:
            {
                "status": str,           # "success", "duplicate", or "failed"
                "document_id": int,      # Database document ID (None if failed)
                "module": str,           # Module name
                "chunks_created": int,   # Number of chunks created
                "ocr_applied": bool,     # Whether OCR was used
                "error": str,            # Error message if status == "failed" (optional)
                "metadata": dict         # Additional metadata (optional)
            }
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement ingest()")
    
    @abstractmethod
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Query the vector store and retrieve relevant chunks.
        
        This method should:
        1. Process the query (optionally with HyDE)
        2. Generate query embedding
        3. Retrieve top-k similar chunks from module's table
        4. Optionally rerank results
        5. Return structured results with metadata
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            **kwargs: Additional parameters (filters, rerank options, etc.)
        
        Returns:
            Dictionary containing:
                - chunks: List of retrieved text chunks
                - metadata: Associated metadata for each chunk
                - scores: Similarity scores
                - response: Optional generated response
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement query()")
    
    # ========================================================================
    # Shared utility methods
    # ========================================================================
    
    def validate_metadata(self, metadata: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Validate that metadata contains all required fields.
        
        Args:
            metadata: Metadata dictionary to validate
            required_fields: List of required field names
        
        Returns:
            True if valid, False otherwise
        """
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            self.logger.error(
                f"Metadata validation failed. Missing fields: {missing_fields}"
            )
            return False
        
        return True
    
    def stamp_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add pipeline metadata stamps to document metadata.
        
        Args:
            metadata: Original metadata
        
        Returns:
            Metadata with added pipeline stamps
        """
        stamped = metadata.copy()
        stamped.update({
            "module": self.module_name,
            "pipeline_version": self.pipeline_version,
            "processed_at": datetime.now().isoformat()
        })
        return stamped
    
    def log_operation(self, operation: str, details: Dict[str, Any]):
        """
        Log pipeline operation with consistent formatting.
        
        Args:
            operation: Operation name (e.g., "ingestion", "query")
            details: Operation details to log
        """
        self.logger.info(
            f"[{self.module_name}] {operation}: {details}"
        )
    
    def handle_error(self, error: Exception, context: str) -> None:
        """
        Handle and log errors consistently.
        
        Args:
            error: The exception that occurred
            context: Context description (e.g., "during ingestion")
        """
        self.logger.error(
            f"[{self.module_name}] Error {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about this pipeline instance.
        
        Returns:
            Dictionary with pipeline metadata
        """
        return {
            "pipeline_class": self.__class__.__name__,
            "module_name": self.module_name,
            "pipeline_version": self.pipeline_version,
            "store": str(self.store)
        }
    
    def create_success_response(
        self,
        document_id: int,
        chunks_created: int,
        ocr_applied: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a standardized success response for ingestion.
        
        Args:
            document_id: Database document ID
            chunks_created: Number of chunks created
            ocr_applied: Whether OCR was used
            **kwargs: Additional metadata to include
        
        Returns:
            Standardized success response dictionary
        """
        response = {
            "status": "success",
            "document_id": document_id,
            "module": self.module_name,
            "chunks_created": chunks_created,
            "ocr_applied": ocr_applied
        }
        if kwargs:
            response["metadata"] = kwargs
        return response
    
    def create_duplicate_response(
        self,
        document_id: Optional[int] = None,
        reason: str = "Document already exists"
    ) -> Dict[str, Any]:
        """
        Create a standardized duplicate response for ingestion.
        
        Args:
            document_id: Existing document ID (if available)
            reason: Reason for duplicate detection
        
        Returns:
            Standardized duplicate response dictionary
        """
        return {
            "status": "duplicate",
            "document_id": document_id,
            "module": self.module_name,
            "chunks_created": 0,
            "ocr_applied": False,
            "message": reason
        }
    
    def create_failure_response(
        self,
        error: str,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized failure response for ingestion.
        
        Args:
            error: Error message
            document_id: Document ID if partially created
        
        Returns:
            Standardized failure response dictionary
        """
        return {
            "status": "failed",
            "document_id": document_id,
            "module": self.module_name,
            "chunks_created": 0,
            "ocr_applied": False,
            "error": error
        }
    
    def __repr__(self) -> str:
        """String representation of the pipeline"""
        return (
            f"{self.__class__.__name__}(module='{self.module_name}', "
            f"version='{self.pipeline_version}')"
        )

