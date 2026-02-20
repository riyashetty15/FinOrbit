"""
Text Ingestion Pipeline using LlamaIndex.

Handles multi-format document ingestion with LlamaParse support, using LlamaIndex 
for chunking, embedding, and storage in module-specific vector tables.

Supported formats: PDF, DOCX, PPTX, XLSX, images, and more via LlamaParse.
Falls back to PyMuPDF for PDFs when LlamaParse is unavailable.
"""

import os
import hashlib
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import psycopg2
import psycopg2.extras
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core import BaseRAGPipeline
from config import (
    get_database_config,
    get_embedding_config,
    get_llamaindex_config,
    get_ocr_config,
    get_llamaparse_config,
    DatabaseConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class UnsupportedFileTypeError(Exception):
    """Raised when file type is not supported by any parser"""
    pass


class TextIngestionPipeline(BaseRAGPipeline):
    """
    Text ingestion pipeline using LlamaIndex for document processing.
    
    Handles:
    - Multi-format document parsing (PDF, DOCX, PPTX, images, etc.) via LlamaParse
    - PDF text extraction fallback (PyMuPDF with OCR support)
    - Document registration in documents table
    - Chunking via LlamaIndex SentenceSplitter
    - Embedding generation via HuggingFace models
    - Storage in module-specific PGVectorStore
    """
    
    def __init__(self, module_name: str, store):
        """
        Initialize text ingestion pipeline.
        
        Args:
            module_name: Module identifier
            store: PGVectorStore instance for this module
        """
        super().__init__(module_name, store)
        
        # Load configurations
        self.db_config = get_database_config()
        self.embedding_config = get_embedding_config()
        self.llamaindex_config = get_llamaindex_config()
        self.ocr_config = get_ocr_config()
        self.llamaparse_config = get_llamaparse_config()
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Initialize node parser (chunker)
        self._initialize_node_parser()
        
        # Initialize storage context
        self._initialize_storage_context()
        
        self.logger.info(
            f"TextIngestionPipeline initialized for module '{module_name}' "
            f"with embedding model '{self.embedding_config.model_name}'"
        )
        
        if self.llamaparse_config.enabled:
            self.logger.info(
                f"LlamaParse enabled: result_type='{self.llamaparse_config.result_type}', "
                f"supports {len(self.llamaparse_config.supported_formats)} file formats"
            )
        else:
            self.logger.info("LlamaParse disabled, using PyMuPDF for PDFs only")
    
    def _initialize_embedding_model(self):
        """Initialize HuggingFace embedding model"""
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_config.model_name,
                device=self.embedding_config.device,
                embed_batch_size=self.embedding_config.batch_size,
            )
            self.logger.info(
                f"Initialized embedding model: {self.embedding_config.model_name} "
                f"(dim={self.embedding_config.dimension})"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _initialize_node_parser(self):
        """Initialize LlamaIndex node parser for chunking"""
        # Check if we should use MarkdownNodeParser
        use_markdown = (
            self.llamaindex_config.use_markdown_parser and
            self.llamaparse_config.enabled and
            self.llamaparse_config.result_type == "markdown"
        )
        
        if use_markdown:
            # Use MarkdownNodeParser for markdown documents
            self.node_parser = MarkdownNodeParser(
                chunk_size=self.llamaindex_config.chunk_size,
                chunk_overlap=self.llamaindex_config.chunk_overlap,
                include_metadata=True,  # Extract header info to metadata
                include_prev_next_rel=True,  # Track chunk ordering
            )
            parser_type = "MarkdownNodeParser"
        else:
            # Fallback to SentenceSplitter for non-markdown or when disabled
            self.node_parser = SentenceSplitter(
                chunk_size=self.llamaindex_config.chunk_size,
                chunk_overlap=self.llamaindex_config.chunk_overlap,
            )
            parser_type = "SentenceSplitter"
        
        self.logger.info(
            f"Initialized node parser: {parser_type}, "
            f"chunk_size={self.llamaindex_config.chunk_size}, "
            f"overlap={self.llamaindex_config.chunk_overlap}"
        )
    
    def _initialize_storage_context(self):
        """Initialize storage context with PGVectorStore"""
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.store
        )
        self.logger.info(f"Storage context initialized for module '{self.module_name}'")
    
    def _is_valid_header(self, header_value: str) -> bool:
        """
        Validate if a header value is meaningful before using it.
        
        Rejects:
        - Single characters or very short strings
        - Pure numbers (e.g., "2", "2.1")
        - Only special characters (e.g., "/", "-")
        
        Args:
            header_value: Header string to validate
            
        Returns:
            True if header is valid, False otherwise
        """
        if not header_value or not isinstance(header_value, str):
            return False
        
        header_clean = header_value.strip()
        
        # Too short
        if len(header_clean) < 2:
            return False
        
        # Only special characters (no alphanumeric content)
        if not any(c.isalnum() for c in header_clean):
            return False
        
        # Pure number (e.g., "2", "2.", "3.1")
        try:
            float(header_clean.rstrip('.'))
            return False  # It's a number, not a meaningful header
        except ValueError:
            pass  # Not a number, continue validation
        
        return True
    
    def _extract_valid_headers_from_documents(self, documents: list) -> list:
        """
        Extract valid headers from each document's markdown text.
        
        Returns list of header maps, one per document:
        [{char_pos: [header_1, header_2, ...]}, {char_pos: [header_1, header_2, ...]}, ...]
        
        Args:
            documents: List of Document objects with markdown text
            
        Returns:
            List of dictionaries mapping character positions to header paths
        """
        if not (self.llamaindex_config.use_markdown_parser and 
                self.llamaparse_config.enabled and
                self.llamaparse_config.result_type == "markdown"):
            return []
        
        all_doc_headers = []
        max_levels = self.llamaindex_config.max_header_levels
        
        for doc in documents:
            doc_header_map = {}
            current_headers = [None, None, None]  # Track H1, H2, H3
            char_pos = 0
            
            for line in doc.text.split('\n'):
                line_stripped = line.strip()
                
                # Check if line is a markdown header
                if line_stripped.startswith('#'):
                    # Count header level
                    header_parts = line_stripped.split(' ', 1)
                    if len(header_parts) == 2:
                        level = len(header_parts[0])  # Count of '#'
                        header_text = header_parts[1].strip()
                        
                        # Validate header and check level
                        if (self._is_valid_header(header_text) and 
                            1 <= level <= max_levels):
                            
                            # Update current headers
                            current_headers[level - 1] = header_text
                            # Clear deeper levels (if H2 changes, clear H3)
                            for i in range(level, 3):
                                current_headers[i] = None
                            
                            # Store header path at this position
                            header_path = [h for h in current_headers if h is not None]
                            if header_path:
                                doc_header_map[char_pos] = header_path
                
                # Update character position (include newline)
                char_pos += len(line) + 1
            
            all_doc_headers.append(doc_header_map)
            self.logger.debug(
                f"Extracted {len(doc_header_map)} valid headers from document "
                f"(total positions tracked)"
            )
        
        total_headers = sum(len(hmap) for hmap in all_doc_headers)
        self.logger.info(
            f"Extracted valid headers from {len(documents)} document(s): "
            f"{total_headers} header positions found"
        )
        
        return all_doc_headers
    
    def _find_node_source_document(self, node, documents: list) -> Optional[int]:
        """
        Find which document a node came from by matching metadata or relationships.
        
        Args:
            node: TextNode to find source for
            documents: List of source Document objects
            
        Returns:
            Document index if found, None otherwise
        """
        node_meta = node.metadata or {}
        
        # Method 1: Check for explicit document index (most reliable - set during node creation)
        if '_source_doc_index' in node_meta:
            doc_idx = node_meta['_source_doc_index']
            if isinstance(doc_idx, int) and 0 <= doc_idx < len(documents):
                return doc_idx
        
        # Method 2: Try to match by metadata (filename, document_id, etc.)
        for idx, doc in enumerate(documents):
            doc_meta = doc.metadata or {}
            
            # Match by common metadata fields
            if (node_meta.get('filename') == doc_meta.get('filename') or
                node_meta.get('document_id') == doc_meta.get('document_id') or
                node_meta.get('id') == doc_meta.get('id')):
                return idx
        
        # Method 3: Check node relationships (if available)
        if hasattr(node, 'relationships') and node.relationships:
            from llama_index.core.schema import NodeRelationship
            
            source_rel = node.relationships.get(NodeRelationship.SOURCE)
            if source_rel and hasattr(source_rel, 'node_id'):
                # Try to match source document by comparing metadata
                for idx, doc in enumerate(documents):
                    doc_meta = doc.metadata or {}
                    # Compare key metadata fields
                    if (node_meta.get('filename') == doc_meta.get('filename') or
                        node_meta.get('id') == doc_meta.get('id')):
                        return idx
        
        # Method 4: If we can't determine, return None
        # The calling code should handle this gracefully
        return None
    
    def _assign_headers_to_nodes(self, nodes: list, doc_headers: list, documents: list) -> list:
        """
        Assign headers to nodes based on their start_char_idx position in source documents.
        
        Args:
            nodes: List of nodes from MarkdownNodeParser
            doc_headers: List of header maps (one per document) from _extract_valid_headers_from_documents
            documents: List of source Document objects
            
        Returns:
            List of nodes with header metadata assigned
        """
        if not doc_headers or not nodes:
            return nodes
        
        nodes_assigned = 0
        nodes_skipped = 0
        
        for node in nodes:
            # Find which document this node came from
            doc_idx = self._find_node_source_document(node, documents)
            
            if doc_idx is None or doc_idx >= len(doc_headers):
                nodes_skipped += 1
                continue
            
            header_map = doc_headers[doc_idx]
            if not header_map:
                nodes_skipped += 1
                continue
            
            # Get node's position in the document
            start_idx = getattr(node, 'start_char_idx', None)
            
            if start_idx is None:
                nodes_skipped += 1
                continue
            
            # Find closest header before this position
            header_path = None
            sorted_positions = sorted(header_map.keys(), reverse=True)
            
            for pos in sorted_positions:
                if pos <= start_idx:
                    header_path = header_map[pos]
                    break
            
            # Assign header to node metadata
            if header_path:
                node.metadata = node.metadata or {}
                node.metadata['header_path'] = ' > '.join(header_path)
                node.metadata['header_level'] = len(header_path)
                node.metadata['has_context_header'] = True
                
                # Also store individual header levels for easy access
                for i, header in enumerate(header_path, 1):
                    node.metadata[f'header_{i}'] = header
                
                nodes_assigned += 1
            else:
                # No header found for this position
                node.metadata = node.metadata or {}
                # Clear any invalid headers that MarkdownNodeParser might have set
                # (e.g., header_path: '/' which is invalid)
                node.metadata.pop('header_path', None)
                node.metadata.pop('header_1', None)
                node.metadata.pop('header_2', None)
                node.metadata.pop('header_3', None)
                node.metadata['has_context_header'] = False
                nodes_skipped += 1
        
        self.logger.info(
            f"Assigned headers to {nodes_assigned} nodes "
            f"({nodes_skipped} skipped - no position or no matching document)"
        )
        
        return nodes
    
    def _prepend_headers_to_nodes(self, nodes: list) -> list:
        """
        Prepend section headers to node text based on metadata assigned by _assign_headers_to_nodes().
        
        Headers are already in node metadata as:
        - header_1, header_2, header_3: Individual header levels
        - header_path: Full path like "Header1 > Header2 > Header3"
        
        This method reads from metadata and prepends to node.text.
        
        Args:
            nodes: List of nodes with header metadata already assigned
            
        Returns:
            List of nodes with headers prepended to text
        """
        if not self.llamaindex_config.prepend_headers_to_chunks:
            # Feature disabled, return nodes unchanged
            return nodes
        
        separator = self.llamaindex_config.header_separator
        max_levels = self.llamaindex_config.max_header_levels
        headers_prepended = 0
        
        for node in nodes:
            node_metadata = node.metadata or {}
            
            # Check if node has headers assigned
            has_header = node_metadata.get("has_context_header", False)
            
            if not has_header:
                continue
            
            # Build header string from metadata
            header_parts = []
            
            # Method 1: Try to build from individual header fields first (most reliable)
            # This is what we set in _assign_headers_to_nodes()
            for level in range(1, max_levels + 1):
                header_key = f"header_{level}"
                header_value = node_metadata.get(header_key)
                
                # Validate header value before using it
                if (header_value and 
                    str(header_value).strip() and 
                    str(header_value).strip() != '/' and
                    self._is_valid_header(str(header_value).strip())):
                    header_parts.append(f"{'#' * level} {header_value}")
            
            # Method 2: If no individual headers found, try header_path as fallback
            if not header_parts:
                header_path = node_metadata.get("header_path")
                
                # Validate header_path - skip invalid markers like '/'
                if (header_path and 
                    str(header_path).strip() and 
                    str(header_path).strip() != '/' and
                    self._is_valid_header(str(header_path).split('>')[0].strip() if '>' in str(header_path) else str(header_path).strip())):
                    # header_path format: "Header1 > Header2 > Header3"
                    path_parts = [p.strip() for p in str(header_path).split(">")]
                    for idx, part in enumerate(path_parts[:max_levels], start=1):
                        if part and part.strip() and self._is_valid_header(part):  # Validate each part
                            header_parts.append(f"{'#' * idx} {part}")
            
            # Prepend headers to node text
            if header_parts:
                header_string = "\n".join(header_parts)
                original_text = node.get_content()
                
                # Check if header is already in text (avoid duplication)
                if not original_text.startswith(header_string):
                    # Prepend header to text
                    enhanced_text = f"{header_string}{separator}{original_text}"
                    node.text = enhanced_text
                    headers_prepended += 1
            else:
                # Log warning if has_context_header=True but no header parts found
                self.logger.debug(
                    f"Node has has_context_header=True but no header parts found. "
                    f"Metadata keys: {list(node_metadata.keys())}, "
                    f"header_path: {node_metadata.get('header_path')}, "
                    f"header_1: {node_metadata.get('header_1')}"
                )
        
        self.logger.info(
            f"Prepended headers to {headers_prepended} out of {len(nodes)} nodes"
        )
        
        return nodes
    
    def _detect_file_type(self, file_path: str) -> str:
        """
        Detect file type from file extension.
        
        Args:
            file_path: Path to file
        
        Returns:
            File extension without dot (e.g., 'pdf', 'docx')
        """
        return Path(file_path).suffix.lower().lstrip('.')
    
    def _normalize_document_metadata(
        self,
        raw_metadata: Dict[str, Any],
        checksum: str,
        document_id: int,
        filename: str | None = None,
    ) -> Dict[str, Any]:
        """
        Normalize incoming metadata into a consistent document-level schema.

        This is the doc-level JSON (your schema #1) that we will:
        - store on documents.tags
        - propagate into every chunk's metadata_
        """
        module = raw_metadata.get("module", self.module_name)
        language = raw_metadata.get("language", "EN")
        doc_type = raw_metadata.get("doc_type", "Unknown")
        issuer = raw_metadata.get("issuer", "Unknown")
        security = raw_metadata.get("security", "Public")
        pii = bool(raw_metadata.get("pii", False))
        is_current = raw_metadata.get("is_current")

        # Try to derive year from effective_from / valid_from if not present
        year = raw_metadata.get("year")
        if not year:
            for key in ("effective_from", "valid_from"):
                v = raw_metadata.get(key)
                if v:
                    try:
                        year = str(v)[:4]
                        break
                    except Exception:
                        pass

        # Stable doc id slug
        stable_id = raw_metadata.get("id")
        if not stable_id:
            base = (
                filename
                or raw_metadata.get("original_filename")
                or f"doc_{document_id}"
            )
            stable_id = (
                str(base)
                .lower()
                .replace(" ", "-")
                .replace("/", "-")
            )

        normalized = {
            "id": stable_id,
            "module": module,
            "subtopic": raw_metadata.get("subtopic"),
            "title": raw_metadata.get("title"),
            "language": language,
            "locale": raw_metadata.get("locale", ["en"]),
            "doc_type": doc_type,
            "issuer": issuer,
            "regulator_tag": raw_metadata.get("regulator_tag"),
            "jurisdiction": raw_metadata.get("jurisdiction", "India"),
            "source_uri": raw_metadata.get("source_uri"),
            "original_uri": raw_metadata.get("original_uri"),
            "version": raw_metadata.get("version"),
            "version_id": raw_metadata.get("version_id"),
            "effective_from": raw_metadata.get("effective_from"),
            "valid_from": raw_metadata.get("valid_from"),
            "valid_to": raw_metadata.get("valid_to"),
            "checksum": checksum,
            "license": raw_metadata.get("license", "Open"),
            "pii": pii,
            "security": security,
            "compliance_tags": raw_metadata.get("compliance_tags", []),
            "created_at": raw_metadata.get("created_at"),
            "fetched_at": raw_metadata.get("fetched_at"),
            "validation_status": raw_metadata.get("validation_status", "valid"),
            "validated_at": raw_metadata.get("validated_at"),
            "text_uri": raw_metadata.get("text_uri"),
            # convenience fields already indexed / used elsewhere
            "filename": filename or raw_metadata.get("original_filename") or raw_metadata.get("filename"),
            "year": str(year) if year else None,
            "document_id": document_id,
            "is_current": is_current,
        }

        # Drop explicit Nones
        return {k: v for k, v in normalized.items() if v is not None}

    
    def _get_mime_type(self, file_type: str) -> str:
        """
        Get MIME type for a given file extension.
        
        Args:
            file_type: File extension (without dot)
        
        Returns:
            MIME type string
        """
        mime_types = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "txt": "text/plain",
            "rtf": "application/rtf",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "epub": "application/epub+zip",
            "pages": "application/vnd.apple.pages",
            "key": "application/vnd.apple.keynote",
        }
        return mime_types.get(file_type, "application/octet-stream")
    
    def _extract_with_llamaparse(self, file_path: str) -> tuple[list, dict]:
        """
        Extract content from document using LlamaParse API.
        
        Returns LlamaIndex Document objects directly for better structure preservation.
        
        Args:
            file_path: Path to document file
        
        Returns:
            Tuple of (documents_list, structured_data_dict)
            - documents_list: List of LlamaIndex Document objects (ready for chunking)
            - structured_data_dict: Tables, images, charts metadata for future processing
        
        Raises:
            Exception: If LlamaParse API fails
        """
        try:
            from llama_parse import LlamaParse
            
            self.logger.info(f"Parsing document with LlamaParse: {file_path}")
            
            # Initialize LlamaParse client
            parser = LlamaParse(
                api_key=self.llamaparse_config.api_key,
                result_type=self.llamaparse_config.result_type,
                language=self.llamaparse_config.language,
                verbose=self.llamaparse_config.verbose,
            )
            
            # Parse the document - returns list of LlamaIndex Document objects
            documents = parser.load_data(file_path)
            
            # Calculate total text length for logging/response
            total_text_length = sum(len(doc.text) for doc in documents)
            
            # Extract structured data (tables, images, charts) from document metadata
            # LlamaParse may include table/image data in document metadata
            structured_data = {
                "tables": [],  # Future: Extract table data from metadata
                "images": [],  # Future: Extract image metadata
                "charts": [],  # Future: Extract chart data
                "raw_metadata": [doc.metadata for doc in documents if doc.metadata],
                "num_documents": len(documents)  # Track how many document sections returned
            }
            
            self.logger.info(
                f"LlamaParse returned {len(documents)} document(s) with "
                f"{total_text_length} total characters"
            )
            
            return documents, structured_data
            
        except ImportError:
            self.logger.error("llama-parse package not installed. Run: pip install llama-parse")
            raise
        except Exception as e:
            self.logger.error(f"LlamaParse extraction failed: {e}")
            raise
    
    def _extract_with_pymupdf(self, pdf_path: str) -> tuple[list, dict]:
        """
        Extract text from PDF using PyMuPDF (fallback method).
        
        This is the original extraction logic, kept as a reliable fallback
        for PDFs when LlamaParse is unavailable or fails.
        
        Returns a LlamaIndex Document object for consistency with LlamaParse flow.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (documents_list, empty_structured_data_dict)
            - documents_list: List with single Document object containing extracted text
        """
        try:
            import fitz  # PyMuPDF
            from llama_index.core import Document
            
            self.logger.info(f"Extracting text with PyMuPDF (fallback): {pdf_path}")
            
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Check if we got meaningful text
            if text.strip() and len(text.strip()) > 100:
                self.logger.info(f"PyMuPDF extracted {len(text)} characters from PDF")
            else:
                self.logger.warning("Little/no text extracted with PyMuPDF, may need OCR")
                # TODO: Apply OCR here if configured
            
            # Create a Document object for consistency with LlamaParse
            pymupdf_doc = Document(text=text, metadata={})
            
            return [pymupdf_doc], {}
                
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {e}")
            raise
    
    def _extract_content(self, file_path: str, file_type: str) -> tuple[list, dict, str]:
        """
        Orchestrate content extraction with LlamaParse primary and PyMuPDF fallback.
        
        Returns LlamaIndex Document objects directly for efficient processing.
        
        Extraction strategy:
        1. If LlamaParse enabled and file type supported → use LlamaParse
        2. If LlamaParse fails and file is PDF → fallback to PyMuPDF
        3. If LlamaParse disabled and file is PDF → use PyMuPDF
        4. Otherwise → raise UnsupportedFileTypeError
        
        Args:
            file_path: Path to document file
            file_type: Detected file type (extension)
        
        Returns:
            Tuple of (documents_list, structured_data, parse_engine)
            - documents_list: List of LlamaIndex Document objects (ready for chunking)
            - structured_data: Dict with tables/images/charts metadata
            - parse_engine: "llamaparse" or "pymupdf"
        
        Raises:
            UnsupportedFileTypeError: If file type cannot be processed
        """
        documents = None
        structured_data = {}
        parse_engine = None
        
        # Strategy 1: Try LlamaParse if enabled and file type supported
        if self.llamaparse_config.enabled and file_type in self.llamaparse_config.supported_formats:
            try:
                documents, structured_data = self._extract_with_llamaparse(file_path)
                parse_engine = "llamaparse"
                self.logger.info(f"Successfully parsed with LlamaParse: {file_type}")
                return documents, structured_data, parse_engine
                
            except Exception as e:
                self.logger.warning(
                    f"LlamaParse failed for {file_type}: {e}. "
                    f"Attempting fallback..."
                )
                
                # Strategy 2: Fallback to PyMuPDF if file is PDF
                if file_type == "pdf":
                    self.logger.info("Falling back to PyMuPDF for PDF")
                    documents, structured_data = self._extract_with_pymupdf(file_path)
                    parse_engine = "pymupdf"
                    return documents, structured_data, parse_engine
                else:
                    # No fallback for non-PDF formats
                    raise
        
        # Strategy 3: Use PyMuPDF if LlamaParse disabled and file is PDF
        elif file_type == "pdf":
            documents, structured_data = self._extract_with_pymupdf(file_path)
            parse_engine = "pymupdf"
            return documents, structured_data, parse_engine
        
        # Strategy 4: Unsupported file type
        else:
            supported = list(self.llamaparse_config.supported_formats) if self.llamaparse_config.enabled else ["pdf"]
            raise UnsupportedFileTypeError(
                f"File type '{file_type}' is not supported. "
                f"Supported formats: {', '.join(supported)}"
            )
    
    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum of file.
        
        Args:
            file_path: Path to file
        
        Returns:
            Hex digest of file checksum
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(8192), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _check_duplicate(self, checksum: str) -> Optional[int]:
        """
        Check if document with same checksum exists.
        
        Args:
            checksum: Document checksum
        
        Returns:
            Document ID if duplicate exists, None otherwise
        """
        try:
            with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM documents WHERE checksum = %s",
                    (checksum,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except psycopg2.Error as e:
            self.logger.error(f"Error checking for duplicates: {e}")
            return None
    
    def _register_document(
        self,
        file_path: str,
        checksum: str,
        metadata: Dict[str, Any],
        file_type: str = "pdf"
    ) -> Optional[int]:
        """
        Register document in documents table and document_modules mapping.
        
        Args:
            file_path: Path to document
            checksum: Document checksum
            metadata: Document metadata (may include 'additional_modules' list)
            file_type: Detected file type (extension)
        
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            # Use original filename from metadata if available (from upload), otherwise use basename
            # This ensures we store the actual uploaded filename, not the temporary file name
            original_filename = metadata.pop("original_filename", None)  # Remove from metadata to avoid storing in metadata_
            if original_filename:
                filename = original_filename
                self.logger.debug(f"Using original filename from metadata: {filename}")
            else:
                filename = os.path.basename(file_path)
                self.logger.debug(f"Using basename from file path: {filename}")
            
            file_size = os.path.getsize(file_path)
            
            # Detect MIME type based on file extension
            mime_type = self._get_mime_type(file_type)
            
            # Get additional modules if document should be tagged for multiple modules
            additional_modules = metadata.get("additional_modules", [])
            
            with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                # Insert into documents table
                cursor.execute("""
                    INSERT INTO documents (
                        filename, uri_or_path, mime_type, source_type, checksum,
                        size_bytes, module_primary, tags, ingest_status,
                        pipeline_version, ocr_applied, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    ) RETURNING id
                """, (
                    filename,
                    file_path,
                    mime_type,
                    metadata.get("source_type", "upload"),
                    checksum,
                    file_size,
                    self.module_name,
                    psycopg2.extras.Json(metadata.get("tags", {})),
                    "processing",
                    self.pipeline_version,
                    False,  # OCR flag, will update later
                ))
                document_id = cursor.fetchone()[0]
                
                # Insert into document_modules mapping table (primary module)
                cursor.execute("""
                    INSERT INTO document_modules (document_id, module)
                    VALUES (%s, %s)
                """, (document_id, self.module_name))
                
                # Insert additional module mappings if specified
                if additional_modules:
                    for module in additional_modules:
                        if module != self.module_name:  # Avoid duplicate
                            cursor.execute("""
                                INSERT INTO document_modules (document_id, module)
                                VALUES (%s, %s)
                                ON CONFLICT (document_id, module) DO NOTHING
                            """, (document_id, module))
                    
                    self.logger.info(
                        f"Registered document ID {document_id}: {filename} "
                        f"(modules: {self.module_name}, {', '.join(additional_modules)})"
                    )
                else:
                    self.logger.info(
                        f"Registered document ID {document_id}: {filename} "
                        f"(module: {self.module_name})"
                    )
                
                conn.commit()
                return document_id
                
        except psycopg2.Error as e:
            self.logger.error(f"Error registering document: {e}")
            return None
    
    def _update_document_status(
        self,
        document_id: int,
        status: str,
        error: Optional[str] = None,
        ocr_applied: bool = False
    ):
        """
        Update document status in documents table.
        
        Args:
            document_id: Document ID
            status: New status (complete, failed, etc.)
            error: Error message if failed
            ocr_applied: Whether OCR was used
        """
        try:
            with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE documents
                    SET ingest_status = %s,
                        ingest_error = %s,
                        ocr_applied = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (status, error, ocr_applied, document_id))
                conn.commit()
                
                self.logger.info(f"Updated document {document_id} status to '{status}'")
                
        except psycopg2.Error as e:
            self.logger.error(f"Error updating document status: {e}")
    
    def _create_llamaindex_document(
        self,
        text: str,
        document_id: int,
        metadata: Dict[str, Any]
    ) -> Document:
        """
        Create LlamaIndex Document object.
        
        Args:
            text: Document text
            document_id: Database document ID
            metadata: Document metadata
        
        Returns:
            LlamaIndex Document object
        """
        # Enhance metadata with our tracking info
        # This metadata will be stored in metadata_ column by LlamaIndex's VectorStoreIndex
        # All business metadata (doc_type, year, filename, etc.) should be included here
        doc_metadata = {
            "document_id": document_id,
            "module": self.module_name,
            "pipeline_version": self.pipeline_version,
            **metadata  # Includes doc_type, year, filename, tags, etc. from user input
        }
        
        return Document(
            text=text,
            metadata=doc_metadata,  # LlamaIndex stores this in metadata_ column
            id_=f"doc_{document_id}"
        )
    
    def _update_chunks_document_id(
        self,
        document_id: int,
        nodes: list
    ) -> int:
        """
        Update chunks in database to populate document_id column after LlamaIndex stores them.
        
        Uses node_id from the nodes list to match chunks, not metadata.
        This is more reliable than parsing metadata.
        
        Args:
            document_id: Database document ID (from documents table)
            nodes: List of nodes that were just stored by LlamaIndex
        
        Returns:
            Number of chunks updated
        """
        from config import get_table_name_for_module
        from psycopg2 import sql
        
        table_name = get_table_name_for_module(self.module_name)
        chunk_table = table_name  # Already includes "data_" prefix if configured
        
        try:
            # Extract node_ids from the nodes list
            node_ids = []
            for node in nodes:
                if hasattr(node, 'node_id') and node.node_id:
                    node_ids.append(node.node_id)
            
            if not node_ids:
                self.logger.warning(f"No node_ids found in nodes list for document_id={document_id}")
                return 0
            
            with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                # Update chunks where node_id matches the nodes we just inserted
                update_query = sql.SQL("""
                    UPDATE {}
                    SET document_id = %s
                    WHERE node_id = ANY(%s)
                      AND document_id IS NULL
                """).format(sql.Identifier(chunk_table))
                
                cursor.execute(update_query, (document_id, node_ids))
                chunks_updated = cursor.rowcount
                
                if chunks_updated == 0:
                    self.logger.warning(
                        f"No chunks updated for document_id={document_id}. "
                        f"Tried to match {len(node_ids)} node_ids: {node_ids[:5]}..."
                    )
                else:
                    self.logger.info(
                        f"Updated {chunks_updated} chunks with document_id={document_id} "
                        f"using {len(node_ids)} node_ids"
                    )
                
                # Update chunk_index based on node order
                if chunks_updated > 0:
                    # Get all chunks for this document that were just updated
                    get_chunks_query = sql.SQL("""
                        SELECT id, node_id
                        FROM {}
                        WHERE document_id = %s
                        ORDER BY id
                    """).format(sql.Identifier(chunk_table))
                    
                    cursor.execute(get_chunks_query, (document_id,))
                    chunk_rows = cursor.fetchall()
                    
                    # Create a mapping of node_id to chunk_index
                    node_id_to_index = {}
                    for idx, node in enumerate(nodes):
                        if hasattr(node, 'node_id') and node.node_id:
                            node_id_to_index[node.node_id] = idx
                    
                    # Update chunk_index for each chunk
                    updated_indices = 0
                    for chunk_id, node_id in chunk_rows:
                        if node_id and node_id in node_id_to_index:
                            chunk_index = node_id_to_index[node_id]
                            update_index_query = sql.SQL("""
                                UPDATE {}
                                SET chunk_index = %s
                                WHERE id = %s
                            """).format(sql.Identifier(chunk_table))
                            cursor.execute(update_index_query, (chunk_index, chunk_id))
                            updated_indices += 1
                    
                    if updated_indices > 0:
                        self.logger.debug(f"Updated chunk_index for {updated_indices} chunks")
                
                conn.commit()
                return chunks_updated
                
        except psycopg2.Error as e:
            self.logger.error(f"Database error updating chunks document_id: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error updating chunks document_id: {e}")
            raise
    
    def _insert_nodes_to_db(
        self,
        nodes: list,
        document_id: int,
        table_name: str
    ) -> int:
        """
        Manually insert embedded nodes into our custom schema using batch insert.
        
        Args:
            nodes: List of LlamaIndex nodes with embeddings
            document_id: Database document ID
            table_name: Target chunk table (e.g., 'credit_chunks')
        
        Returns:
            Number of chunks successfully inserted
        """
        if not nodes:
            return 0
        
        try:
            # Get database connection parameters
            conn_params = {
                "host": self.db_config.host,
                "port": self.db_config.port,
                "database": self.db_config.database,
                "user": self.db_config.user,
                "password": self.db_config.password,
            }
            
            # Prepare batch data
            batch_data = []
            skipped_nodes = 0
            
            for idx, node in enumerate(nodes):
                # Extract node data
                text_content = node.get_content()
                embedding = node.embedding
                
                if embedding is None:
                    self.logger.warning(f"Node {idx} has no embedding, skipping")
                    skipped_nodes += 1
                    continue
                
                # Extract metadata
                node_metadata = node.metadata or {}
                
                # Calculate chunk hash for deduplication
                chunk_hash = hashlib.sha256(text_content.encode('utf-8')).hexdigest()
                
                # Calculate stats
                word_count = len(text_content.split())
                char_count = len(text_content)
                
                # Get page number from metadata if available
                page_number = node_metadata.get("page_number")
                
                # Prepare metadata JSONB (store everything from node metadata)
                # Ensure traceability fields are present
                metadata_json = {
                    **node_metadata,
                    "node_id": node.node_id,
                    "llama_index_id": node.id_,
                    "document_id": document_id,
                    "chunk_index": idx,
                }
                # Add filename if available in provided document-level metadata
                # Expect that upstream ingestion calls pass filename in pipeline metadata
                if "filename" not in metadata_json:
                    # Attempt to infer filename from node metadata or path fields
                    inferred_filename = node_metadata.get("filename") or node_metadata.get("source_filename")
                    if inferred_filename:
                        metadata_json["filename"] = inferred_filename
                
                # Add to batch
                # Include metadata_ (LlamaIndex compatible) and node_id (separate column)
                batch_data.append((
                    document_id,
                    idx,
                    text_content,
                    page_number,
                    embedding,  # psycopg2 with pgvector handles list conversion
                    self.embedding_config.model_name,
                    "1.0",  # embedding model version
                    psycopg2.extras.Json(metadata_json),  # metadata (our custom)
                    psycopg2.extras.Json(metadata_json),  # metadata_ (LlamaIndex compatible, same data)
                    node.node_id,  # node_id (separate column for LlamaIndex)
                    self.module_name,
                    word_count,
                    char_count,
                    chunk_hash,
                    self.pipeline_version,
                ))
            
            if not batch_data:
                self.logger.warning("No valid nodes to insert")
                return 0
            
            # Batch insert into database
            with psycopg2.connect(**conn_params) as conn, conn.cursor() as cursor:
                insert_query = f"""
                    INSERT INTO {table_name} (
                        document_id, chunk_index, text, page_number,
                        embedding, embedding_model, embedding_model_version,
                        embedding_created_at, metadata, metadata_, node_id, module,
                        word_count, char_count, chunk_hash,
                        pipeline_version, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s,
                        %s, %s, %s, %s, NOW(), NOW()
                    )
                    ON CONFLICT (document_id, chunk_hash) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_model_version = EXCLUDED.embedding_model_version,
                        embedding_created_at = NOW(),
                        metadata = EXCLUDED.metadata_,
                        metadata_ = EXCLUDED.metadata_,
                        updated_at = NOW()
                    RETURNING id
                """
                
                # Use execute_batch for efficient batch insert
                psycopg2.extras.execute_batch(
                    cursor,
                    insert_query,
                    batch_data,
                    page_size=100  # Insert 100 rows at a time
                )
                
                # Get count of inserted rows
                chunks_inserted = len(batch_data)
                
                conn.commit()
                
                self.logger.info(
                    f"Batch inserted {chunks_inserted} chunks into {table_name} "
                    f"({skipped_nodes} skipped)"
                )
        
        except psycopg2.Error as e:
            self.logger.error(f"Database error inserting chunks: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error inserting chunks: {e}")
            raise
        
        return chunks_inserted
    
    def ingest(self, document_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a document into the vector store.
        
        Args:
            document_path: Path to the document to ingest
            metadata: Document metadata (year, doc_type, tags, etc.)
        
        Returns:
            Dictionary with ingestion results:
            {
                "status": "success" | "duplicate" | "failed",
                "document_id": int,
                "module": str,
                "chunks_created": int,
                "ocr_applied": bool,
                "error": str (if failed)
            }
        """
        document_id = None
        ocr_applied = False
        original_filename = None  # Store original filename for response
        
        try:
            # Validate file exists
            if not os.path.exists(document_path):
                return self.create_failure_response(
                    error=f"File not found: {document_path}"
                )
            
            # Detect file type
            file_type = self._detect_file_type(document_path)
            self.logger.info(f"Detected file type: {file_type}")
            
            # Validate file type is supported
            if not self.llamaparse_config.enabled and file_type != "pdf":
                return self.create_failure_response(
                    error=f"File type '{file_type}' not supported. LlamaParse is disabled and only PDF is supported via PyMuPDF."
                )
            
            # Extract original filename from metadata if available (before it gets popped)
            original_filename = metadata.get("original_filename")
            
            self.log_operation("ingestion_start", {
                "document_path": document_path,
                "file_type": file_type,
                "metadata": metadata
            })
            
            # Step 1: Calculate checksum and check for duplicates
            checksum = self._calculate_checksum(document_path)
            duplicate_id = self._check_duplicate(checksum)
            
            if duplicate_id:
                self.logger.info(f"Duplicate document detected (ID: {duplicate_id})")
                return self.create_duplicate_response(
                    document_id=duplicate_id,
                    reason=f"Document with checksum {checksum[:8]}... already exists"
                )
            
            # Step 2: Register document in documents table
            # This will pop original_filename from metadata, so we need to store it first
            document_id = self._register_document(document_path, checksum, metadata, file_type)
            if not document_id:
                return self.create_failure_response(
                    error="Failed to register document in database"
                )
            
            # Build normalized document-level metadata (schema #1)
            normalized_doc_meta = self._normalize_document_metadata(
                raw_metadata=metadata,
                checksum=checksum,
                document_id=document_id,
                filename=original_filename or os.path.basename(document_path),
            )

            # Optionally persist normalized metadata to documents.tags for later inspection
            try:
                with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE documents SET tags = %s, updated_at = NOW() WHERE id = %s",
                        (psycopg2.extras.Json(normalized_doc_meta), document_id),
                    )
                    conn.commit()
            except Exception as e:
                self.logger.warning(f"Failed to update documents.tags with normalized metadata: {e}")

            # If original_filename wasn't in metadata, get it from database (fallback for direct file ingestion)
            if not original_filename:
                # Query database to get the actual filename that was stored
                try:
                    with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                        cursor.execute("SELECT filename FROM documents WHERE id = %s", (document_id,))
                        result = cursor.fetchone()
                        if result:
                            original_filename = result[0]
                except Exception as e:
                    self.logger.warning(f"Could not retrieve filename from database: {e}, using basename")
                    original_filename = os.path.basename(document_path)
            
            # Step 3: Extract content using LlamaParse or PyMuPDF
            # Returns list of LlamaIndex Document objects directly
            documents, structured_data, parse_engine = self._extract_content(document_path, file_type)
            
            if not documents or len(documents) == 0:
                self._update_document_status(document_id, "failed", "No documents returned from parser")
                return self.create_failure_response(
                    error=f"No documents could be extracted from {file_type.upper()} file",
                    document_id=document_id
                )
            
            # Validate that documents have text
            total_text_length = sum(len(doc.text) for doc in documents)
            if total_text_length == 0:
                self._update_document_status(document_id, "failed", "No text extracted")
                return self.create_failure_response(
                    error=f"No text could be extracted from {file_type.upper()} file",
                    document_id=document_id
                )
            
            self.logger.info(
                f"Extracted {len(documents)} document(s) with {total_text_length} characters "
                f"using {parse_engine}, structured_data keys: {list(structured_data.keys())}"
            )
            
            # Step 3.5: Extract valid headers from documents (before chunking)
            # This builds a mapping of character positions to headers for later assignment
            doc_headers = self._extract_valid_headers_from_documents(documents)
            
            # Step 4: Enhance metadata for all documents
            # Merge existing metadata with our tracking info
            enhanced_metadata = {
                **normalized_doc_meta,            # doc-level schema (id, module, issuer, etc.)
                **metadata,                       # raw extra metadata
                "document_id": document_id,
                "module": self.module_name,
                "pipeline_version": self.pipeline_version,
                "parse_engine": parse_engine,
                "file_type": file_type,
                "has_tables": bool(structured_data.get("tables")),
                "has_images": bool(structured_data.get("images")),
                "structured_data": structured_data,  # For future table/image processing
            }
            
            # Update metadata for all documents returned by parser
            for doc in documents:
                # Merge parser's metadata with our enhanced metadata
                # Parser metadata takes precedence for document-specific info (e.g., page numbers)
                doc.metadata = {
                    **doc.metadata,# Parser's metadata (page numbers, etc.)
                    **enhanced_metadata    
                }
            
            # Step 5: Parse documents into nodes (chunks) directly
            # Process documents individually to track which document each node came from
            all_nodes = []
            for doc_idx, doc in enumerate(documents):
                # Process each document separately
                doc_nodes = self.node_parser.get_nodes_from_documents([doc])
                
                # Tag each node with its source document index for header assignment
                for node in doc_nodes:
                    node.metadata = node.metadata or {}
                    node.metadata['_source_doc_index'] = doc_idx
                
                all_nodes.extend(doc_nodes)
            
            nodes = all_nodes
            
            # Step 5.5: Assign headers to nodes based on their position in documents
            if self.llamaindex_config.prepend_headers_to_chunks and doc_headers:
                nodes = self._assign_headers_to_nodes(nodes, doc_headers, documents)
            
            # Step 5.6: Prepend headers to node text if enabled
            if self.llamaindex_config.prepend_headers_to_chunks:
                nodes = self._prepend_headers_to_nodes(nodes)
            
            chunks_created = len(nodes)
            
            self.logger.info(f"Created {chunks_created} chunks from document")
            
            # ---------- Enrich nodes with chunk-level metadata ----------
            parent_id = normalized_doc_meta.get("id") or f"doc_{document_id}"

            base_chunk_meta = {
                # doc-level fields propagated to each chunk
                "parent_id": parent_id,
                "module": normalized_doc_meta.get("module"),
                "language": normalized_doc_meta.get("language"),
                "issuer": normalized_doc_meta.get("issuer"),
                "doc_type": normalized_doc_meta.get("doc_type"),
                "regulator_tag": normalized_doc_meta.get("regulator_tag"),
                "compliance_tags": normalized_doc_meta.get("compliance_tags", []),
                "version_id": normalized_doc_meta.get("version_id"),
                "effective_from": normalized_doc_meta.get("effective_from"),
                "valid_from": normalized_doc_meta.get("valid_from"),
                "valid_to": normalized_doc_meta.get("valid_to"),
                "is_current": normalized_doc_meta.get("is_current"),
                "validation_status": normalized_doc_meta.get("validation_status", "valid"),
                "pii": normalized_doc_meta.get("pii"),
                "security": normalized_doc_meta.get("security"),
                "document_id": document_id,
                "filename": normalized_doc_meta.get("filename"),
                "year": normalized_doc_meta.get("year"),
            }

            for idx, node in enumerate(nodes):
                text_content = node.get_content()
                approx_tokens = max(1, int(len(text_content.split()) / 0.75))  # rough estimate

                chunk_meta = {
                    **(node.metadata or {}),
                    **base_chunk_meta,
                    "chunk_id": f"{parent_id}_chunk_{idx}",
                    "chunk_index": idx,
                    "chunk_tokens": approx_tokens,
                    "embedding_version": self.embedding_config.model_name,
                    # you can add section_title / chunk_summary later via LLM post-process
                }

                node.metadata = chunk_meta

            # ---------- Ingestion-stage metadata filtering (stage 1) ----------
            # filtered_nodes = []
            # skipped_for_policy = 0
            # for node in nodes:
            #     m = node.metadata or {}
            #     # Example policy: only index public, non-PII nodes
            #     if m.get("security", "Public") != "Public":
            #         skipped_for_policy += 1
            #         continue
            #     if m.get("pii", False):
            #         skipped_for_policy += 1
            #         continue
            #     filtered_nodes.append(node)

            # if skipped_for_policy:
            #     self.logger.info(
            #         f"Skipped {skipped_for_policy} chunks due to ingestion metadata policy (security/pii)"
            #     )

            # nodes = filtered_nodes
            chunks_created = len(nodes)
            self.logger.info(f"{chunks_created} chunks remain after ingestion-stage metadata filtering")

            # Step 6: Generate embeddings using VectorStoreIndex with storage context
            # LlamaIndex automatically stores nodes in metadata_ column (not custom metadata column)
            self.logger.info(f"Generating embeddings for {chunks_created} chunks...")
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,  # Use storage context - LlamaIndex will handle DB insertion
                embed_model=self.embed_model,
                show_progress=True,
            )
            
            # LlamaIndex automatically stores nodes in PGVectorStore via storage_context
            # All node metadata (from Document.metadata) is stored in metadata_ column
            chunks_inserted = len(nodes)
            self.logger.info(f"Successfully stored {chunks_inserted} chunks via LlamaIndex storage context (metadata in metadata_ column)")
            
            # Step 7: Update chunks to populate document_id column using node_id matching
            # LlamaIndex doesn't populate custom fields, so we update them after storage
            # We use node_id from the nodes list to match chunks (not metadata)
            chunks_updated = self._update_chunks_document_id(document_id, nodes)
            self.logger.info(f"Updated {chunks_updated} chunks with document_id={document_id}")
            
            # Step 8: Manually insert nodes into our custom schema (COMMENTED OUT - using LlamaIndex default storage)
            # from config import get_table_name_for_module
            # table_name = get_table_name_for_module(self.module_name)
            # 
            # embedded_nodes = []
            # for node_id in index.index_struct.nodes_dict.keys():
            #     node = index.docstore.get_node(node_id)
            #     embedding = index.vector_store.data.embedding_dict.get(node_id)
            #     if embedding is not None:
            #         node.embedding = embedding
            #         embedded_nodes.append(node)
            # 
            # self.logger.info(f"Extracted {len(embedded_nodes)} embedded nodes from index")
            # 
            # chunks_inserted = self._insert_nodes_to_db(
            #     nodes=embedded_nodes,
            #     document_id=document_id,
            #     table_name=table_name
            # )
            # 
            # self.logger.info(f"Successfully inserted {chunks_inserted} chunks into {table_name}")
            
            # Step 9: Update document status to complete
            self._update_document_status(document_id, "complete", ocr_applied=ocr_applied)
            
            # Return success response with enhanced info
            return self.create_success_response(
                document_id=document_id,
                chunks_created=chunks_inserted,  # Use actual inserted count
                ocr_applied=ocr_applied,
                filename=original_filename or os.path.basename(document_path),  # Use original filename if available
                text_length=total_text_length,  # Total text from all documents
                parse_engine=parse_engine,  # "llamaparse" or "pymupdf"
                file_type=file_type,  # File extension
                has_tables=bool(structured_data.get("tables")),
                has_images=bool(structured_data.get("images")),
                num_source_documents=len(documents)  # Number of Document objects returned by parser
            )
            
        except UnsupportedFileTypeError as e:
            # Handle unsupported file type gracefully
            self.logger.error(f"Unsupported file type: {e}")
            
            if document_id:
                self._update_document_status(document_id, "failed", str(e))
            
            return self.create_failure_response(
                error=str(e),
                document_id=document_id
            )
            
        except Exception as e:
            self.handle_error(e, "during ingestion")
            
            # Update document status if we have document_id
            if document_id:
                self._update_document_status(document_id, "failed", str(e))
            
            return self.create_failure_response(
                error=str(e),
                document_id=document_id
            )
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Query method - not implemented for ingestion pipeline.
        
        Use retrieval_pipeline for querying.
        """
        raise NotImplementedError(
            "TextIngestionPipeline is for ingestion only. "
            "Use RetrievalPipeline for querying."
        )
