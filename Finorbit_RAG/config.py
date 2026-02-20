"""
Configuration module for modular RAG pipeline with LlamaIndex.

Centralizes database credentials, module definitions, embedding models,
and LlamaIndex settings.

Note: Module definitions are imported from shared_config.py for consistency
across Finorbit_LLM and Finorbit_RAG.
"""

import os
import logging
from typing import Dict, List
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Import Shared Configuration
# ============================================================================
# Add parent directory to path to import shared_config
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from shared_config import FINORBIT_MODULES, FINORBIT_MODULE_TABLES
    MODULES = FINORBIT_MODULES
    MODULE_TABLES = FINORBIT_MODULE_TABLES
    logger.info("[OK] Loaded module definitions from shared_config.py")
except ImportError as e:
    logger.warning(f"[WARNING]  Could not import from shared_config.py: {e}. Using local definitions.")
    # Fallback to local definitions if shared_config is not available
    MODULES: List[str] = [
        "credit",
        "investment",
        "insurance",
        "retirement",
        "taxation"
    ]

    MODULE_TABLES: Dict[str, str] = {
        "credit": "data_credit_chunks",
        "investment": "data_investment_chunks",
        "insurance": "data_insurance_chunks",
        "retirement": "data_retirement_chunks",
        "taxation": "data_tax_chunks"
    }

# Validate module consistency
assert set(MODULES) == set(MODULE_TABLES.keys()), "MODULES and MODULE_TABLES must match"


# ============================================================================
# Database Configuration
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    sslmode: str = "prefer"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for psycopg2/pgvector connections"""
        return {
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "port": self.port,
            "sslmode": self.sslmode
        }
    
    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string with URL-encoded password"""
        # URL-encode password to handle special characters like @, :, /, etc.
        encoded_password = quote_plus(self.password)
        encoded_user = quote_plus(self.user)
        return (
            f"postgresql://{encoded_user}:{encoded_password}@"
            f"{self.host}:{self.port}/{self.database}?sslmode={self.sslmode}"
        )
    
    def get_url_encoded_password(self) -> str:
        """Get URL-encoded password for use in connection strings"""
        return quote_plus(self.password)


def get_database_config() -> DatabaseConfig:
    """Load database configuration from environment variables"""
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "financial_rag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        port=int(os.getenv("DB_PORT", "5432")),
        sslmode=os.getenv("DB_SSLMODE", "prefer")
    )


# ============================================================================
# Embedding Model Configuration
# ============================================================================

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "philschmid/bge-base-financial-matryoshka"
    dimension: int = 768  # Default for fin-mpnet-base (matches schema)
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 32
    
    def __post_init__(self):
        """
        Validate dimension matches model or update based on model_name.
        
        If model_name is recognized, dimension is overridden to match the model's
        actual output dimension. This prevents dimension mismatches.
        
        Note: If EMBEDDING_DIMENSION env var is set but differs from the model's
        actual dimension, it will be overridden and a warning will be logged.
        """
        # Store original dimension before override for logging
        original_dimension = self.dimension
        
        # Common model dimensions
        model_dims = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "mukaj/fin-mpnet-base": 768,
            "FinanceMTEB/FinE5": 768,
            "philschmid/bge-base-financial-matryoshka": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
        }
        
        if self.model_name in model_dims:
            model_dim = model_dims[self.model_name]
            if original_dimension != model_dim:
                logger.warning(
                    f"Embedding dimension overridden: {original_dimension} -> {model_dim} "
                    f"(model '{self.model_name}' outputs {model_dim} dimensions)"
                )
            self.dimension = model_dim


def get_embedding_config() -> EmbeddingConfig:
    """Load embedding configuration from environment variables"""
    return EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "mukaj/fin-mpnet-base"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )


# ============================================================================
# LlamaIndex Configuration
# ============================================================================

@dataclass
class LlamaIndexConfig:
    """LlamaIndex-specific settings"""
    chunk_size: int = 512  # Characters/tokens per chunk
    chunk_overlap: int = 50  # Overlap between chunks
    top_k: int = 5  # Default number of chunks to retrieve
    similarity_top_k: int = 5  # LlamaIndex retrieval parameter
    embedding_dim: int = 768  # Should match EmbeddingConfig.dimension
    
    # Query engine settings
    response_mode: str = "compact"  # "default", "compact", "tree_summarize", "refine"
    streaming: bool = False
    
    # LLM settings (if using LlamaIndex LLM integration)
    llm_provider: str = "openai"  # "openai" or "gemini"
    llm_model: str = None  # e.g., "gpt-3.5-turbo" or "models/gemini-pro"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    google_api_key: str = None  # For Gemini
    
    # Retrieval enhancement settings
    hyde_enabled: bool = False  # Hypothetical Document Embeddings
    rerank_enabled: bool = False  # Reranking for improved relevance
    rerank_model: str = None  # Optional reranker model name
    
    # Header/contextual chunk settings
    use_markdown_parser: bool = True  # Use MarkdownNodeParser for markdown docs
    prepend_headers_to_chunks: bool = True  # Prepend section headers to chunk text
    header_separator: str = "\n\n"  # Separator between header and content
    max_header_levels: int = 3  # Maximum header depth to include (H1, H2, H3)


def get_llamaindex_config() -> LlamaIndexConfig:
    """Load LlamaIndex configuration from environment variables"""
    embedding_cfg = get_embedding_config()
    
    return LlamaIndexConfig(
        chunk_size=int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("LLAMAINDEX_CHUNK_OVERLAP", "50")),
        top_k=int(os.getenv("LLAMAINDEX_TOP_K", "5")),
        similarity_top_k=int(os.getenv("LLAMAINDEX_SIMILARITY_TOP_K", "5")),
        embedding_dim=embedding_cfg.dimension,
        response_mode=os.getenv("LLAMAINDEX_RESPONSE_MODE", "compact"),
        streaming=os.getenv("LLAMAINDEX_STREAMING", "false").lower() == "true",
        llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
        llm_model=os.getenv("LLAMAINDEX_LLM_MODEL"),
        llm_temperature=float(os.getenv("LLAMAINDEX_LLM_TEMPERATURE", "0.1")),
        llm_max_tokens=int(os.getenv("LLAMAINDEX_LLM_MAX_TOKENS", "512")),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        hyde_enabled=os.getenv("HYDE_ENABLED", "false").lower() == "true",
        rerank_enabled=os.getenv("RERANK_ENABLED", "false").lower() == "true",
        rerank_model=os.getenv("RERANK_MODEL"),
        use_markdown_parser=os.getenv("USE_MARKDOWN_PARSER", "true").lower() == "true",
        prepend_headers_to_chunks=os.getenv("PREPEND_HEADERS_TO_CHUNKS", "true").lower() == "true",
        header_separator=os.getenv("HEADER_SEPARATOR", "\n\n"),
        max_header_levels=int(os.getenv("MAX_HEADER_LEVELS", "3"))
    )


# ============================================================================
# OCR Configuration (reused from existing system)
# ============================================================================

@dataclass
class OCRConfig:
    """OCR processing configuration"""
    use_ocr: bool = True
    tesseract_cmd: str = None
    tessdata_dir: str = None
    ocr_language: str = "eng"
    dpi: int = 300
    use_ocrmypdf: bool = True
    ocr_timeout: int = 300


def get_ocr_config() -> OCRConfig:
    """Load OCR configuration from environment variables"""
    return OCRConfig(
        use_ocr=os.getenv("OCR_ENABLED", "true").lower() == "true",
        tesseract_cmd=os.getenv("TESSERACT_CMD"),
        tessdata_dir=os.getenv("TESSDATA_DIR"),
        ocr_language=os.getenv("OCR_LANGUAGE", "eng"),
        dpi=int(os.getenv("OCR_DPI", "300")),
        use_ocrmypdf=os.getenv("USE_OCRMYPDF", "true").lower() == "true",
        ocr_timeout=int(os.getenv("OCR_TIMEOUT", "300"))
    )


# ============================================================================
# LlamaParse Configuration
# ============================================================================

@dataclass
class LlamaParseConfig:
    """LlamaParse document parsing configuration"""
    api_key: str = None
    enabled: bool = False
    result_type: str = "markdown"  # "text", "markdown", or "json"
    extract_charts: bool = True
    save_images: bool = True
    timeout: int = 300  # seconds
    language: str = "en"
    verbose: bool = False
    
    # Supported file formats by LlamaParse
    supported_formats: tuple = (
        "pdf", "docx", "pptx", "xlsx", "txt", "rtf",
        "jpg", "jpeg", "png", "gif", "bmp", "tiff",
        "doc", "ppt", "xls", "epub", "pages", "key"
    )
    
    def __post_init__(self):
        """Auto-enable if API key is provided"""
        if self.api_key and self.api_key.strip():
            self.enabled = True
        else:
            self.enabled = False


def get_llamaparse_config() -> LlamaParseConfig:
    """Load LlamaParse configuration from environment variables"""
    api_key = os.getenv("LLAMAPARSE_API_KEY")
    
    config = LlamaParseConfig(
        api_key=api_key,
        result_type=os.getenv("LLAMAPARSE_RESULT_TYPE", "markdown"),
        extract_charts=os.getenv("LLAMAPARSE_EXTRACT_CHARTS", "true").lower() == "true",
        save_images=os.getenv("LLAMAPARSE_SAVE_IMAGES", "true").lower() == "true",
        timeout=int(os.getenv("LLAMAPARSE_TIMEOUT", "300")),
        language=os.getenv("LLAMAPARSE_LANGUAGE", "en"),
        verbose=os.getenv("LLAMAPARSE_VERBOSE", "false").lower() == "true"
    )
    
    if config.enabled:
        logger.info("LlamaParse enabled with result_type='%s'", config.result_type)
    else:
        logger.info("LlamaParse disabled (no API key found)")
    
    return config


# ============================================================================
# Pipeline Version
# ============================================================================

PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "1.0.0")


# ============================================================================
# Convenience Functions
# ============================================================================

def get_table_name_for_module(module: str) -> str:
    """Get the chunk table name for a given module"""
    if module not in MODULE_TABLES:
        raise ValueError(f"Unknown module: {module}. Valid modules: {MODULES}")
    return MODULE_TABLES[module]


def validate_module(module: str) -> bool:
    """Validate that a module name is recognized"""
    return module in MODULES


# ============================================================================
# Schema Validation
# ============================================================================

def validate_schema_dimension(embedding_dim: int, db_config: DatabaseConfig = None) -> bool:
    """
    Validate that the database schema vector dimension matches the embedding dimension.
    
    This should be called before ingestion to fail fast if there's a mismatch.
    
    Args:
        embedding_dim: The embedding dimension from EmbeddingConfig
        db_config: Database configuration (if None, will load from env)
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If schema dimension doesn't match embedding_dim
    """
    import psycopg2
    
    if db_config is None:
        db_config = get_database_config()
    
    try:
        conn = psycopg2.connect(**db_config.to_dict())
        cursor = conn.cursor()
        
        # Check the first chunk table's embedding column dimension
        # All tables should have the same dimension (enforced by schema)
        test_table = MODULE_TABLES[MODULES[0]]  # Use first module table as sample
        
        cursor.execute("""
            SELECT 
                pg_catalog.format_type(a.atttypid, a.atttypmod) as column_type
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
            JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public'
              AND c.relname = %s
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
        """, (test_table,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise ValueError(
                f"Could not find embedding column in table '{test_table}'. "
                "Ensure schema has been applied."
            )
        
        # Parse vector dimension from type string (e.g., "vector(768)")
        type_str = result[0]
        if "vector" not in type_str.lower():
            raise ValueError(f"Embedding column is not a vector type: {type_str}")
        
        # Extract dimension from vector(N) format
        import re
        match = re.search(r'vector\((\d+)\)', type_str)
        if not match:
            raise ValueError(f"Could not parse vector dimension from: {type_str}")
        
        schema_dim = int(match.group(1))
        
        if schema_dim != embedding_dim:
            raise ValueError(
                f"Schema dimension mismatch: database has vector({schema_dim}), "
                f"but embedding config expects {embedding_dim}. "
                f"Update schema.sql or EMBEDDING_DIMENSION to match."
            )
        
        logger.info(f"Schema validation passed: vector({schema_dim}) matches embedding_dim={embedding_dim}")
        return True
        
    except psycopg2.Error as e:
        raise ValueError(f"Database connection error during schema validation: {e}")

