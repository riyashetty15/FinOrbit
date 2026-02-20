"""
Vector store setup for modular RAG pipeline with LlamaIndex.

Initializes and manages PGVectorStore instances for each business module,
providing connection management and validation.
"""

import logging
from typing import Dict, Optional
import psycopg2

from llama_index.vector_stores.postgres import PGVectorStore

from config import (
    MODULES,
    MODULE_TABLES,
    get_database_config,
    get_embedding_config,
    get_table_name_for_module,
    validate_module,
    DatabaseConfig,
    EmbeddingConfig
)

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level cache for VectorStoreManager (singleton pattern)
# ============================================================================

_manager_cache = None  # Will be VectorStoreManager instance
_cache_config_hash: Optional[str] = None


def _get_config_hash(db_config: DatabaseConfig, embedding_config: EmbeddingConfig) -> str:
    """Generate a hash of the configuration for cache key"""
    import hashlib
    config_str = (
        f"{db_config.host}:{db_config.port}:{db_config.database}:"
        f"{db_config.user}:{embedding_config.model_name}:{embedding_config.dimension}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_or_create_manager(
    db_config: Optional[DatabaseConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    validate_schema: bool = True,
    force_new: bool = False
):
    """
    Get or create a cached VectorStoreManager instance.
    
    Args:
        db_config: Database configuration
        embedding_config: Embedding configuration
        validate_schema: Whether to validate schema
        force_new: Force creation of a new manager (bypass cache)
    
    Returns:
        VectorStoreManager instance
    """
    global _manager_cache, _cache_config_hash
    
    if db_config is None:
        db_config = get_database_config()
    if embedding_config is None:
        embedding_config = get_embedding_config()
    
    config_hash = _get_config_hash(db_config, embedding_config)
    
    # Return cached manager if config matches and not forcing new
    if not force_new and _manager_cache is not None and _cache_config_hash == config_hash:
        logger.debug("Reusing cached VectorStoreManager")
        return _manager_cache
    
    # Create new manager
    logger.debug("Creating new VectorStoreManager instance")
    _manager_cache = VectorStoreManager(
        db_config=db_config,
        embedding_config=embedding_config,
        validate_schema=validate_schema
    )
    _cache_config_hash = config_hash
    return _manager_cache


def clear_manager_cache():
    """Clear the cached VectorStoreManager instance"""
    global _manager_cache, _cache_config_hash
    if _manager_cache is not None:
        _manager_cache.close_all()
    _manager_cache = None
    _cache_config_hash = None
    logger.debug("VectorStoreManager cache cleared")


class VectorStoreManager:
    """Manages PGVectorStore instances for all modules"""
    
    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        validate_schema: bool = True
    ):
        """
        Initialize vector store manager for all modules.
        
        Args:
            db_config: Database configuration (defaults to loading from env)
            embedding_config: Embedding configuration (defaults to loading from env)
            validate_schema: Whether to validate tables/indexes exist on init
        """
        self.db_config = db_config or get_database_config()
        self.embedding_config = embedding_config or get_embedding_config()
        self.vector_stores: Dict[str, PGVectorStore] = {}
        
        if validate_schema:
            self._validate_schema()
        
        self._initialize_stores()
    
    def _validate_schema(self):
        """Validate that all required tables and indexes exist"""
        logger.info("Validating database schema...")
        
        try:
            with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                # Check if pgvector extension exists
                cursor.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """)
                if not cursor.fetchone()[0]:
                    raise ValueError(
                        "pgvector extension not found. Run: CREATE EXTENSION vector;"
                    )
                logger.info("✓ pgvector extension found")
                
                # Check documents table
                cursor.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'documents'
                    )
                """)
                if not cursor.fetchone()[0]:
                    raise ValueError(
                        "documents table not found. Please run schema.sql first."
                    )
                logger.info("documents table found")
                
                # Check all module chunk tables
                missing_tables = []
                for module, table_name in MODULE_TABLES.items():
                    cursor.execute("""
                        SELECT EXISTS(
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_schema = 'public' AND table_name = %s
                        )
                    """, (table_name,))
                    
                    if not cursor.fetchone()[0]:
                        missing_tables.append(f"{table_name} (module: {module})")
                
                if missing_tables:
                    raise ValueError(
                        f"Missing chunk tables: {', '.join(missing_tables)}. "
                        "Please run schema.sql first."
                    )
                
                logger.info(f"All {len(MODULE_TABLES)} chunk tables found")
                
                # Check embedding indexes (sample check on first table)
                first_table = list(MODULE_TABLES.values())[0]
                cursor.execute("""
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE tablename = %s AND indexname LIKE %s
                """, (first_table, '%embedding_idx%'))
                
                index_count = cursor.fetchone()[0]
                if index_count == 0:
                    logger.warning(
                        f"No embedding index found on {first_table}. "
                        "Performance may be degraded. Check schema.sql."
                    )
                else:
                    logger.info("Embedding indexes found")
                
                logger.info("Schema validation completed successfully")
                
        except psycopg2.Error as e:
            raise ValueError(f"Database connection error during schema validation: {e}")
    
    def _initialize_stores(self):
        """Initialize PGVectorStore for each module"""
        logger.info(f"Initializing PGVectorStore instances for {len(MODULES)} modules...")
        
        for module in MODULES:
            try:
                table_name = get_table_name_for_module(module)
                # LlamaIndex prepends 'data_' to table names automatically in PGVectorStore.
                # Since our config contains the full table name (e.g. 'data_credit_chunks'),
                # we must strip the 'data_' prefix to avoid 'data_data_credit_chunks'.
                llama_table_name = table_name
                if llama_table_name.startswith("data_"):
                    llama_table_name = llama_table_name[5:]  # Strip 'data_'
                
                # The actual table name in SQL (used for verification)
                expected_table = table_name
                
                # Initialize PGVectorStore with module-specific table
                # LlamaIndex will create data_{module}_chunks tables automatically
                # Note: PGVectorStore.from_params() constructs a connection string internally,
                # so we need to URL-encode the password to handle special characters like @
                encoded_password = self.db_config.get_url_encoded_password()
                vector_store = PGVectorStore.from_params(
                    database=self.db_config.database,
                    host=self.db_config.host,
                    password=encoded_password,
                    port=self.db_config.port,
                    user=self.db_config.user,
                    table_name=llama_table_name,
                    embed_dim=self.embedding_config.dimension,
                    # LlamaIndex will auto-create tables with schema:
                    # (id, text, metadata_, node_id, embedding)
                )
                
                # Verify the table name being used
                actual_table = getattr(vector_store, 'table_name', expected_table)
                logger.info(f"✓ Initialized vector store for module '{module}' (table: {table_name}, actual: {actual_table}, expected: {expected_table})")
                
                # Check if table has data (LlamaIndex creates tables with 'data_' prefix)
                try:
                    with psycopg2.connect(**self.db_config.to_dict()) as conn, conn.cursor() as cursor:
                        cursor.execute(f"SELECT COUNT(*) FROM {expected_table} WHERE embedding IS NOT NULL")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            logger.info(f"  → Table '{expected_table}' contains {count} chunks with embeddings")
                        else:
                            logger.warning(f"  → Table '{expected_table}' is empty or has no embeddings")
                except Exception as e:
                    # Table might not exist yet (LlamaIndex will create it on first insert)
                    logger.debug(f"  → Table '{expected_table}' not found or not accessible: {e}")
                
                self.vector_stores[module] = vector_store
                
            except Exception as e:
                logger.error(f"✗ Failed to initialize vector store for module '{module}': {e}")
                raise
    
    def get_store(self, module: str) -> PGVectorStore:
        """
        Get PGVectorStore instance for a specific module.
        
        Args:
            module: Module name (credit, investment, insurance, retirement, taxation)
        
        Returns:
            PGVectorStore instance for the module
        
        Raises:
            ValueError: If module is not recognized
        """
        if not validate_module(module):
            raise ValueError(f"Unknown module: {module}. Valid modules: {MODULES}")
        
        if module not in self.vector_stores:
            raise ValueError(
                f"Vector store not initialized for module '{module}'. "
                "Ensure VectorStoreManager was properly initialized."
            )
        
        return self.vector_stores[module]
    
    def get_all_stores(self) -> Dict[str, PGVectorStore]:
        """Get dictionary of all initialized vector stores"""
        return self.vector_stores.copy()
    
    def close_all(self):
        """Close all vector store connections"""
        # PGVectorStore manages its own connections, but we can clear references
        self.vector_stores.clear()
        logger.info("All vector store references cleared")


# ============================================================================
# Convenience Functions
# ============================================================================

def initialize_vector_stores(
    db_config: Optional[DatabaseConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    validate_schema: bool = True,
    force_new: bool = False
) -> Dict[str, PGVectorStore]:
    """
    Initialize and return all PGVectorStore instances.
    
    Convenience function that creates or reuses a cached VectorStoreManager
    and returns the dictionary of stores.
    
    Args:
        db_config: Database configuration (defaults to loading from env)
        embedding_config: Embedding configuration (defaults to loading from env)
        validate_schema: Whether to validate schema on initialization
        force_new: Force creation of a new manager (bypass cache)
    
    Returns:
        Dictionary mapping module names to PGVectorStore instances
    """
    manager = _get_or_create_manager(
        db_config=db_config,
        embedding_config=embedding_config,
        validate_schema=validate_schema,
        force_new=force_new
    )
    return manager.get_all_stores()


def get_vector_store_for_module(
    module: str,
    db_config: Optional[DatabaseConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    force_new: bool = False
) -> PGVectorStore:
    """
    Get a single PGVectorStore instance for a specific module.
    
    Uses a cached VectorStoreManager to avoid reinitializing stores on each call.
    
    Args:
        module: Module name
        db_config: Database configuration (defaults to loading from env)
        embedding_config: Embedding configuration (defaults to loading from env)
        force_new: Force creation of a new manager (bypass cache)
    
    Returns:
        PGVectorStore instance for the module
    """
    manager = _get_or_create_manager(
        db_config=db_config,
        embedding_config=embedding_config,
        validate_schema=False,  # Skip validation for single-store access
        force_new=force_new
    )
    return manager.get_store(module)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("Testing vector store initialization...")
        stores = initialize_vector_stores()
        
        logger.info(f"\n✓ Successfully initialized {len(stores)} vector stores:")
        for module, store in stores.items():
            logger.info(f"  - {module}: {store}")
        
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        raise

