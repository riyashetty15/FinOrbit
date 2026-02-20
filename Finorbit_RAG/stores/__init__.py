"""
Vector store management for module-specific embeddings.
"""

from .vector_store_setup import (
    VectorStoreManager,
    initialize_vector_stores,
    get_vector_store_for_module,
    clear_manager_cache,
)

__all__ = [
    "VectorStoreManager",
    "initialize_vector_stores",
    "get_vector_store_for_module",
    "clear_manager_cache",
]

