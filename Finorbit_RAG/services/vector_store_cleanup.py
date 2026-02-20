"""
Vector Store Cleanup Service.

Provides methods to clear vector store data per module with proper
database transaction handling and LlamaIndex index management.
"""

import logging
from typing import Dict, Any, Literal
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

from config import (
    get_database_config,
    get_table_name_for_module,
    validate_module,
    DatabaseConfig,
)

logger = logging.getLogger(__name__)


class VectorStoreCleanupService:
    """
    Service for clearing vector store data per module.
    
    Supports clearing:
    - Chunks only (keeps documents)
    - Documents only (cascades to chunks)
    - All (chunks + documents)
    """
    
    def __init__(self, db_config: DatabaseConfig = None):
        """
        Initialize cleanup service.
        
        Args:
            db_config: Database configuration (defaults to loading from env)
        """
        self.db_config = db_config or get_database_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def clear_module(
        self,
        module: str,
        scope: Literal["chunks", "documents", "all"] = "all",
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Clear vector store data for a specific module.
        
        Args:
            module: Module name (credit, investment, insurance, retirement, taxation)
            scope: What to clear - "chunks", "documents", or "all" (default: "all")
            confirm: Must be True to proceed (safety check)
        
        Returns:
            Dictionary with:
                - status: "success" or "error"
                - module: Module name
                - scope: What was cleared
                - statistics: Counts of deleted records
                - message: Status message
        
        Raises:
            ValueError: If module is invalid or confirm is False
        """
        # Validate module
        if not validate_module(module):
            from config import MODULES
            raise ValueError(f"Invalid module: {module}. Valid modules: {MODULES}")
        
        # Require confirmation
        if not confirm:
            raise ValueError("Confirmation required. Set confirm=true to proceed with deletion.")
        
        self.logger.warning(
            f"Starting cleanup for module '{module}' with scope '{scope}' "
            f"(CONFIRMED: {confirm})"
        )
        
        # Get table name
        # Table names from config.MODULE_TABLES already include the "data_" prefix
        # (e.g., "data_investment_chunks"), so we use them directly here.
        table_name = get_table_name_for_module(module)
        chunk_table = table_name
        
        conn = None
        stats = {
            "chunks_deleted": 0,
            "documents_deleted": 0,
            "document_modules_deleted": 0
        }
        
        try:
            conn = psycopg2.connect(**self.db_config.to_dict())
            conn.autocommit = False  # Use transactions
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Clear chunks if scope is "chunks" or "all"
                if scope in ["chunks", "all"]:
                    chunks_deleted = self._clear_chunks(cur, chunk_table, module)
                    stats["chunks_deleted"] = chunks_deleted
                    self.logger.info(f"Deleted {chunks_deleted} chunks from {chunk_table}")
                
                # Clear documents if scope is "documents" or "all"
                if scope in ["documents", "all"]:
                    docs_deleted, modules_deleted = self._clear_documents(cur, module)
                    stats["documents_deleted"] = docs_deleted
                    stats["document_modules_deleted"] = modules_deleted
                    self.logger.info(
                        f"Deleted {docs_deleted} documents and {modules_deleted} document_modules "
                        f"entries for module '{module}'"
                    )
                
                # Commit transaction
                conn.commit()
                self.logger.info(f"Successfully committed cleanup transaction for module '{module}'")
            
            return {
                "status": "success",
                "module": module,
                "scope": scope,
                "statistics": stats,
                "message": f"Successfully cleared {scope} data for module '{module}'"
            }
            
        except Exception as e:
            if conn:
                conn.rollback()
                self.logger.error(f"Rolled back transaction due to error: {e}")
            
            self.logger.error(f"Error during cleanup for module '{module}': {e}", exc_info=True)
            raise
        
        finally:
            if conn:
                conn.close()
    
    def _clear_chunks(self, cursor, chunk_table: str, module: str) -> int:
        """
        Clear chunks from module-specific chunk table.
        
        Args:
            cursor: Database cursor
            chunk_table: Table name (e.g., "data_credit_chunks")
            module: Module name for filtering
        
        Returns:
            Number of chunks deleted
        """
        # Delete chunks where module matches
        delete_query = sql.SQL("""
            DELETE FROM {}
            WHERE module = %s
        """).format(sql.Identifier(chunk_table))
        
        cursor.execute(delete_query, (module,))
        deleted_count = cursor.rowcount
        
        
        return deleted_count
    
    def _clear_documents(self, cursor, module: str) -> tuple[int, int]:
        """
        Clear documents and document_modules entries for a module.
        
        Args:
            cursor: Database cursor
            module: Module name
        
        Returns:
            Tuple of (documents_deleted, document_modules_deleted)
        """
        # First, get document IDs for this module
        get_doc_ids_query = """
            SELECT DISTINCT document_id 
            FROM document_modules 
            WHERE module = %s
        """
        cursor.execute(get_doc_ids_query, (module,))
        doc_ids = [row['document_id'] for row in cursor.fetchall()]
        
        # Delete from document_modules
        delete_modules_query = """
            DELETE FROM document_modules
            WHERE module = %s
        """
        cursor.execute(delete_modules_query, (module,))
        modules_deleted = cursor.rowcount
        
        # Delete documents where module_primary matches
        # This will cascade to chunks via FK constraint
        delete_docs_query = """
            DELETE FROM documents
            WHERE module_primary = %s
        """
        cursor.execute(delete_docs_query, (module,))
        docs_deleted = cursor.rowcount
        
        return docs_deleted, modules_deleted
    
    def delete_document_by_filename(
        self,
        filename: str,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Delete document(s) and associated chunks by filename (case-insensitive exact match).
        
        Args:
            filename: Document filename (case-insensitive match)
            confirm: Must be True to proceed (safety check)
        
        Returns:
            Dictionary with:
                - status: "success" or "error"
                - filename: Filename searched
                - documents_deleted: Number of documents deleted
                - chunks_deleted_by_module: Dict mapping module -> chunks deleted
                - modules_affected: List of modules that had chunks deleted
                - message: Status message
        
        Raises:
            ValueError: If confirm is False or document not found
        """
        # Require confirmation
        if not confirm:
            raise ValueError("Confirmation required. Set confirm=true to proceed with deletion.")
        
        self.logger.warning(
            f"Starting deletion for filename '{filename}' (CONFIRMED: {confirm})"
        )
        
        conn = None
        stats = {
            "documents_deleted": 0,
            "chunks_deleted_by_module": {},
            "modules_affected": [],
            "document_modules_deleted": 0
        }
        
        try:
            conn = psycopg2.connect(**self.db_config.to_dict())
            conn.autocommit = False  # Use transactions
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Step 1: Find all documents with matching filename (case-insensitive)
                find_docs_query = """
                    SELECT id, module_primary
                    FROM documents
                    WHERE LOWER(filename) = LOWER(%s)
                """
                cur.execute(find_docs_query, (filename,))
                documents = cur.fetchall()
                
                if not documents:
                    raise ValueError(f"No documents found with filename '{filename}'")
                
                doc_ids = [row['id'] for row in documents]
                self.logger.info(f"Found {len(doc_ids)} document(s) with filename '{filename}'")
                
                # Step 2: Determine all affected modules
                # First, get modules from document_modules table
                get_modules_query = """
                    SELECT DISTINCT module
                    FROM document_modules
                    WHERE document_id = ANY(%s)
                """
                cur.execute(get_modules_query, (doc_ids,))
                modules_from_table = {row['module'] for row in cur.fetchall()}
                
                # Fallback to module_primary if document_modules is empty
                modules_from_primary = {row['module_primary'] for row in documents}
                
                # Union all modules
                all_modules = modules_from_table.union(modules_from_primary)
                stats["modules_affected"] = sorted(list(all_modules))
                
                self.logger.info(
                    f"Affected modules: {stats['modules_affected']} "
                    f"(from document_modules: {modules_from_table}, "
                    f"from module_primary: {modules_from_primary})"
                )
                
                # Step 3: Delete chunks from all affected module chunk tables
                for module in all_modules:
                    table_name = get_table_name_for_module(module)
                    # MODULE_TABLES entries already include the "data_" prefix
                    # (e.g., "data_credit_chunks"), so use them directly.
                    chunk_table = table_name
                    
                    # Count chunks before deletion for statistics
                    count_query = sql.SQL("""
                        SELECT COUNT(*) as count
                        FROM {}
                        WHERE document_id = ANY(%s)
                    """).format(sql.Identifier(chunk_table))
                    cur.execute(count_query, (doc_ids,))
                    chunk_count_before = cur.fetchone()['count']
                    
                    # Delete chunks
                    delete_chunks_query = sql.SQL("""
                        DELETE FROM {}
                        WHERE document_id = ANY(%s)
                    """).format(sql.Identifier(chunk_table))
                    cur.execute(delete_chunks_query, (doc_ids,))
                    chunks_deleted = cur.rowcount
                    
                    stats["chunks_deleted_by_module"][module] = chunks_deleted
                    self.logger.info(
                        f"Deleted {chunks_deleted} chunks from {chunk_table} "
                        f"(expected: {chunk_count_before})"
                    )
                
                # Step 4: Delete from document_modules table
                delete_modules_query = """
                    DELETE FROM document_modules
                    WHERE document_id = ANY(%s)
                """
                cur.execute(delete_modules_query, (doc_ids,))
                modules_deleted = cur.rowcount
                stats["document_modules_deleted"] = modules_deleted
                self.logger.info(f"Deleted {modules_deleted} entries from document_modules")
                
                # Step 5: Delete from documents table
                # This will cascade to chunks via FK, but we already deleted explicitly above
                delete_docs_query = """
                    DELETE FROM documents
                    WHERE id = ANY(%s)
                """
                cur.execute(delete_docs_query, (doc_ids,))
                docs_deleted = cur.rowcount
                stats["documents_deleted"] = docs_deleted
                self.logger.info(f"Deleted {docs_deleted} document(s) from documents table")
                
                # Commit transaction
                conn.commit()
                self.logger.info(
                    f"Successfully committed deletion transaction for filename '{filename}'"
                )
            
            total_chunks = sum(stats["chunks_deleted_by_module"].values())
            return {
                "status": "success",
                "filename": filename,
                "statistics": stats,
                "message": (
                    f"Successfully deleted {docs_deleted} document(s) and "
                    f"{total_chunks} chunk(s) across {len(stats['modules_affected'])} module(s)"
                )
            }
            
        except ValueError:
            # Re-raise ValueError (e.g., document not found) without rollback
            raise
        except Exception as e:
            if conn:
                conn.rollback()
                self.logger.error(f"Rolled back transaction due to error: {e}")
            
            self.logger.error(
                f"Error during deletion for filename '{filename}': {e}",
                exc_info=True
            )
            raise
        
        finally:
            if conn:
                conn.close()
    
    def delete_document_by_id(
        self,
        document_id: int,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Delete document and associated chunks by document ID.
        
        Args:
            document_id: Database document ID
            confirm: Must be True to proceed (safety check)
        
        Returns:
            Dictionary with:
                - status: "success" or "error"
                - document_id: Document ID deleted
                - documents_deleted: Number of documents deleted (should be 1)
                - chunks_deleted_by_module: Dict mapping module -> chunks deleted
                - modules_affected: List of modules that had chunks deleted
                - message: Status message
        
        Raises:
            ValueError: If confirm is False or document not found
        """
        # Require confirmation
        if not confirm:
            raise ValueError("Confirmation required. Set confirm=true to proceed with deletion.")
        
        self.logger.warning(
            f"Starting deletion for document_id={document_id} (CONFIRMED: {confirm})"
        )
        
        conn = None
        stats = {
            "documents_deleted": 0,
            "chunks_deleted_by_module": {},
            "modules_affected": [],
            "document_modules_deleted": 0
        }
        
        try:
            conn = psycopg2.connect(**self.db_config.to_dict())
            conn.autocommit = False  # Use transactions
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Step 1: Verify document exists and get its info
                find_doc_query = """
                    SELECT id, module_primary, filename
                    FROM documents
                    WHERE id = %s
                """
                cur.execute(find_doc_query, (document_id,))
                document = cur.fetchone()
                
                if not document:
                    raise ValueError(f"No document found with document_id={document_id}")
                
                doc_ids = [document_id]
                self.logger.info(f"Found document ID {document_id}: {document['filename']}")
                
                # Step 2: Determine all affected modules
                # First, get modules from document_modules table
                get_modules_query = """
                    SELECT DISTINCT module
                    FROM document_modules
                    WHERE document_id = %s
                """
                cur.execute(get_modules_query, (document_id,))
                modules_from_table = {row['module'] for row in cur.fetchall()}
                
                # Fallback to module_primary if document_modules is empty
                modules_from_primary = {document['module_primary']}
                
                # Union all modules
                all_modules = modules_from_table.union(modules_from_primary)
                stats["modules_affected"] = sorted(list(all_modules))
                
                self.logger.info(
                    f"Affected modules: {stats['modules_affected']} "
                    f"(from document_modules: {modules_from_table}, "
                    f"from module_primary: {modules_from_primary})"
                )
                
                # Step 3: Delete chunks from all affected module chunk tables
                for module in all_modules:
                    table_name = get_table_name_for_module(module)
                    chunk_table = f"data_{table_name}"
                    
                    # Count chunks before deletion for statistics
                    count_query = sql.SQL("""
                        SELECT COUNT(*) as count
                        FROM {}
                        WHERE document_id = %s
                    """).format(sql.Identifier(chunk_table))
                    cur.execute(count_query, (document_id,))
                    chunk_count_before = cur.fetchone()['count']
                    
                    # Delete chunks
                    delete_chunks_query = sql.SQL("""
                        DELETE FROM {}
                        WHERE document_id = %s
                    """).format(sql.Identifier(chunk_table))
                    cur.execute(delete_chunks_query, (document_id,))
                    chunks_deleted = cur.rowcount
                    
                    stats["chunks_deleted_by_module"][module] = chunks_deleted
                    self.logger.info(
                        f"Deleted {chunks_deleted} chunks from {chunk_table} "
                        f"(expected: {chunk_count_before})"
                    )
                
                # Step 4: Delete from document_modules table
                delete_modules_query = """
                    DELETE FROM document_modules
                    WHERE document_id = %s
                """
                cur.execute(delete_modules_query, (document_id,))
                modules_deleted = cur.rowcount
                stats["document_modules_deleted"] = modules_deleted
                self.logger.info(f"Deleted {modules_deleted} entries from document_modules")
                
                # Step 5: Delete from documents table
                delete_docs_query = """
                    DELETE FROM documents
                    WHERE id = %s
                """
                cur.execute(delete_docs_query, (document_id,))
                docs_deleted = cur.rowcount
                stats["documents_deleted"] = docs_deleted
                self.logger.info(f"Deleted {docs_deleted} document(s) from documents table")
                
                # Commit transaction
                conn.commit()
                self.logger.info(
                    f"Successfully committed deletion transaction for document_id={document_id}"
                )
            
            total_chunks = sum(stats["chunks_deleted_by_module"].values())
            return {
                "status": "success",
                "document_id": document_id,
                "statistics": stats,
                "message": (
                    f"Successfully deleted {docs_deleted} document(s) and "
                    f"{total_chunks} chunk(s) across {len(stats['modules_affected'])} module(s)"
                )
            }
            
        except ValueError:
            # Re-raise ValueError (e.g., document not found) without rollback
            raise
        except Exception as e:
            if conn:
                conn.rollback()
                self.logger.error(f"Rolled back transaction due to error: {e}")
            
            self.logger.error(
                f"Error during deletion for document_id={document_id}: {e}",
                exc_info=True
            )
            raise
        
        finally:
            if conn:
                conn.close()
    
    def get_module_statistics(self, module: str) -> Dict[str, Any]:
        """
        Get statistics about data in a module (chunks, documents count).
        
        Args:
            module: Module name
        
        Returns:
            Dictionary with statistics
        """
        if not validate_module(module):
            raise ValueError(f"Invalid module: {module}")
        
        table_name = get_table_name_for_module(module)
        # Table names from config.MODULE_TABLES already include the "data_" prefix
        # (e.g., "data_investment_chunks"), so use them directly.
        chunk_table = table_name
        
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config.to_dict())
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Count chunks
                chunk_count_query = sql.SQL("""
                    SELECT COUNT(*) as count
                    FROM {}
                    WHERE module = %s
                """).format(sql.Identifier(chunk_table))
                cur.execute(chunk_count_query, (module,))
                chunk_count = cur.fetchone()['count']
                
                # Count documents
                doc_count_query = """
                    SELECT COUNT(*) as count
                    FROM documents
                    WHERE module_primary = %s
                """
                cur.execute(doc_count_query, (module,))
                doc_count = cur.fetchone()['count']
                
                # Count document_modules entries
                module_count_query = """
                    SELECT COUNT(*) as count
                    FROM document_modules
                    WHERE module = %s
                """
                cur.execute(module_count_query, (module,))
                module_count = cur.fetchone()['count']
            
            return {
                "module": module,
                "chunks": chunk_count,
                "documents": doc_count,
                "document_modules": module_count
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics for module '{module}': {e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

