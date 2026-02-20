"""
Retrieval Pipeline using LlamaIndex for document retrieval.

Implements retrieval-only pipeline with optional features:
- HyDE (Hypothetical Document Embeddings)
- Hybrid search (vector + BM25 using QueryFusionRetriever)
- Reranking
- Metadata filtering
"""

import logging
from typing import Dict, Any, Optional, List
import Stemmer
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

# Note: nest_asyncio is applied only when needed for QueryFusionRetriever
# Not at module level to avoid conflicts with uvicorn

from core import BaseRAGPipeline
from config import (
    get_database_config,
    get_embedding_config,
    get_llamaindex_config,
    get_table_name_for_module,
    DatabaseConfig,
    EmbeddingConfig,
    LlamaIndexConfig,
)

logger = logging.getLogger(__name__)


class RetrievalPipeline(BaseRAGPipeline):
    """
    Retrieval pipeline using LlamaIndex for document retrieval.
    
    Supports optional features:
    - HyDE: Hypothetical Document Embeddings
    - Hybrid search: Vector + full-text search
    - Reranking: Post-processing reranking
    - Metadata filtering: Filter by doc_type, year, filename, etc.
    """
    
    def __init__(self, module_name: str, store):
        """
        Initialize retrieval pipeline for a specific module.
        
        Args:
            module_name: Module identifier (credit, investment, insurance, retirement, taxation)
            store: PGVectorStore instance for this module (already configured with correct table)
        """
        super().__init__(module_name, store)
        
        # Load configurations
        self.db_config = get_database_config()
        self.embedding_config = get_embedding_config()
        self.llamaindex_config = get_llamaindex_config()
        
        # Initialize embedding model (same as ingestion)
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embedding_config.model_name
        )
        
        # Create VectorStoreIndex from PGVectorStore
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.store,
            embed_model=self.embedding_model
        )
        
        # Load nodes from database into docstore for BM25Retriever
        # This is required because BM25Retriever needs nodes in-memory, not just in database
        self.logger.info(f"Loading nodes from database into docstore for module '{module_name}'...")
        nodes_loaded = self._load_nodes_to_docstore()
        self.logger.info(f"Loaded {nodes_loaded} nodes into docstore for module '{module_name}'")
        
        # Initialize BM25Retriever by default (if docstore has documents)
        # This allows hybrid search to be used immediately without lazy initialization
        self.bm25_retriever = None
        self._bm25_initialized = False
        
        if nodes_loaded > 0:
            try:
                docstore = self.index.docstore
                
                # Check if docstore has documents
                doc_count = 0
                if hasattr(docstore, 'docs'):
                    doc_count = len(docstore.docs) if docstore.docs else 0
                elif hasattr(docstore, 'get_all_document_hashes'):
                    doc_hashes = docstore.get_all_document_hashes()
                    doc_count = len(doc_hashes) if doc_hashes else 0
                elif hasattr(docstore, '_node_dict'):
                    doc_count = len(docstore._node_dict) if docstore._node_dict else 0
                
                if doc_count > 0:
                    # Initialize BM25Retriever with default parameters
                    # Note: Filters and top_k will be set per-query, but base retriever is ready
                    self.bm25_retriever = BM25Retriever.from_defaults(
                        docstore=docstore,
                        similarity_top_k=10,  # Default, will be overridden per query
                        filters=None,  # No filters by default, will be set per query
                        stemmer=Stemmer.Stemmer("english"),
                        language="english"
                    )
                    self._bm25_initialized = True
                    self.logger.info(
                        f"Initialized BM25Retriever by default for module '{module_name}' "
                        f"(docstore has {doc_count} documents)"
                    )
                else:
                    self.logger.warning(
                        f"BM25Retriever not initialized: docstore is empty for module '{module_name}'"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Could not initialize BM25Retriever by default for module '{module_name}': {e}. "
                    f"Hybrid search will not be available until retriever is initialized."
                )
                self.bm25_retriever = None
                self._bm25_initialized = False
        else:
            self.logger.warning(
                f"No nodes loaded for module '{module_name}', BM25Retriever not initialized"
            )
        
        self.logger.info(
            f"Initialized RetrievalPipeline for module '{module_name}' "
            f"with embedding model '{self.embedding_config.model_name}'"
        )
    
    
    def ingest(self, document_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest method - not implemented for retrieval pipeline.
        
        Use TextIngestionPipeline for ingestion.
        """
        raise NotImplementedError(
            "RetrievalPipeline is for querying only. "
            "Use TextIngestionPipeline for ingestion."
        )
    def _expand_with_neighbors(
        self,
        centers: List[NodeWithScore],
        window: int = 1,
    ) -> Dict[str, str]:
        """
        For each central node, fetch neighbor chunks from the same document
        using (filename, chunk_index) and build an expanded context string
        (previous + center + next chunks).

        Uses a JOIN to documents on filename so it works even when
        metadata['document_id'] is a UUID string and the DB FK is BIGINT.

        Returns:
            dict[node_id -> contextual_text]
        """
        from config import get_table_name_for_module

        table_name = get_table_name_for_module(self.module_name)
        full_table_name = f"data_{table_name}"

        # Group centers by filename so we can do one DB query per file
        # { filename: [ (node, chunk_index), ... ] }
        centers_by_file: Dict[str, List[tuple[NodeWithScore, int]]] = {}

        for node in centers:
            meta = node.metadata or {}
            filename = meta.get("filename")
            cidx = meta.get("chunk_index")

            if not filename or cidx is None:
                continue

            try:
                cidx_int = int(cidx)
            except (TypeError, ValueError):
                continue

            centers_by_file.setdefault(filename, []).append((node, cidx_int))

        if not centers_by_file:
            return {}

        contextual_map: Dict[str, str] = {}
        conn = None

        try:
            conn = psycopg2.connect(**self.db_config.to_dict())
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for filename, node_list in centers_by_file.items():
                    # Compute combined min/max chunk_index window for this file
                    all_indices = [idx for _, idx in node_list]
                    if not all_indices:
                        continue

                    min_idx = max(0, min(all_indices) - window)
                    max_idx = max(all_indices) + window

                    # Fetch all chunks for this filename in that index window
                    cur.execute(
                        sql.SQL("""
                            SELECT c.chunk_index, c.text
                            FROM {} AS c
                            JOIN documents d ON c.document_id = d.id
                            WHERE d.filename = %s
                              AND c.chunk_index BETWEEN %s AND %s
                            ORDER BY c.chunk_index
                        """).format(sql.Identifier(full_table_name)),
                        (filename, min_idx, max_idx),
                    )
                    rows = cur.fetchall()
                    by_idx = {r["chunk_index"]: r["text"] for r in rows}

                    # Build expanded text for each center in this file
                    for center, cidx in node_list:
                        start_idx = cidx - window
                        end_idx = cidx + window

                        snippets: List[str] = []
                        for i in range(start_idx, end_idx + 1):
                            if i in by_idx:
                                snippets.append(by_idx[i])

                        contextual_text = "\n".join(snippets) if snippets else center.get_content()
                        contextual_map[center.node_id] = contextual_text

        except Exception as e:
            self.logger.error(f"Error expanding neighbors: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

        return contextual_map

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        use_hyde: bool = False,
        use_hybrid: bool = True,
        use_rerank: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the vector store and retrieve relevant chunks.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            use_hyde: Use HyDE for enhanced retrieval (TODO: implement)
            use_hybrid: Use hybrid search (vector + BM25) - combines vector similarity and BM25 text search using QueryFusionRetriever with RRF
            use_rerank: Use reranking for improved relevance (TODO: implement)
            **kwargs: Additional parameters (doc_type, year, filename for filtering)
        
        Returns:
            Dictionary containing:
                - query: Original query text
                - module: Module name
                - chunks: List of retrieved chunks with scores and metadata
                - total_results: Number of results
                - hyde_used: Whether HyDE was used
                - hybrid_used: Whether hybrid search was used
                - rerank_used: Whether reranking was used
        """
        self.logger.info(
            f"Querying module '{self.module_name}' with query: '{query_text[:50]}...' "
            f"(top_k={top_k}, hyde={use_hyde}, hybrid={use_hybrid}, rerank={use_rerank})"
        )
        
        # Validate query text
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        # Only current, public, non-PII chunks unless explicitly overridden
        # if "is_current" not in kwargs:
        #     kwargs["is_current"] = True
        # if "security" not in kwargs:
        #     kwargs["security"] = "Public"
        # if "pii" not in kwargs:
        #     kwargs["pii"] = False
        
        # Handle HyDE (TODO: implement full HyDE generation)
        query_text_for_embedding = query_text
        if use_hyde:
            self.logger.warning("HyDE is requested but not yet implemented, using original query")
            # TODO: Generate hypothetical document using LLM
            # hyde_text = self._generate_hyde_document(query_text)
            # query_text_for_embedding = hyde_text
        
        # Build metadata filters
        metadata_filters = self._build_metadata_filters(kwargs)
        
        # Determine retrieval count (retrieve more if reranking is enabled)
        retrieve_count = top_k * 2 if use_rerank else top_k
        
        # Retrieve nodes using hybrid search or vector-only
        try:
            # BM25Retriever should already be initialized in __init__ if nodes were loaded
            # If hybrid search is requested but BM25Retriever is not available, fall back to vector-only
            if use_hybrid and not self._bm25_initialized:
                self.logger.warning(
                    "Hybrid search requested but BM25Retriever is not initialized. "
                    "This may happen if no documents were loaded or initialization failed. "
                    "Falling back to vector-only search."
                )
                use_hybrid = False
            
            if use_hybrid and self.bm25_retriever:
                # Use QueryFusionRetriever to combine vector and BM25 retrievers
                # This automatically handles RRF merging
                # Create a new BM25Retriever instance with updated parameters for this query
                # (filters and top_k might be different from the default initialized one)
                # Note: We create a new instance per query to apply query-specific filters and top_k
                bm25_retriever_instance = BM25Retriever.from_defaults(
                    docstore=self.index.docstore,
                    similarity_top_k=retrieve_count,
                    filters=metadata_filters,  # Apply same metadata filters
                    stemmer=Stemmer.Stemmer("english"),
                    language="english"
                )
                
                hybrid_retriever = QueryFusionRetriever(
                    retrievers=[
                        self.index.as_retriever(
                            similarity_top_k=retrieve_count,
                            filters=metadata_filters
                        ),
                        bm25_retriever_instance,
                    ],
                    num_queries=1,  # Don't generate query variations
                    use_async=False,  # Use async for better performance
                    similarity_top_k=retrieve_count,  # Final number of results after fusion
                    llm=None,  # Disable LLM usage - not needed when num_queries=1
                )
                
                nodes = hybrid_retriever.retrieve(query_text_for_embedding)
                self.logger.info(f"Hybrid search: Retrieved {len(nodes)} merged nodes from vector + BM25")
            
            else:
                # Vector-only retrieval
                if use_hybrid and not self.bm25_retriever:
                    self.logger.warning("Hybrid search requested but BM25Retriever not available, using vector-only")
                
                vector_retriever = self.index.as_retriever(
                    similarity_top_k=retrieve_count,
                    filters=metadata_filters
                )
                nodes = vector_retriever.retrieve(query_text_for_embedding)
                self.logger.info(f"Vector search: Retrieved {len(nodes)} nodes from vector store")
                
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}", exc_info=True)
            raise
        
        # Handle reranking (TODO: implement)
        if use_rerank:
            self.logger.warning("Reranking is requested but not yet implemented")
            # TODO: Apply reranking
            # from llama_index.core.postprocessor import SentenceTransformerRerank
            # reranker = SentenceTransformerRerank(...)
            # nodes = reranker.postprocess_nodes(nodes, query_str=query_text)
            # nodes = nodes[:top_k]  # Take top_k after reranking
        
        # Format results
        results = self._format_results(nodes, query_text, use_hyde, use_hybrid, use_rerank)
        
        contextual_map = self._expand_with_neighbors(nodes)
        results["context_window"] = {"before_text": 1, "after_text": 1}

        # Attach contextual_text per chunk (fallback to original text if missing)
        for chunk in results.get("chunks", []):
            node_id = chunk.get("id")
            if node_id and node_id in contextual_map:
                chunk["contextual_text"] = contextual_map[node_id]
            else:
                chunk["contextual_text"] = chunk.get("text", "")

        self.logger.info(
            f"Query completed: {results['total_results']} results returned "
            f"for module '{self.module_name}' with neighbor expansion "
            f"(before={results['context_window']['before_text']}, after={results['context_window']['after_text']})"
        )

        return results
    
    def _build_metadata_filters(self, kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        Build LlamaIndex metadata filters from kwargs.

        Supported keys:
        - doc_type, year, year_min, filename
        - module, issuer, jurisdiction, language, regulator_tag
        - security, version, version_id, effective_date
        - is_current (bool-ish), pii (bool-ish)
        - compliance_tags_any: list[str] -> IN filter
        """
        from llama_index.core.vector_stores import (
            MetadataFilters,
            ExactMatchFilter,
            MetadataFilter,
            FilterOperator,
        )

        filter_list = []

        # Simple exact-match fields
        simple_exact = [
            "doc_type",
            "year",
            "filename",
            "module",
            "issuer",
            "jurisdiction",
            "language",
            "regulator_tag",
            "security",
            "version",
            "version_id",
            "effective_date",
        ]

        for key in simple_exact:
            if key in kwargs and kwargs[key] not in (None, ""):
                filter_list.append(ExactMatchFilter(key=key, value=str(kwargs[key])))

        # Range filter for year_min (year >= year_min)
        if "year_min" in kwargs and kwargs["year_min"] is not None:
            try:
                year_min_int = int(kwargs["year_min"])
                filter_list.append(
                    MetadataFilter(
                        key="year",
                        value=year_min_int,
                        operator=FilterOperator.GTE,
                    )
                )
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid year_min value: {kwargs['year_min']}, skipping filter")

        # Boolean flags stored as booleans in metadata_
        if "is_current" in kwargs and kwargs["is_current"] is not None:
            filter_list.append(
                ExactMatchFilter(key="is_current", value="true" if bool(kwargs["is_current"]) else "false")
            )

        if "pii" in kwargs and kwargs["pii"] is not None:
            filter_list.append(
                ExactMatchFilter(key="pii", value="true" if bool(kwargs["pii"]) else "false")
            )

        # Array-style: compliance_tags_any -> IN
        compliance_tags_any = kwargs.get("compliance_tags_any")
        if compliance_tags_any:
            if not isinstance(compliance_tags_any, (list, tuple)):
                compliance_tags_any = [compliance_tags_any]
            filter_list.append(
                MetadataFilter(
                    key="compliance_tags",
                    value=list(compliance_tags_any),
                    operator=FilterOperator.IN,
                )
            )

        if not filter_list:
            return None

        return MetadataFilters(filters=filter_list)
    
    def _load_nodes_to_docstore(self) -> int:
        """
        Load nodes from database into docstore for BM25Retriever.
        
        BM25Retriever requires nodes to be in the docstore (in-memory),
        not just in the database. This method queries the database and
        populates the docstore with all nodes.
        
        Returns:
            Number of nodes loaded into docstore
        """
        from config import get_table_name_for_module
        
        table_name = get_table_name_for_module(self.module_name)
        full_table_name = f"data_{table_name}"
        docstore = self.index.docstore
        
        conn = None
        nodes_loaded = 0
        
        try:
            conn = psycopg2.connect(**self.db_config.to_dict())
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query all nodes from the database
                query_sql = sql.SQL("""
                    SELECT 
                        c.node_id,
                        c.text,
                        c.metadata_,
                        c.page_number,
                        c.chunk_index,
                        c.document_id,
                        d.filename
                    FROM {} AS c
                    LEFT JOIN documents AS d ON c.document_id = d.id
                    WHERE c.text IS NOT NULL 
                      AND c.text != ''
                      AND c.node_id IS NOT NULL
                    ORDER BY c.id
                """).format(sql.Identifier(full_table_name))
                
                cur.execute(query_sql)
                rows = cur.fetchall()
            
            # Convert database rows to TextNode objects and add to docstore
            for row in rows:
                try:
                    # Extract metadata
                    node_metadata = row.get('metadata_') or {}
                    if isinstance(node_metadata, str):
                        import json
                        try:
                            node_metadata = json.loads(node_metadata)
                        except:
                            node_metadata = {}
                    
                    # Enrich metadata with DB fields for traceability
                    # Ensure page_number, chunk_index, document_id, and filename are present
                    try:
                        if 'page_number' not in node_metadata:
                            node_metadata['page_number'] = row.get('page_number')
                        if 'chunk_index' not in node_metadata:
                            node_metadata['chunk_index'] = row.get('chunk_index')
                        # Prefer integer document_id
                        if 'document_id' not in node_metadata or node_metadata.get('document_id') is None:
                            node_metadata['document_id'] = row.get('document_id')
                        # Add filename for convenience if available
                        if 'filename' not in node_metadata:
                            node_metadata['filename'] = row.get('filename')
                    except Exception:
                        # Non-fatal; proceed even if enrichment fails
                        pass

                    # Create TextNode
                    text_node = TextNode(
                        text=row['text'],
                        node_id=row['node_id'],
                        metadata=node_metadata
                    )
                    
                    # Add to docstore
                    docstore.add_documents([text_node])
                    nodes_loaded += 1
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load node {row.get('node_id', 'unknown')} into docstore: {e}"
                    )
                    continue
            
            self.logger.info(
                f"Successfully loaded {nodes_loaded} nodes from database into docstore "
                f"for module '{self.module_name}'"
            )
            
            return nodes_loaded
            
        except Exception as e:
            self.logger.error(
                f"Error loading nodes into docstore for module '{self.module_name}': {e}",
                exc_info=True
            )
            return 0
        finally:
            if conn:
                conn.close()
    
    def _format_results(
        self,
        nodes: List[NodeWithScore],
        query_text: str,
        hyde_used: bool,
        hybrid_used: bool,
        rerank_used: bool
    ) -> Dict[str, Any]:
        """
        Format LlamaIndex NodeWithScore objects to structured response.
        
        Filters out internal LlamaIndex metadata fields and keeps only business metadata.
        
        Args:
            nodes: List of NodeWithScore objects from retriever
            query_text: Original query text
            hyde_used: Whether HyDE was used
            hybrid_used: Whether hybrid search was used
            rerank_used: Whether reranking was used
        
        Returns:
            Dictionary with formatted results
        """
        chunks = []
        
        # Internal LlamaIndex fields to filter out
        internal_fields = {
            '_node_type', '_node_content', 'ref_doc_id', 
            'doc_id',  # String version, we'll use document_id (integer) instead
        }
        
        for node in nodes:
            node_metadata = node.metadata or {}
            
            # Filter out internal LlamaIndex fields and keep only business metadata
            clean_metadata = {
                k: v for k, v in node_metadata.items() 
                if not k.startswith('_') and k not in internal_fields
            }
            
            # Convert document_id from string to integer if it exists as string
            if 'document_id' in clean_metadata:
                doc_id = clean_metadata['document_id']
                if isinstance(doc_id, str) and doc_id.startswith('doc_'):
                    # Try to extract integer from "doc_1" format
                    try:
                        clean_metadata['document_id'] = int(doc_id.replace('doc_', ''))
                    except (ValueError, AttributeError):
                        pass
            
            chunk_data = {
                "id": node.node_id,
                "text": node.get_content(),
                "score": float(node.score) if node.score is not None else 0.0,
                "metadata": clean_metadata,
                "document_filename": clean_metadata.get("filename"),
                "page_number": clean_metadata.get("page_number"),
                "chunk_index": clean_metadata.get("chunk_index")
            }
            
            chunks.append(chunk_data)
        
        return {
            "query": query_text,
            "module": self.module_name,
            "chunks": chunks,
            "total_results": len(chunks),
            "hyde_used": hyde_used,
            "hybrid_used": hybrid_used,
            "rerank_used": rerank_used
        }

