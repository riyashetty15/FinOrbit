-- schema.sql
-- Modular RAG database schema for LlamaIndex + PostgreSQL (pgvector)
-- 
-- This schema creates tables that are compatible with LlamaIndex's PGVectorStore
-- while adding custom tracking fields for document management.

-- ============================================================================
-- Database Creation Check
-- ============================================================================
-- PostgreSQL doesn't support CREATE DATABASE IF NOT EXISTS directly.
-- Database must be created before running this script.
--
-- The script below checks if the database exists and provides instructions.
-- To create the database, use one of these methods:
--
-- Option 1: Using psql command line (recommended)
--   createdb -U postgres financial_rag
--
-- Option 2: Using SQL (connect to 'postgres' database first)
--   \c postgres
--   CREATE DATABASE financial_rag;
--   \c financial_rag
--
-- Option 3: Using shell script
--   psql -U postgres -c "CREATE DATABASE financial_rag;"

DO $$
DECLARE
    db_exists BOOLEAN;
    current_db TEXT;
BEGIN
    -- Get current database name
    SELECT current_database() INTO current_db;
    
    -- Check if target database exists
    SELECT EXISTS (
        SELECT 1 FROM pg_database WHERE datname = 'financial_rag'
    ) INTO db_exists;
    
    -- Provide status and instructions
    IF current_db = 'financial_rag' THEN
        RAISE NOTICE '✓ Connected to financial_rag database - ready to create schema';
    ELSIF db_exists THEN
        RAISE NOTICE '✓ Database financial_rag exists';
        RAISE NOTICE '  Please connect to it: \c financial_rag';
        RAISE WARNING 'Not connected to financial_rag database. Connect first before running schema.';
    ELSE
        RAISE NOTICE '✗ Database financial_rag does not exist';
        RAISE NOTICE '  To create it, run one of these commands:';
        RAISE NOTICE '    1. createdb -U postgres financial_rag';
        RAISE NOTICE '    2. psql -U postgres -c "CREATE DATABASE financial_rag;"';
        RAISE NOTICE '    3. \c postgres (then) CREATE DATABASE financial_rag;';
        RAISE WARNING 'Database financial_rag does not exist. Please create it first.';
    END IF;
END $$;

-- ============================================================================
-- Enable pgvector extension for embedding support
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-------------------------------------------------------------------------------
-- documents: single source of truth for all ingested files
-------------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id                  BIGSERIAL PRIMARY KEY,
    external_id         TEXT,
    filename            TEXT NOT NULL,
    uri_or_path         TEXT NOT NULL,
    mime_type           TEXT NOT NULL,
    source_type         TEXT NOT NULL,
    checksum            TEXT NOT NULL,
    size_bytes          BIGINT,
    pages               INTEGER,

    module_primary      TEXT NOT NULL,
    tags                JSONB DEFAULT '{}'::JSONB,

    ingest_status       TEXT NOT NULL DEFAULT 'queued' CHECK (ingest_status IN (
                            'queued', 'processing', 'complete', 'failed', 'skipped_duplicate'
                        )),
    ingest_error        TEXT,
    owner               TEXT,
    retention_policy    TEXT,
    pii_level           TEXT,
    pipeline_version    TEXT,
    ocr_applied         BOOLEAN DEFAULT FALSE,
    ocr_language        TEXT,

    version             INTEGER NOT NULL DEFAULT 1,
    content_hash        TEXT,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (checksum),
    UNIQUE (content_hash)
);

COMMENT ON TABLE documents IS 'Authoritative metadata for all ingested source documents. Embeddings are stored in module-specific chunk tables.';

CREATE TABLE IF NOT EXISTS document_modules (
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    module      TEXT   NOT NULL,
    PRIMARY KEY (document_id, module)
);

-- Helpful supporting indexes on documents
CREATE INDEX IF NOT EXISTS documents_module_primary_idx ON documents(module_primary);
CREATE INDEX IF NOT EXISTS documents_mime_type_idx ON documents(mime_type);
CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents(created_at);

-------------------------------------------------------------------------------
-- Trigger to maintain updated_at on updates
-------------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at_tr
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-------------------------------------------------------------------------------
-- Create chunk tables per module (LlamaIndex compatible + custom fields)
-------------------------------------------------------------------------------
-- 
-- LlamaIndex's default schema requires:
--   - id (bigint, primary key)
--   - text (character varying, NOT NULL)
--   - metadata_ (json/jsonb, nullable)
--   - node_id (character varying, nullable)
--   - embedding (vector, nullable)
--
-- We add custom tracking fields:
--   - document_id (links to documents table)
--   - chunk_index (ordering within document)
--   - metadata_ (PRIMARY: LlamaIndex stores all business metadata here - doc_type, year, filename, etc.)
--   - module (module name for filtering)
--   - Additional tracking fields (page_number, etc.)

DO $$
DECLARE
    chunk_table  TEXT;
    module_name  TEXT;
BEGIN
    FOR chunk_table, module_name IN
        SELECT * FROM (VALUES
            ('data_credit_chunks',    'credit'),
            ('data_investment_chunks','investment'),
            ('data_insurance_chunks', 'insurance'),
            ('data_retirement_chunks','retirement'),
            ('data_tax_chunks',       'taxation')
        ) AS modules(table_name, module_tag)
    LOOP
        EXECUTE format($ddl$
            CREATE TABLE IF NOT EXISTS %I (
                -- LlamaIndex required columns (must match exactly)
                id                      BIGSERIAL PRIMARY KEY,
                text                    VARCHAR NOT NULL,
                metadata_               JSONB,
                node_id                 VARCHAR,
                embedding               vector(768),
                
                -- Custom tracking fields (additional to LlamaIndex schema)
                document_id             BIGINT REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index             INTEGER,
                module                  TEXT NOT NULL DEFAULT %L,
                page_number             INTEGER,
                
                -- Additional metadata fields
                embedding_model         TEXT,
                embedding_model_version TEXT,
                embedding_created_at    TIMESTAMPTZ,
                language                TEXT,
                ocr_processed           BOOLEAN DEFAULT FALSE,
                quality_flags           JSONB,
                
                -- Statistics
                token_count             INTEGER,
                char_count              INTEGER,
                word_count              INTEGER,
                chunk_hash              TEXT,
                pipeline_version        TEXT,
                
                -- Timestamps
                created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                
                -- Full-text search support (OPTIONAL: for direct SQL queries)
                -- NOTE: BM25 retriever uses in-memory text indexing, not this column
                -- This column can be used for custom SQL-based FTS if needed
                tsv                     TSVECTOR GENERATED ALWAYS AS (
                                            to_tsvector('english', coalesce(text, ''))
                                        ) STORED,
                
                -- Constraints
                CONSTRAINT %I_module_ck CHECK (module = %L),
                UNIQUE (document_id, chunk_index),
                UNIQUE (document_id, chunk_hash)
            );
        $ddl$, chunk_table, module_name, chunk_table, module_name);
        
        -- Embedding index for similarity search (HNSW for better performance)
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_embedding_idx
            ON %I USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        $ddl$, chunk_table, chunk_table);
        
        -- Index on node_id for LlamaIndex compatibility
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_node_id_idx ON %I (node_id);
        $ddl$, chunk_table, chunk_table);
        
        -- Supporting indexes for fast filtering
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_doc_chunk_idx ON %I (document_id, chunk_index);
        $ddl$, chunk_table, chunk_table);
        
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_created_at_idx ON %I (created_at);
        $ddl$, chunk_table, chunk_table);
        
        -- JSONB indexes for metadata-rich queries
        -- Primary index on metadata_ (LlamaIndex built-in column with all business metadata)
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_metadata__idx
            ON %I USING gin (metadata_);
        $ddl$, chunk_table, chunk_table);
        
        -- Expression indexes on metadata_ for common filter fields
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_doc_type_idx
            ON %I ((metadata_->>'doc_type'));
        $ddl$, chunk_table, chunk_table);
        
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_year_idx
            ON %I ((metadata_->>'year'));
        $ddl$, chunk_table, chunk_table);
        
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_filename_idx
            ON %I ((metadata_->>'filename'));
        $ddl$, chunk_table, chunk_table);

                EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_module_idx
            ON %I ((metadata_->>'module'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_issuer_idx
            ON %I ((metadata_->>'issuer'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_language_idx
            ON %I ((metadata_->>'language'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_regulator_tag_idx
            ON %I ((metadata_->>'regulator_tag'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_security_idx
            ON %I ((metadata_->>'security'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_version_id_idx
            ON %I ((metadata_->>'version_id'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_is_current_idx
            ON %I ((metadata_->>'is_current'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_pii_idx
            ON %I ((metadata_->>'pii'));
        $ddl$, chunk_table, chunk_table);

        -- for array-style JSONB fields
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_compliance_tags_idx
            ON %I USING gin ((metadata_->'compliance_tags'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_tags_array_idx
            ON %I USING gin ((metadata_->'tags'));
        $ddl$, chunk_table, chunk_table);
        
        -- Full-text search index (OPTIONAL: for direct SQL queries)
        -- NOTE: BM25Retriever doesn't use this index - it uses in-memory indexing
        -- Keep this index only if you plan to do custom SQL-based FTS queries
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_tsv_idx
            ON %I USING gin (tsv);
        $ddl$, chunk_table, chunk_table);
        
        RAISE NOTICE 'Created table % with LlamaIndex-compatible schema', chunk_table;
    END LOOP;
END $$;

-------------------------------------------------------------------------------
-- ingestion_jobs: Track async document ingestion jobs
-------------------------------------------------------------------------------
-- This table stores the status and results of async ingestion jobs.
-- Used to support non-blocking file upload and background processing.

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id                  BIGSERIAL PRIMARY KEY,
    job_id              VARCHAR(36) UNIQUE NOT NULL,           -- UUID for client tracking
    module              VARCHAR(50) NOT NULL,                  -- credit, investment, insurance, retirement, taxation
    filename            VARCHAR(500),                          -- original uploaded filename
    status              VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN (
                            'pending', 'processing', 'completed', 'failed', 'cancelled'
                        )),
    
    -- Job data
    metadata            JSONB,                                 -- ingestion metadata (doc_type, year, issuer, etc.)
    file_path           TEXT,                                  -- temp file path for worker to process
    file_size           BIGINT,                                -- file size in bytes
    
    -- Results
    result              JSONB,                                 -- ingestion result when completed
    error_message       TEXT,                                  -- error details if failed
    error_traceback     TEXT,                                  -- full traceback for debugging
    
    -- Progress tracking
    progress_percent    INTEGER DEFAULT 0,                     -- 0-100
    progress_message    TEXT,                                  -- "Parsing page 5/10", etc.
    
    -- Related document
    document_id         BIGINT REFERENCES documents(id) ON DELETE SET NULL,  -- linked document if completed
    
    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,                          -- when processing started
    completed_at        TIMESTAMPTZ,                          -- when finished (success or fail)
    
    -- Worker info (optional, for debugging)
    worker_id           TEXT,                                 -- which worker processed this job
    retry_count         INTEGER DEFAULT 0,                    -- number of retries
    
    CONSTRAINT ingestion_jobs_progress_ck CHECK (progress_percent BETWEEN 0 AND 100)
);

COMMENT ON TABLE ingestion_jobs IS 'Tracks async document ingestion jobs for non-blocking file uploads';

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS ingestion_jobs_job_id_idx ON ingestion_jobs(job_id);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS ingestion_jobs_module_idx ON ingestion_jobs(module);
CREATE INDEX IF NOT EXISTS ingestion_jobs_created_at_idx ON ingestion_jobs(created_at);
CREATE INDEX IF NOT EXISTS ingestion_jobs_document_id_idx ON ingestion_jobs(document_id);

-- Trigger to maintain updated_at on ingestion_jobs (reuse existing function)
-- Note: Need to add updated_at column first
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE TRIGGER ingestion_jobs_updated_at_tr
BEFORE UPDATE ON ingestion_jobs
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

