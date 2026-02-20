-- schema.sql
-- Modular RAG database schema for LlamaIndex + PostgreSQL (pgvector)

-- Create database (optional; run with superuser privileges)
-- CREATE DATABASE rag_system;
-- \c rag_system

-- Enable pgvector extension for embedding support
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
-- Helper function to create chunk tables per module
-------------------------------------------------------------------------------

-- The following DO block iterates over the module chunk tables and creates
-- identical schemas with module-specific defaults and indexes.

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
                id                      BIGSERIAL PRIMARY KEY,
                document_id             BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index             INTEGER NOT NULL,
                parent_id               BIGINT,
                text                    TEXT NOT NULL,
                page_number             INTEGER,

                embedding               vector(768),
                embedding_model         TEXT,
                embedding_model_version TEXT,
                embedding_created_at    TIMESTAMPTZ,

                metadata                JSONB DEFAULT '{}'::JSONB,
                metadata_               JSONB DEFAULT '{}'::JSONB,
                node_id                 TEXT,
                module                  TEXT NOT NULL DEFAULT %L,
                language                TEXT,
                ocr_processed           BOOLEAN DEFAULT FALSE,
                quality_flags           JSONB,

                token_count             INTEGER,
                char_count              INTEGER,
                word_count              INTEGER,
                chunk_hash              TEXT,
                pipeline_version        TEXT,

                created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                tsv                     TSVECTOR GENERATED ALWAYS AS (
                                            to_tsvector('english', coalesce(text, ''))
                                        ) STORED,

                CONSTRAINT %I_module_ck CHECK (module = %L),
                CONSTRAINT %I_parent_fk FOREIGN KEY (parent_id) REFERENCES %I(id) ON DELETE CASCADE,
                UNIQUE (document_id, chunk_index),
                UNIQUE (document_id, chunk_hash)
            );
        $ddl$, chunk_table, module_name, chunk_table, module_name, chunk_table, chunk_table);

        -- Embedding index for similarity search
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_embedding_idx
            ON %I USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        $ddl$, chunk_table, chunk_table);

        -- Supporting indexes for fast filtering
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_doc_chunk_idx ON %I (document_id, chunk_index);
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_created_at_idx ON %I (created_at);
        $ddl$, chunk_table, chunk_table);

        -- Optional JSONB and expression indexes for metadata-rich queries
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_metadata_idx
            ON %I USING gin (metadata);
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_doc_type_idx
            ON %I ((metadata->>'doc_type'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_year_idx
            ON %I ((metadata->>'year'));
        $ddl$, chunk_table, chunk_table);

        -- Additional metadata indexes required for enhanced filtering
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_module_idx
            ON %I ((metadata->>'module'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_issuer_idx
            ON %I ((metadata->>'issuer'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_language_idx
            ON %I ((metadata->>'language'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_regulator_tag_idx
            ON %I ((metadata->>'regulator_tag'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_security_idx
            ON %I ((metadata->>'security'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_version_id_idx
            ON %I ((metadata->>'version_id'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_is_current_idx
            ON %I ((metadata->>'is_current'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_pii_idx
            ON %I ((metadata->>'pii'));
        $ddl$, chunk_table, chunk_table);

        -- Array-type metadata fields
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_compliance_tags_idx
            ON %I USING GIN ((metadata->'compliance_tags'));
        $ddl$, chunk_table, chunk_table);

        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_tags_array_idx
            ON %I USING GIN ((metadata->'tags'));
        $ddl$, chunk_table, chunk_table);

        -- Optional keyword search support for hybrid retrieval
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_tsv_idx
            ON %I USING gin (tsv);
        $ddl$, chunk_table, chunk_table);

        -- Index for LlamaIndex node_id compatibility
        EXECUTE format($ddl$
            CREATE INDEX IF NOT EXISTS %I_node_id_idx ON %I (node_id);
        $ddl$, chunk_table, chunk_table);
    END LOOP;
END $$;



