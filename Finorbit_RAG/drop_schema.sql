-- drop_schema.sql
-- Script to drop all existing RAG system tables
-- ⚠️ WARNING: This will delete ALL data! Backup first if needed.
--
-- This script drops all tables, functions, and triggers created by schema.sql
-- It handles both current schema and legacy table names for backward compatibility

-- Connect to your database first
-- \c financial_rag

-- ============================================================================
-- BACKUP REMINDER
-- ============================================================================
-- Before running this script, create a backup:
-- pg_dump -U postgres financial_rag > backup_$(date +%Y%m%d_%H%M%S).sql
-- ============================================================================

-- ============================================================================
-- Drop Triggers (must be dropped before tables)
-- ============================================================================

DROP TRIGGER IF EXISTS documents_updated_at_tr ON documents CASCADE;

-- ============================================================================
-- Drop Module Chunk Tables (LlamaIndex tables with data_ prefix)
-- ============================================================================
-- These are the current tables created by schema.sql
-- Must be dropped before documents table due to foreign key constraints

DROP TABLE IF EXISTS data_credit_chunks CASCADE;
DROP TABLE IF EXISTS data_investment_chunks CASCADE;
DROP TABLE IF EXISTS data_insurance_chunks CASCADE;
DROP TABLE IF EXISTS data_retirement_chunks CASCADE;
DROP TABLE IF EXISTS data_tax_chunks CASCADE;

-- ============================================================================
-- Drop Legacy Chunk Tables (if they exist from older versions)
-- ============================================================================
-- These may exist from previous schema versions

DROP TABLE IF EXISTS credit_chunks CASCADE;
DROP TABLE IF EXISTS investment_chunks CASCADE;
DROP TABLE IF EXISTS insurance_chunks CASCADE;
DROP TABLE IF EXISTS retirement_chunks CASCADE;
DROP TABLE IF EXISTS tax_chunks CASCADE;

-- ============================================================================
-- Drop Document Relationship Table
-- ============================================================================
-- Drop before documents table to handle foreign keys

DROP TABLE IF EXISTS document_modules CASCADE;

-- ============================================================================
-- Drop Main Documents Table
-- ============================================================================
-- This will cascade to any remaining chunk tables via foreign keys

DROP TABLE IF EXISTS documents CASCADE;

-- ============================================================================
-- Drop Functions
-- ============================================================================

DROP FUNCTION IF EXISTS set_updated_at() CASCADE;

-- ============================================================================
-- Optional: Drop pgvector Extension
-- ============================================================================
-- Uncomment the line below if you want to completely remove pgvector extension
-- WARNING: This will remove vector support from the entire database!
-- DROP EXTENSION IF EXISTS vector CASCADE;

-- ============================================================================
-- Verify Cleanup
-- ============================================================================
-- Check that all RAG system tables have been dropped

SELECT tablename 
FROM pg_tables 
WHERE schemaname = 'public' 
  AND (
    tablename LIKE '%chunk%' 
    OR tablename = 'documents' 
    OR tablename = 'document_modules'
  );

-- Expected: No rows returned (all tables should be dropped)

-- ============================================================================
-- Final Comment
-- ============================================================================

COMMENT ON SCHEMA public IS 'All RAG system tables have been dropped';

