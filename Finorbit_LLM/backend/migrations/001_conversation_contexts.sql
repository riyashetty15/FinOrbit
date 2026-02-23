-- Migration 001: Conversation Context Persistence
-- Stores per-conversation state (last agent, user profile, turn count)
-- with a 24-hour sliding TTL enforced at query time.

CREATE TABLE IF NOT EXISTS conversation_contexts (
    conversation_id  TEXT PRIMARY KEY,
    last_agent       TEXT NOT NULL,
    profile_json     JSONB NOT NULL DEFAULT '{}',
    turn_count       INTEGER NOT NULL DEFAULT 1,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cc_updated_at
    ON conversation_contexts(updated_at);
