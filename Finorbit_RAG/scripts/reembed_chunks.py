#!/usr/bin/env python3
"""Recompute embeddings for existing chunk rows and update DB.

This script reads text chunks from each module chunk table, computes
embeddings using the configured embedding model (sentence-transformers),
and updates the `embedding` column so stored vectors match the current model.

Usage:
  python scripts/reembed_chunks.py --batch-size 128
  python scripts/reembed_chunks.py --modules credit,insurance --batch-size 64
"""
import argparse
import logging
from math import ceil
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor

from config import get_database_config, get_embedding_config, MODULE_TABLES

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


logger = logging.getLogger("reembed")
logging.basicConfig(level=logging.INFO)


def chunked(iterable: List, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def reembed_table(conn_params, table_name: str, model_name: str, batch_size: int, device: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required to run re-embed script")

    model = SentenceTransformer(model_name)
    if device and device != "cpu":
        try:
            model.to(device)
        except Exception:
            logger.warning("Could not set model device, proceeding with default")

    # Database operations wrapped with reconnect/retry logic to be resilient in hackathon envs
    max_conn_attempts = 3
    for attempt in range(1, max_conn_attempts + 1):
        try:
            conn = psycopg2.connect(**conn_params)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            break
        except Exception as e:
            logger.warning("DB connect attempt %s/%s failed: %s", attempt, max_conn_attempts, e)
            if attempt == max_conn_attempts:
                raise
    try:
        cur.execute(f"SELECT id, text FROM {table_name} WHERE text IS NOT NULL")
        rows = cur.fetchall()
        total = len(rows)
        logger.info(f"Table {table_name}: {total} chunks to re-embed")

        for batch in chunked(rows, batch_size):
            texts = [r["text"] for r in batch]
            try:
                emb = model.encode(texts, show_progress_bar=False)
            except Exception as e:
                logger.exception("Embedding model failed on batch; skipping batch: %s", e)
                continue

            for r, e in zip(batch, emb):
                try:
                    vec = e.tolist() if hasattr(e, "tolist") else list(e)
                    cur.execute(
                        f"UPDATE {table_name} SET embedding=%s, embedding_model=%s, embedding_model_version=%s, embedding_created_at=NOW() WHERE id=%s",
                        (vec, model_name, "1.0", r["id"]),
                    )
                except Exception:
                    logger.exception("Failed updating row id=%s in %s; continuing", r["id"], table_name)
                    continue

            conn.commit()
            logger.info(f"Updated {min(batch_size, total)} rows in {table_name}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", help="Comma-separated module names to reembed (default: all)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    db_cfg = get_database_config()
    emb_cfg = get_embedding_config()

    conn_params = db_cfg.to_dict()

    selected_tables = list(MODULE_TABLES.values())
    if args.modules:
        names = [m.strip() for m in args.modules.split(",") if m.strip()]
        selected_tables = [MODULE_TABLES[m] for m in names if m in MODULE_TABLES]

    for table in selected_tables:
        reembed_table(conn_params, table, emb_cfg.model_name, args.batch_size, emb_cfg.device)


if __name__ == "__main__":
    main()
