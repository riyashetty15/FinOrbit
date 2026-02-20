#!/usr/bin/env python3
"""Simple config validator to ensure required env vars are set."""
import sys
from config import get_database_config, get_llamaindex_config
import os


SENSITIVE_KEYS = [
    "DB_PASSWORD",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "LLAMAPARSE_API_KEY",
]


def main():
    db = get_database_config()
    li = get_llamaindex_config()

    missing = []
    if not db.password:
        missing.append('DB_PASSWORD')
    if li.llm_provider == 'gemini' and not li.google_api_key:
        missing.append('GOOGLE_API_KEY')
    if not li.embedding_dim:
        missing.append('EMBEDDING_DIMENSION')

    if missing:
        print('Missing required environment variables:', ', '.join(missing))
        sys.exit(2)
    # Warn if sensitive keys found in local `.env` file (common accident)
    # This is a best-effort check: if `.env` exists in repo root and contains keys, warn.
    if os.path.exists('.env'):
        with open('.env') as fh:
            content = fh.read()
        leaked = [k for k in SENSITIVE_KEYS if k in content]
        if leaked:
            print('WARNING: Found possible secrets in `.env`:', ', '.join(leaked))
            print('Please remove them from the repo and rotate keys before sharing.')

    print('Config looks OK')


if __name__ == '__main__':
    main()
