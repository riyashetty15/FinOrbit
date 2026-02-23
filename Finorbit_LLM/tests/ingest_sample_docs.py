"""
Sample document ingestion script for testing RAG pipeline.

Uses three different PDFs (one per module) to avoid duplicate content
in the vector store. File paths are absolute so os.chdir() is never needed.
"""

import asyncio
import httpx
import os
from pathlib import Path

# RAG server URL
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8081")

# Absolute paths — derived from this file's location, no os.chdir() needed.
_REPO_ROOT = Path(__file__).parent.parent.parent  # …/FinOrbit/

# Each module gets a DISTINCT source document to avoid vector-store bloat.
DOCUMENTS = [
    {
        "file_path": str(_REPO_ROOT / "Financial Education Booklet - English.pdf"),
        "module": "investment",
        "doc_type": "guide",
        "year": 2024,
        "issuer": "Financial Authority",
        "jurisdiction": "India",
    },
    {
        "file_path": str(_REPO_ROOT / "Financial-Literacy-Toolkit-Trainers-Guide.pdf"),
        "module": "credit",
        "doc_type": "guide",
        "year": 2024,
        "issuer": "Financial Authority",
        "jurisdiction": "India",
    },
    {
        "file_path": str(_REPO_ROOT / "RBI-MASTER-DIRECTION-NBFC-19-10-2023.pdf"),
        "module": "credit",
        "doc_type": "master_direction",
        "year": 2023,
        "issuer": "RBI",
        "jurisdiction": "India",
    },
]


async def ingest_document(doc: dict, max_retries: int = 3) -> bool:
    """Ingest a single document into the RAG system with retry / exponential backoff."""
    file_path = Path(doc["file_path"])

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    for attempt in range(1, max_retries + 1):
        print(f"\n[INGESTING] {file_path.name} → {doc['module']} module (attempt {attempt}/{max_retries})")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(file_path, "rb") as f:
                    files = {"file": (file_path.name, f, "application/pdf")}
                    data = {
                        "module": doc["module"],
                        "doc_type": doc.get("doc_type", "guide"),
                        "year": str(doc.get("year", 2024)),
                        "issuer": doc.get("issuer", "Unknown"),
                        "jurisdiction": doc.get("jurisdiction", "India"),
                    }
                    response = await client.post(
                        f"{RAG_API_URL}/ingest",
                        files=files,
                        data=data,
                    )

            if response.status_code == 200:
                result = response.json()
                print(f"[SUCCESS] Ingested: {result.get('chunks_created', 0)} chunks created")
                print(f"          Job ID: {result.get('job_id', 'N/A')}")
                return True

            print(f"[ERROR] HTTP {response.status_code}: {response.text[:200]}")
            # Do not retry on client errors (4xx)
            if response.status_code < 500:
                return False

        except Exception as e:
            print(f"[ERROR] Attempt {attempt} failed: {e}")

        if attempt < max_retries:
            wait = 2 ** attempt  # 2 s, 4 s
            print(f"[RETRY] Waiting {wait}s before retry…")
            await asyncio.sleep(wait)

    print(f"[FAILED] All {max_retries} attempts exhausted for {file_path.name}")
    return False


async def test_retrieval(module: str, query: str) -> bool:
    """Test retrieval after ingestion."""
    print(f"\n[TEST RETRIEVAL] Module: {module}, Query: {query[:50]}…")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_API_URL}/query",
                json={"query": query, "module": module, "top_k": 3},
            )

        if response.status_code == 200:
            result = response.json()
            chunks = result.get("chunks", [])
            print(f"[SUCCESS] Retrieved {len(chunks)} chunks")
            if chunks:
                top = chunks[0]
                print(f"  Score: {top.get('score', 0):.4f}")
                print(f"  Source: {top.get('metadata', {}).get('source', 'N/A')}")
                print(f"  Text preview: {top.get('text', '')[:150]}…")
            else:
                print("[WARNING] No chunks retrieved")
            return len(chunks) > 0

        print(f"[ERROR] HTTP {response.status_code}: {response.text[:200]}")
        return False

    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return False


async def main():
    """Main ingestion workflow."""
    print("=" * 80)
    print("RAG DOCUMENT INGESTION SCRIPT")
    print("=" * 80)

    # --- Validate all file paths BEFORE connecting to anything ---
    missing = [doc["file_path"] for doc in DOCUMENTS if not Path(doc["file_path"]).exists()]
    if missing:
        print("\n[ERROR] The following files are missing:")
        for f in missing:
            print(f"  - {f}")
        print("Aborting. Fix file paths before running.")
        return
    print(f"\n[OK] All {len(DOCUMENTS)} document(s) found")

    # --- Check RAG server health ---
    print(f"\nChecking RAG server at {RAG_API_URL}…")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{RAG_API_URL}/health")
        if response.status_code == 200:
            print("[OK] RAG server is healthy")
        else:
            print(f"[ERROR] RAG server returned {response.status_code}")
            return
    except Exception as e:
        print(f"[ERROR] Cannot connect to RAG server: {e}")
        print("Make sure the RAG server is running: cd Finorbit_RAG && python main.py")
        return

    # --- Ingest documents ---
    print(f"\n{'=' * 80}")
    print(f"INGESTING {len(DOCUMENTS)} DOCUMENT(S)")
    print(f"{'=' * 80}")

    success_count = 0
    for doc in DOCUMENTS:
        if await ingest_document(doc):
            success_count += 1
        await asyncio.sleep(2)  # Brief pause between ingestions

    print(f"\n{'=' * 80}")
    print(f"INGESTION COMPLETE: {success_count}/{len(DOCUMENTS)} successful")
    print(f"{'=' * 80}")

    # --- Test retrieval ---
    if success_count > 0:
        print(f"\n{'=' * 80}")
        print("TESTING RETRIEVAL")
        print(f"{'=' * 80}")

        test_queries = [
            ("investment", "What are mutual funds?"),
            ("credit", "How to improve credit score?"),
            ("credit", "What are NBFC NPA classification rules?"),
        ]

        for module, query in test_queries:
            await test_retrieval(module, query)
            await asyncio.sleep(1)

    print(f"\n{'=' * 80}")
    print("Done! RAG database is now populated with sample documents.")
    print(f"{'=' * 80}")
    print("\nNext steps:")
    print("  1. Run evaluation: python tests/run_evaluation.py")
    print("  2. Check citations are returning in responses")
    print("  3. Verify evidence coverage scoring is working")


if __name__ == "__main__":
    asyncio.run(main())
