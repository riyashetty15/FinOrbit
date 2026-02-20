"""
Sample document ingestion script for testing RAG pipeline
Ingests the Financial Literacy Guide into multiple modules
"""

import os
import sys
import asyncio
import httpx
from pathlib import Path

# RAG server URL
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8081")

# Sample documents to ingest
DOCUMENTS = [
    {
        "file_path": "Financial Literacy Guide.pdf",
        "module": "investment",
        "doc_type": "guide",
        "year": 2024,
        "issuer": "Financial Authority"
    },
    {
        "file_path": "Financial Literacy Guide.pdf",
        "module": "credit",
        "doc_type": "guide",
        "year": 2024,
        "issuer": "Financial Authority"
    },
    {
        "file_path": "Financial Literacy Guide.pdf",
        "module": "insurance",
        "doc_type": "guide",
        "year": 2024,
        "issuer": "Financial Authority"
    },
]


async def ingest_document(doc: dict):
    """Ingest a single document into RAG system"""
    
    file_path = Path(doc["file_path"])
    
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False
    
    print(f"\n[INGESTING] {file_path.name} → {doc['module']} module")
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/pdf')}
                data = {
                    'module': doc['module'],
                    'doc_type': doc.get('doc_type', 'guide'),
                    'year': str(doc.get('year', 2024)),
                    'issuer': doc.get('issuer', 'Unknown'),
                    'jurisdiction': doc.get('jurisdiction', 'India')
                }
                
                response = await client.post(
                    f"{RAG_API_URL}/ingest",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[SUCCESS] Ingested: {result.get('chunks_created', 0)} chunks created")
                    print(f"          Job ID: {result.get('job_id', 'N/A')}")
                    return True
                else:
                    print(f"[ERROR] HTTP {response.status_code}: {response.text[:200]}")
                    return False
                    
    except Exception as e:
        print(f"[ERROR] Failed to ingest: {str(e)}")
        return False


async def test_retrieval(module: str, query: str):
    """Test retrieval after ingestion"""
    
    print(f"\n[TEST RETRIEVAL] Module: {module}, Query: {query[:50]}...")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_API_URL}/query",
                json={
                    "query": query,
                    "module": module,
                    "top_k": 3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                chunks = result.get('chunks', [])
                
                print(f"[SUCCESS] Retrieved {len(chunks)} chunks")
                
                if chunks:
                    print(f"\nTop result:")
                    top_chunk = chunks[0]
                    print(f"  Score: {top_chunk.get('score', 0):.4f}")
                    print(f"  Source: {top_chunk.get('metadata', {}).get('source', 'N/A')}")
                    print(f"  Text preview: {top_chunk.get('text', '')[:150]}...")
                else:
                    print("[WARNING] No chunks retrieved")
                    
                return len(chunks) > 0
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text[:200]}")
                return False
                
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {str(e)}")
        return False


async def main():
    """Main ingestion workflow"""
    
    print("="*80)
    print("RAG DOCUMENT INGESTION SCRIPT")
    print("="*80)
    
    # Change to RAG directory
    os.chdir(Path(__file__).parent.parent.parent / "Finorbit_RAG")
    print(f"\nWorking directory: {os.getcwd()}")
    
    # Check RAG server health
    print(f"\nChecking RAG server at {RAG_API_URL}...")
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
        print("Make sure RAG server is running: python main.py")
        return
    
    # Ingest documents
    print(f"\n{'='*80}")
    print(f"INGESTING {len(DOCUMENTS)} DOCUMENT(S)")
    print(f"{'='*80}")
    
    success_count = 0
    for doc in DOCUMENTS:
        if await ingest_document(doc):
            success_count += 1
        await asyncio.sleep(2)  # Brief pause between ingestions
    
    print(f"\n{'='*80}")
    print(f"INGESTION COMPLETE: {success_count}/{len(DOCUMENTS)} successful")
    print(f"{'='*80}")
    
    # Test retrieval
    if success_count > 0:
        print(f"\n{'='*80}")
        print("TESTING RETRIEVAL")
        print(f"{'='*80}")
        
        test_queries = [
            ("investment", "What are mutual funds?"),
            ("credit", "How to improve credit score?"),
            ("insurance", "What is term insurance?"),
        ]
        
        for module, query in test_queries:
            await test_retrieval(module, query)
            await asyncio.sleep(1)
    
    print(f"\n{'='*80}")
    print("✓ Done! RAG database is now populated with sample documents.")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Run evaluation: python tests/run_evaluation.py")
    print("  2. Check citations are now returning in responses")
    print("  3. Verify evidence coverage scoring is working")


if __name__ == "__main__":
    asyncio.run(main())
