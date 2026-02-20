import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.retrieval_service import RetrievalService

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_service():
    print("Initializing Retrieval Service...")
    service = RetrievalService()
    
    query = "What is the objective of the RBI Master Direction for NBFCs?"
    print(f"\nTesting Query: {query}")
    
    # Test Module Determination
    module = service.determine_module(query)
    print(f"Determined Module: {module}")
    
    # Test Retrieval
    print("Calling Retrieve Evidence (this talks to the real RAG server)...")
    try:
        evidence_pack = await service.retrieve_evidence(query, module=module)
        
        print("\n--- Evidence Pack Result ---")
        print(f"Confidence: {evidence_pack.confidence}")
        print(f"Sufficient: {evidence_pack.sufficient}")
        print(f"Contexts Found: {len(evidence_pack.answer_context)}")
        
        if evidence_pack.answer_context:
            first = evidence_pack.answer_context[0]
            print(f"First Chunk Source: {first.source}")
            print(f"First Chunk Text: {first.text[:100]}...")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    asyncio.run(test_service())
