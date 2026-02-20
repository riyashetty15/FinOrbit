"""
Test script for RAG tool integration
Tests the knowledge_lookup MCP tool endpoint
"""

import asyncio
import httpx
import json


async def test_rag_tool():
    """Test the RAG knowledge lookup tool"""
    
    # Test endpoint
    tool_url = "http://localhost:8000/tools/knowledge_lookup"
    
    # Test queries
    test_cases = [
        {
            "name": "Investment query",
            "payload": {
                "query": "What are mutual funds?",
                "module": "investment",
                "top_k": 3
            }
        },
        {
            "name": "Tax planning query",
            "payload": {
                "query": "What are tax deductions under Section 80C?",
                "module": "taxation",
                "top_k": 3
            }
        },
        {
            "name": "Credit query",
            "payload": {
                "query": "How to improve credit score?",
                "module": "credit",
                "top_k": 2
            }
        },
        {
            "name": "RAG Query",
            "payload": {
                "query": "who is the sponsor of parag parekh flexi cap fund?",
                "module": "investment",
                "top_k": 2
            }
        },
        {
            "name": "Invalid module test",
            "payload": {
                "query": "Test query",
                "module": "invalid_module",
                "top_k": 2
            }
        }
    ]
    
    print("=" * 80)
    print("RAG TOOL INTEGRATION TEST")
    print("=" * 80)
    print()
    
    async with httpx.AsyncClient(timeout=30) as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'=' * 80}")
            print(f"Test {i}: {test_case['name']}")
            print(f"{'=' * 80}")
            print(f"Query: {test_case['payload']['query']}")
            print(f"Module: {test_case['payload']['module']}")
            print()
            
            try:
                response = await client.post(
                    tool_url,
                    params=test_case['payload']
                )
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Success!")
                    print(f"Found: {result.get('found', False)}")
                    print(f"Total Chunks: {result.get('total_chunks', 0)}")
                    
                    if result.get('results'):
                        print(f"\nResults:")
                        for idx, chunk in enumerate(result['results'], 1):
                            print(f"\n  Chunk {idx}:")
                            print(f"    Document: {chunk.get('document_filename', 'N/A')}")
                            print(f"    Similarity Score: {chunk.get('similarity_score', 0):.4f}")
                            print(f"    Content Preview: {chunk.get('content', '')[:150]}...")
                    else:
                        print(f"\nNo results found")
                        
                    if result.get('error'):
                        print(f"\nError: {result['error']}")
                        
                else:
                    print(f"[ERROR] Request failed")
                    print(f"Response: {response.text[:500]}")
                    
            except httpx.ConnectError:
                print(f"[ERROR] Connection Error: Cannot connect to {tool_url}")
                print(f"   Make sure the server is running on port 8000")
                break
                
            except Exception as e:
                print(f"[ERROR] Error: {str(e)}")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    print("\nStarting RAG Tool Test...")
    print("Make sure:")
    print("  1. Backend server is running (uvicorn backend.server:app --reload)")
    print("  2. RAG server is running on the configured port")
    print("  3. RAG_API_URL is set in .env file")
    print()
    
    asyncio.run(test_rag_tool())
