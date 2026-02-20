"""
Test to verify citation extraction is working
"""

import asyncio
import httpx
import json


async def test_citation_extraction():
    """Test that citations are being extracted from RAG responses"""
    
    print("\n" + "="*80)
    print("CITATION EXTRACTION TEST")
    print("="*80)
    
    # Test queries that should return citations
    test_cases = [
        {
            "query": "What are SEBI mutual fund regulations?",
            "expected_module": "investment",
            "should_have_citations": True
        },
        {
            "query": "What are RBI credit lending guidelines?",
            "expected_module": "credit",
            "should_have_citations": True
        },
        {
            "query": "What is mutual fund?",  # General query
            "expected_module": "investment",
            "should_have_citations": False
        }
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, test in enumerate(test_cases, 1):
            print(f"\n[TEST {i}] {test['query']}")
            print("-" * 80)
            
            try:
                response = await client.post(
                    "http://localhost:8000/query",
                    json={
                        "userId": f"test_user_{i}",
                        "conversationId": f"test_conv_{i}",
                        "query": test['query']
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check for sources/citations
                    sources = result.get('sources', [])
                    agents = result.get('agents', [])
                    pipeline_steps = result.get('pipeline_steps', [])
                    response_text = result.get('response', '')
                    
                    print(f"\nAgents used: {', '.join(agents)}")
                    print(f"Sources/Citations: {len(sources) if sources else 0}")
                    
                    if sources:
                        print(f"\nCitation details:")
                        for idx, source in enumerate(sources[:3], 1):
                            print(f"  [{idx}] Source: {source.get('source', source.get('document', 'N/A'))}")
                            print(f"      Score: {source.get('score', 0):.4f}")
                            print(f"      Type: {source.get('type', 'unknown')}")
                            if source.get('excerpt'):
                                print(f"      Excerpt: {source['excerpt'][:100]}...")
                    else:
                        print(f"\n⚠️  No citations found")
                    
                    # Check pipeline steps
                    print(f"\nPipeline steps:")
                    for step in pipeline_steps:
                        print(f"  - {step.get('step')}: {step.get('status')}")
                    
                    # Check if evidence retrieval happened  
                    has_evidence_step = any('Evidence' in step.get('step', '') for step in pipeline_steps)
                    
                    # Verify expectations
                    if test['should_have_citations']:
                        if sources and len(sources) > 0:
                            print(f"\n✓ PASS: Citations found as expected")
                        else:
                            print(f"\n✗ FAIL: Expected citations but found none")
                            print(f"   Response preview: {response_text[:200]}...")
                    else:
                        if not sources:
                            print(f"\n✓ PASS: No citations (general query)")
                        else:
                            print(f"\n⚠️  UNEXPECTED: Citations found for general query")
                    
                else:
                    print(f"✗ HTTP {response.status_code}: {response.text[:200]}")
                    
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_citation_extraction())
