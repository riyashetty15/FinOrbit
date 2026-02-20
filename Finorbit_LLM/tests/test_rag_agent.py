"""
Test script for RAG Agent integration with orchestrator
Tests that the router correctly routes queries to RAG agent
"""

import asyncio
import httpx
import json


async def test_rag_agent_routing():
    """Test the RAG agent routing and execution"""
    
    # Test endpoint
    query_url = "http://localhost:8000/query"
    
    # Test queries that should route to RAG agent
    test_cases = [
        {
            "name": "Direct knowledge base query",
            "payload": {
                "userId": "test_user_001",
                "conversationId": "test_conv_rag_001",
                "query": "what are the guidelines of parag parekh flexi cap fund for the year 2025?"
            }
        },
        {
            "name": "Document lookup query",
            "payload": {
                "userId": "test_user_002",
                "conversationId": "test_conv_rag_002",
                "query": "Find RBI regulations about credit scoring"
            }
        },
        {
            "name": "Guideline query",
            "payload": {
                "userId": "test_user_003",
                "conversationId": "test_conv_rag_003",
                "query": "What are the IRDAI guidelines for term insurance?"
            }
        },
        {
            "name": "Knowledge base lookup",
            "payload": {
                "userId": "test_user_004",
                "conversationId": "test_conv_rag_004",
                "query": "Lookup tax rules under Section 80C"
            }
        },
        {
            "name": "General investment query (should still work)",
            "payload": {
                "userId": "test_user_005",
                "conversationId": "test_conv_rag_005",
                "query": "What are mutual funds and how do they work?"
            }
        }
    ]
    
    print("=" * 80)
    print("RAG AGENT ROUTING TEST")
    print("=" * 80)
    print()
    
    async with httpx.AsyncClient(timeout=60) as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'=' * 80}")
            print(f"Test {i}: {test_case['name']}")
            print(f"{'=' * 80}")
            print(f"Query: {test_case['payload']['query']}")
            print()
            
            try:
                response = await client.post(
                    query_url,
                    json=test_case['payload']
                )
                
                print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Success!")
                    print(f"\nAgent Type: {result.get('agent_type', 'N/A')}")
                    print(f"Confidence Score: {result.get('confidence_score', 0):.2f}")
                    print(f"Needs Clarification: {result.get('needs_clarification', False)}")
                    print(f"\nResponse Preview:")
                    response_text = result.get('response', '')
                    print(f"{response_text[:500]}...")
                    
                    # Check if it routed to RAG agent
                    if result.get('agent_type') == 'rag_agent':
                        print(f"\nCorrectly routed to RAG Agent")
                    else:
                        print(f"\nRouted to {result.get('agent_type')} instead of rag_agent")
                    
                elif response.status_code == 400:
                    error = response.json()
                    print(f"Validation Error:")
                    print(f"Error: {error.get('detail', {}).get('error', 'Unknown')}")
                    print(f"Type: {error.get('detail', {}).get('error_type', 'Unknown')}")
                    
                else:
                    print(f"[ERROR] Request failed")
                    print(f"Response: {response.text[:500]}")
                    
            except httpx.ConnectError:
                print(f"[ERROR] Connection Error: Cannot connect to {query_url}")
                print(f"   Make sure the backend server is running on port 8000")
                break
                
            except httpx.ReadTimeout:
                print(f"[WARNING] Request timed out (may take longer for first query)")
                
            except Exception as e:
                print(f"[ERROR] Error: {str(e)}")
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


async def test_direct_rag_agent():
    """Test the RAG agent directly"""
    print("\n" + "=" * 80)
    print("DIRECT RAG AGENT TEST")
    print("=" * 80)
    
    try:
        from backend.agents.rag.rag_agent import RAGAgent
        
        agent = RAGAgent()
        
        test_query = {
            "query": "What are the guidelines for mutual fund investments?",
            "profile": {
                "user_id": "test_user",
                "age": 30,
                "income": 1000000
            },
            "intent": "general"
        }
        
        print(f"\nTest Query: {test_query['query']}")
        print("\nExecuting RAG Agent...")
        
        result = await agent.run(test_query)
        
        print(f"\n[OK] RAG Agent executed successfully")
        print(f"\nSummary:\n{result.get('summary', 'N/A')}")
        
        if result.get('sources'):
            print(f"\nSources:")
            for source in result['sources']:
                print(f"  - {source}")
        
        if result.get('next_best_actions'):
            print(f"\nNext Best Actions:")
            for action in result['next_best_actions']:
                print(f"  • {action}")
        
    except ImportError as e:
        print(f"[ERROR] Import Error: {e}")
        print("   Make sure the backend is properly set up")
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
    
    print()


if __name__ == "__main__":
    print("\n[START] Starting RAG Agent Integration Test...")
    print("\nPrerequisites:")
    print("  1. Backend server running (uvicorn backend.server:app --reload)")
    print("  2. RAG server running on configured port")
    print("  3. RAG_API_URL set in .env file")
    print()
    
    # Test direct agent first
    asyncio.run(test_direct_rag_agent())
    
    # Then test routing
    input("\nPress Enter to test RAG agent routing through orchestrator...")
    asyncio.run(test_rag_agent_routing())
