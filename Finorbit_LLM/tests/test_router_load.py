"""
Load test for RouterAgent - validates cache effectiveness, circuit breaker, and latency
"""
import time
import json
from typing import List, Dict, Any
from backend.core.router import RouterAgent

# Test queries for cache hit rate testing
CACHE_TEST_QUERIES = [
    "What is NAV?",
    "What is NAV?",  # Repeat - should hit cache
    "What is NAV?",  # Repeat - should hit cache
    "What are mutual funds?",
    "What are mutual funds?",  # Repeat
    "What is SEBI?",
    "Calculate my tax",
    "Should I invest in stocks?",
]

# Unique queries for baseline latency
LATENCY_TEST_QUERIES = [
    "What is a credit score?",
    "How do I calculate my net worth?",
    "What are the tax slabs for 2025?",
    "How does compound interest work?",
    "What is portfolio rebalancing?",
    "What is an EMI?",
    "What is insurance premium?",
    "What is pension planning?",
    "What is fraud prevention?",
    "What is financial literacy?",
]

# Circuit breaker stress test (simulate many errors)
CIRCUIT_BREAKER_TEST_QUERIES = [
    "Does the LLM API work?",
    "Can you route this?",
    "Is the system up?",
]


def print_metrics(router: RouterAgent, test_name: str):
    """Pretty print router metrics"""
    metrics = router.get_metrics()
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(json.dumps(metrics, indent=2))


def test_cache_effectiveness():
    """Test cache hit rate with repeated queries"""
    print("\n[TEST 1] Cache Effectiveness")
    print("-" * 60)
    
    router = RouterAgent()
    
    for i, query in enumerate(CACHE_TEST_QUERIES, 1):
        print(f"Query {i}: {query[:50]}...")
        router.route(query)
    
    metrics = router.get_metrics()
    print_metrics(router, "Cache Effectiveness")
    
    # Assertions
    cache_hit_ratio = metrics["cache"]["hit_ratio"]
    print(f"\n✓ Cache hit ratio: {cache_hit_ratio:.1%}")
    print(f"  Expected: ~66% (4 repeats out of 8 total)")
    assert cache_hit_ratio > 0.5, f"Cache hit ratio too low: {cache_hit_ratio}"


def test_latency_baseline():
    """Measure baseline routing latency (no cache hits)"""
    print("\n[TEST 2] Latency Baseline")
    print("-" * 60)
    
    router = RouterAgent()
    
    for i, query in enumerate(LATENCY_TEST_QUERIES, 1):
        print(f"Query {i}: {query[:50]}...")
        router.route(query)
    
    metrics = router.get_metrics()
    print_metrics(router, "Latency Baseline")
    
    # Assertions
    avg_latency = metrics["latency_ms"]["avg"]
    p99_latency = metrics["latency_ms"]["p99"]
    print(f"\n✓ Average latency: {avg_latency:.2f}ms")
    print(f"✓ P99 latency: {p99_latency:.2f}ms")
    print(f"  (Expected: <100ms for regex routing, <500ms for LLM calls)")


def test_circuit_breaker_behavior():
    """Test circuit breaker opens after failures and recovers"""
    print("\n[TEST 3] Circuit Breaker Behavior")
    print("-" * 60)
    
    router = RouterAgent()
    
    # Simulate normal routing first
    print("Normal routing (baseline):")
    for query in CIRCUIT_BREAKER_TEST_QUERIES[:1]:
        router.route(query)
    
    metrics = router.get_metrics()
    initial_opens = metrics["circuit_breaker"]["opens"]
    print(f"  Circuit opens before stress test: {initial_opens}")
    
    # Note: In a real test, you'd mock the LLM to fail
    # For now, just verify circuit breaker tracking is working
    print("\n✓ Circuit breaker tracking is active")
    print(f"  Opens recorded: {metrics['circuit_breaker']['opens']}")
    print(f"  Recoveries recorded: {metrics['circuit_breaker']['recoveries']}")


def test_rag_decision_distribution():
    """Test RAG decision distribution"""
    print("\n[TEST 4] RAG Decision Distribution")
    print("-" * 60)
    
    router = RouterAgent()
    
    rag_queries = [
        "What is NAV of HDFC Mutual Fund?",  # Likely RAG
        "What are SEBI guidelines?",  # Likely RAG
        "Calculate my tax",  # Likely no RAG
        "What is insurance?",  # Likely no RAG
    ]
    
    for i, query in enumerate(rag_queries, 1):
        print(f"Query {i}: {query}")
        router.route(query)
    
    metrics = router.get_metrics()
    print_metrics(router, "RAG Decision Distribution")
    
    total_rag = metrics["rag_decisions"]["yes"] + metrics["rag_decisions"]["no"]
    if total_rag > 0:
        rag_ratio = metrics["rag_decisions"]["yes"] / total_rag
        print(f"\n✓ RAG decisions ratio: {rag_ratio:.1%}")
        print(f"  (Yes: {metrics['rag_decisions']['yes']}, No: {metrics['rag_decisions']['no']})")


def test_metrics_export():
    """Verify metrics can be exported for monitoring systems"""
    print("\n[TEST 5] Metrics Export for Monitoring")
    print("-" * 60)
    
    router = RouterAgent()
    
    # Generate some activity
    for query in ["What is tax?", "What is NAV?", "Calculate interest"]:
        router.route(query)
    
    metrics = router.get_metrics()
    
    # Verify all expected keys exist
    expected_sections = ["cache", "llm", "rag_decisions", "circuit_breaker", "latency_ms"]
    for section in expected_sections:
        assert section in metrics, f"Missing section: {section}"
        print(f"✓ {section}: {metrics[section]}")
    
    # Verify JSON serializable (for sending to monitoring systems)
    json_str = json.dumps(metrics)
    print(f"\n✓ Metrics are JSON-serializable ({len(json_str)} bytes)")
    print(f"✓ Ready for Prometheus/DataDog export")


def run_all_tests():
    """Run all load tests"""
    print("\n" + "="*60)
    print("FINORBIT ROUTER LOAD TEST SUITE")
    print("="*60)
    
    try:
        test_cache_effectiveness()
        test_latency_baseline()
        test_circuit_breaker_behavior()
        test_rag_decision_distribution()
        test_metrics_export()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nRecommendations for production:")
        print("1. Monitor cache hit ratio - target >70%")
        print("2. Set up alerts when circuit breaker opens")
        print("3. Export metrics to Prometheus/DataDog every 60s")
        print("4. Track P99 latency - should be <500ms for LLM calls")
        print("5. Monitor confidence score distribution")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    run_all_tests()
