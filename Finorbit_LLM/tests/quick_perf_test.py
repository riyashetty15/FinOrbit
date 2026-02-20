"""
Quick Performance Test for FinOrbit
"""

import asyncio
import httpx
import time
import statistics


async def quick_perf_test():
    """Run a quick performance test"""
    
    print("\n" + "="*80)
    print("QUICK PERFORMANCE TEST")
    print("="*80)
    
    queries = [
        "What are mutual funds?",
        "How to improve credit score?",
        "What is term insurance?",
        "Section 80C deductions",
        "How much to save for retirement?",
    ]
    
    latencies = []
    successes = 0
    failures = 0
    
    print(f"\nRunning {len(queries)} test queries...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, query in enumerate(queries, 1):
            try:
                start_time = time.time()
                
                response = await client.post(
                    "http://localhost:8000/query",
                    json={
                        "userId": f"perftest_user_{i}",
                        "conversationId": f"perftest_conv_{i}",
                        "query": query
                    }
                )
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                if response.status_code == 200:
                    successes += 1
                    print(f"  {i}. {query[:40]}... ✓ ({latency:.0f}ms)")
                else:
                    failures += 1
                    print(f"  {i}. {query[:40]}... ✗ (HTTP {response.status_code})")
                    
            except Exception as e:
                failures += 1
                print(f"  {i}. {query[:40]}... ✗ ({str(e)[:30]})")
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    total = len(queries)
    print(f"\nTotal Requests: {total}")
    print(f"Successful: {successes} ({successes/total*100:.1f}%)")
    print(f"Failed: {failures} ({failures/total*100:.1f}%)")
    
    if latencies:
        print(f"\nLatency:")
        print(f"  Min: {min(latencies):.0f}ms")
        print(f"  Avg: {statistics.mean(latencies):.0f}ms")
        print(f"  Max: {max(latencies):.0f}ms")
        
        if len(latencies) >= 3:
            print(f"  Median: {statistics.median(latencies):.0f}ms")
    
    print("\n" + "="*80)
    
    # Grade
    if successes == total and statistics.mean(latencies) < 3000:
        print("Performance Grade: A (Excellent)")
    elif successes >= total * 0.8 and statistics.mean(latencies) < 5000:
        print("Performance Grade: B (Good)")
    else:
        print("Performance Grade: C (Needs Improvement)")
    
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(quick_perf_test())
    except KeyboardInterrupt:
        print("\n\nTest cancelled")
