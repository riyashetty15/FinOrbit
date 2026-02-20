"""
Performance Benchmarking Suite for FinOrbit
Measures throughput, latency, and resource utilization
"""

import asyncio
import httpx
import time
import statistics
from typing import List, Dict
from dataclasses import dataclass
import json


@dataclass
class PerformanceMetrics:
    """Performance benchmark results"""
    
    # Throughput
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    
    # Latency
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Success rates
    success_rate: float
    error_rate: float


class PerformanceBenchmark:
    """Performance testing for FinOrbit"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.results = []
        
    async def run_throughput_test(self, num_requests: int = 50, concurrency: int = 5) -> PerformanceMetrics:
        """Test system throughput with concurrent requests"""
        
        print(f"\n[THROUGHPUT TEST] Running {num_requests} requests with concurrency={concurrency}")
        print("="*80)
        
        queries = [
            "What are mutual funds?",
            "How to improve credit score?",
            "What is term insurance?",
            "Section 80C deductions",
            "How much to save for retirement?",
            "What are RBI NBFC guidelines?",
            "SEBI mutual fund regulations",
            "What is SIP investment?",
            "How do health insurance claims work?",
            "Tax planning strategies",
        ]
        
        latencies = []
        successes = 0
        failures = 0
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrency)
            
            async def make_request(query_idx):
                nonlocal successes, failures
                
                async with semaphore:
                    query = queries[query_idx % len(queries)]
                    req_start = time.time()
                    
                    try:
                        response = await client.post(
                            f"{self.backend_url}/query",
                            json={
                                "userId": f"bench_user_{query_idx}",
                                "conversationId": f"bench_conv_{query_idx}",
                                "query": query
                            }
                        )
                        
                        req_latency = (time.time() - req_start) * 1000
                        latencies.append(req_latency)
                        
                        if response.status_code == 200:
                            successes += 1
                        else:
                            failures += 1
                        
                        if query_idx % 10 == 0:
                            print(f"  Progress: {query_idx}/{num_requests} requests completed")
                        
                    except Exception as e:
                        failures += 1
                        print(f"  Request {query_idx} failed: {str(e)[:50]}")
            
            # Run all requests
            tasks = [make_request(i) for i in range(num_requests)]
            await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            total_requests=num_requests,
            successful_requests=successes,
            failed_requests=failures,
            requests_per_second=num_requests / total_time,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            success_rate=successes / num_requests if num_requests > 0 else 0,
            error_rate=failures / num_requests if num_requests > 0 else 0
        )
        
        return metrics
    
    async def run_stress_test(self, duration_seconds: int = 30, rps_target: int = 10) -> PerformanceMetrics:
        """Run sustained load for specified duration"""
        
        print(f"\n[STRESS TEST] Running {duration_seconds}s at {rps_target} RPS")
        print("="*80)
        
        queries = [
            "What are mutual funds?",
            "How to improve credit score?",
            "What is term insurance?",
        ]
        
        latencies = []
        successes = 0
        failures = 0
        total_requests = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while time.time() < end_time:
                batch_start = time.time()
                
                # Send batch of requests
                async def make_request():
                    nonlocal successes, failures, total_requests
                    total_requests += 1
                    
                    query = queries[total_requests % len(queries)]
                    req_start = time.time()
                    
                    try:
                        response = await client.post(
                            f"{self.backend_url}/query",
                            json={
                                "userId": "stress_user",
                                "conversationId": f"stress_conv_{total_requests}",
                                "query": query
                            }
                        )
                        
                        req_latency = (time.time() - req_start) * 1000
                        latencies.append(req_latency)
                        
                        if response.status_code == 200:
                            successes += 1
                        else:
                            failures += 1
                            
                    except Exception as e:
                        failures += 1
                
                # Send requests to meet RPS target
                tasks = [make_request() for _ in range(rps_target)]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Sleep to maintain RPS
                batch_duration = time.time() - batch_start
                sleep_time = max(0, 1.0 - batch_duration)
                await asyncio.sleep(sleep_time)
                
                elapsed = int(time.time() - start_time)
                if elapsed % 5 == 0:
                    print(f"  {elapsed}s elapsed - {total_requests} requests sent")
        
        total_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            requests_per_second=total_requests / total_time,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
            p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            success_rate=successes / total_requests if total_requests > 0 else 0,
            error_rate=failures / total_requests if total_requests > 0 else 0
        )
        
        return metrics
    
    def print_results(self, test_name: str, metrics: PerformanceMetrics):
        """Print benchmark results"""
        
        print(f"\n{'='*80}")
        print(f"{test_name} RESULTS")
        print(f"{'='*80}")
        
        print(f"\n--- Request Summary ---")
        print(f"Total Requests: {metrics.total_requests}")
        print(f"Successful: {metrics.successful_requests} ({metrics.success_rate:.1%})")
        print(f"Failed: {metrics.failed_requests} ({metrics.error_rate:.1%})")
        print(f"Throughput: {metrics.requests_per_second:.2f} req/s")
        
        print(f"\n--- Latency Distribution ---")
        print(f"Min: {metrics.min_latency_ms:.0f}ms")
        print(f"Avg: {metrics.avg_latency_ms:.0f}ms")
        print(f"Median: {metrics.median_latency_ms:.0f}ms")
        print(f"P95: {metrics.p95_latency_ms:.0f}ms")
        print(f"P99: {metrics.p99_latency_ms:.0f}ms")
        print(f"Max: {metrics.max_latency_ms:.0f}ms")
        
        print(f"\n--- Performance Grade ---")
        if metrics.success_rate >= 0.99 and metrics.p95_latency_ms < 2000:
            print("Grade: A (Excellent)")
        elif metrics.success_rate >= 0.95 and metrics.p95_latency_ms < 3000:
            print("Grade: B (Good)")
        elif metrics.success_rate >= 0.90 and metrics.p95_latency_ms < 5000:
            print("Grade: C (Acceptable)")
        else:
            print("Grade: D (Needs Improvement)")
        
        print(f"{'='*80}\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to file"""
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBenchmark results saved to: {output_path}")


async def main():
    """Run performance benchmarks"""
    
    benchmark = PerformanceBenchmark(backend_url="http://localhost:8000")
    
    all_results = {}
    
    try:
        # Test 1: Throughput with low concurrency
        print("\n" + "="*80)
        print("TEST 1: THROUGHPUT - LOW CONCURRENCY")
        print("="*80)
        metrics1 = await benchmark.run_throughput_test(num_requests=25, concurrency=3)
        benchmark.print_results("Test 1: Low Concurrency", metrics1)
        all_results['low_concurrency'] = metrics1.__dict__
        
        # Test 2: Throughput with high concurrency
        print("\n" + "="*80)
        print("TEST 2: THROUGHPUT - HIGH CONCURRENCY")
        print("="*80)
        metrics2 = await benchmark.run_throughput_test(num_requests=50, concurrency=10)
        benchmark.print_results("Test 2: High Concurrency", metrics2)
        all_results['high_concurrency'] = metrics2.__dict__
        
        # Test 3: Stress test
        print("\n" + "="*80)
        print("TEST 3: SUSTAINED LOAD (STRESS TEST)")
        print("="*80)
        metrics3 = await benchmark.run_stress_test(duration_seconds=30, rps_target=5)
        benchmark.print_results("Test 3: Stress Test", metrics3)
        all_results['stress_test'] = metrics3.__dict__
        
        # Save results
        benchmark.save_results(all_results, "tests/performance_benchmark_results.json")
        
        print("\n✓ Performance benchmarking complete!")
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n[START] FinOrbit Performance Benchmark")
    print("\nPrerequisites:")
    print("  1. Backend server running on http://localhost:8000")
    print("  2. RAG server running on http://localhost:8081")
    print("\nThis will run load tests - monitor your system resources!")
    print("Press Ctrl+C to cancel...\n")
    
    input("Press Enter to start benchmarking...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user")
