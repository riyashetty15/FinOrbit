"""
Comprehensive Model Evaluation Suite for FinOrbit
Tests RAG quality, agent accuracy, compliance, and system performance
"""

import asyncio
import httpx
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    
    # RAG Metrics
    rag_precision: float = 0.0
    rag_recall: float = 0.0
    rag_hit_rate: float = 0.0
    rag_avg_similarity: float = 0.0
    
    # Agent Routing Metrics
    routing_accuracy: float = 0.0
    rag_decision_accuracy: float = 0.0
    
    # Response Quality Metrics
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    compliance_pass_rate: float = 0.0
    grounding_accuracy: float = 0.0
    
    # Performance Metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Coverage Metrics
    evidence_coverage_accuracy: float = 0.0
    
    # Detailed Results
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0


class FinOrbitEvaluator:
    """Comprehensive evaluation framework for FinOrbit system"""
    
    def __init__(self, backend_url: str = "http://localhost:8000", 
                 rag_url: str = "http://localhost:8081"):
        self.backend_url = backend_url
        self.rag_url = rag_url
        self.results = []
        self.latencies = []
        
    async def evaluate_all(self, dataset_path: str) -> EvaluationMetrics:
        """Run complete evaluation suite"""
        
        print("\n" + "="*80)
        print("FINORBIT MODEL EVALUATION SUITE")
        print("="*80)
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Run all evaluation categories
        metrics = EvaluationMetrics()
        
        print("\n[1/6] Evaluating RAG Retrieval Quality...")
        rag_results = await self._evaluate_rag_retrieval(dataset.get('rag_retrieval_tests', []))
        metrics.rag_precision = rag_results['precision']
        metrics.rag_recall = rag_results['recall']
        metrics.rag_hit_rate = rag_results['hit_rate']
        metrics.rag_avg_similarity = rag_results['avg_similarity']
        
        print("\n[2/6] Evaluating Agent Response Quality...")
        agent_results = await self._evaluate_agent_responses(dataset.get('agent_response_tests', []))
        metrics.routing_accuracy = agent_results['routing_accuracy']
        
        print("\n[3/6] Evaluating Evidence Coverage...")
        coverage_results = await self._evaluate_evidence_coverage(dataset.get('evidence_coverage_tests', []))
        metrics.evidence_coverage_accuracy = coverage_results['accuracy']
        metrics.citation_precision = coverage_results['citation_precision']
        metrics.citation_recall = coverage_results['citation_recall']
        
        print("\n[4/6] Evaluating Compliance & Safety...")
        compliance_results = await self._evaluate_compliance(dataset.get('compliance_tests', []))
        metrics.compliance_pass_rate = compliance_results['pass_rate']
        
        print("\n[5/6] Evaluating Routing Accuracy...")
        routing_results = await self._evaluate_routing(dataset.get('routing_accuracy_tests', []))
        metrics.rag_decision_accuracy = routing_results['rag_decision_accuracy']
        
        print("\n[6/6] Evaluating Grounding Validation...")
        grounding_results = await self._evaluate_grounding(dataset.get('grounding_validation_tests', []))
        metrics.grounding_accuracy = grounding_results['accuracy']
        
        # Calculate latency metrics
        if self.latencies:
            metrics.avg_latency_ms = statistics.mean(self.latencies)
            metrics.p95_latency_ms = statistics.quantiles(self.latencies, n=20)[18]  # 95th percentile
            metrics.p99_latency_ms = statistics.quantiles(self.latencies, n=100)[98]  # 99th percentile
        
        # Calculate overall pass/fail
        metrics.total_tests = len(self.results)
        metrics.passed_tests = sum(1 for r in self.results if r.get('passed', False))
        metrics.failed_tests = metrics.total_tests - metrics.passed_tests
        
        return metrics
    
    async def _evaluate_rag_retrieval(self, tests: List[Dict]) -> Dict[str, float]:
        """Evaluate RAG retrieval quality"""
        
        relevant_retrieved = 0
        total_relevant = 0
        total_retrieved = 0
        hits = 0
        similarities = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for test in tests:
                try:
                    start_time = time.time()
                    
                    # Call RAG retrieval endpoint
                    response = await client.post(
                        f"{self.rag_url}/retrieve",
                        json={
                            "query": test['query'],
                            "module": test['module'],
                            "top_k": 5
                        }
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    self.latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        chunks = result.get('chunks', [])
                        
                        # Check if we got results when we should (hit rate)
                        if test['ground_truth_relevance']:
                            total_relevant += test['min_expected_chunks']
                            if len(chunks) >= test['min_expected_chunks']:
                                hits += 1
                        
                        # Count relevant chunks (keyword matching as proxy)
                        for chunk in chunks:
                            total_retrieved += 1
                            content = chunk.get('text', '').lower()
                            similarities.append(chunk.get('score', 0))
                            
                            # Check if chunk contains expected keywords
                            if any(kw.lower() in content for kw in test['expected_keywords']):
                                relevant_retrieved += 1
                        
                        passed = len(chunks) >= test['min_expected_chunks'] if test['ground_truth_relevance'] else True
                        self.results.append({
                            'test_id': test['id'],
                            'category': 'rag_retrieval',
                            'passed': passed,
                            'chunks_retrieved': len(chunks),
                            'latency_ms': latency
                        })
                        
                        print(f"  {test['id']}: {'PASS' if passed else 'FAIL'} "
                              f"(retrieved={len(chunks)}, latency={latency:.0f}ms)")
                    
                except Exception as e:
                    print(f"  {test['id']}: ERROR - {str(e)}")
                    self.results.append({
                        'test_id': test['id'],
                        'category': 'rag_retrieval',
                        'passed': False,
                        'error': str(e)
                    })
        
        # Calculate metrics
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        hit_rate = hits / len([t for t in tests if t['ground_truth_relevance']]) if tests else 0
        avg_similarity = statistics.mean(similarities) if similarities else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'hit_rate': hit_rate,
            'avg_similarity': avg_similarity
        }
    
    async def _evaluate_agent_responses(self, tests: List[Dict]) -> Dict[str, float]:
        """Evaluate agent response quality"""
        
        correct_routing = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for test in tests:
                try:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{self.backend_url}/query",
                        json={
                            "userId": "eval_user",
                            "conversationId": f"eval_{test['id']}",
                            "query": test['query']
                        }
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    self.latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        agent_type = result.get('agent_type', '').lower()
                        response_text = result.get('response', '').lower()
                        citations = result.get('evidence', {}).get('citations', [])
                        
                        # Check routing accuracy
                        routed_correctly = test['expected_agent'].lower() in agent_type
                        if routed_correctly:
                            correct_routing += 1
                        
                        # Check citation requirements
                        has_citations = len(citations) > 0
                        citation_check = (has_citations == test['should_have_citations']) or not test['should_have_citations']
                        
                        # Check for prohibited keywords
                        no_prohibited = not any(kw.lower() in response_text for kw in test['must_avoid_keywords'])
                        
                        # Check for expected topics
                        has_topics = any(topic.lower() in response_text for topic in test['expected_topics'])
                        
                        passed = routed_correctly and citation_check and no_prohibited and has_topics
                        
                        self.results.append({
                            'test_id': test['id'],
                            'category': 'agent_response',
                            'passed': passed,
                            'routed_correctly': routed_correctly,
                            'citation_check': citation_check,
                            'no_prohibited': no_prohibited,
                            'has_topics': has_topics,
                            'latency_ms': latency
                        })
                        
                        print(f"  {test['id']}: {'PASS' if passed else 'FAIL'} "
                              f"(routing={'✓' if routed_correctly else '✗'}, "
                              f"citations={'✓' if citation_check else '✗'}, "
                              f"latency={latency:.0f}ms)")
                    
                except Exception as e:
                    print(f"  {test['id']}: ERROR - {str(e)}")
                    self.results.append({
                        'test_id': test['id'],
                        'category': 'agent_response',
                        'passed': False,
                        'error': str(e)
                    })
        
        routing_accuracy = correct_routing / len(tests) if tests else 0
        
        return {
            'routing_accuracy': routing_accuracy
        }
    
    async def _evaluate_evidence_coverage(self, tests: List[Dict]) -> Dict[str, float]:
        """Evaluate evidence coverage and citation quality"""
        
        correct_coverage = 0
        citation_tp = 0  # True positives
        citation_fp = 0  # False positives
        citation_fn = 0  # False negatives
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for test in tests:
                try:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{self.backend_url}/query",
                        json={
                            "userId": "eval_user",
                            "conversationId": f"eval_{test['id']}",
                            "query": test['query']
                        }
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    self.latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        evidence = result.get('evidence', {})
                        coverage = evidence.get('coverage', 'insufficient')
                        citations = evidence.get('citations', [])
                        
                        # Check coverage assessment
                        coverage_correct = (
                            (coverage == 'sufficient' and test['expected_coverage'] == 'sufficient') or
                            (coverage in ['partial', 'insufficient'] and test['expected_coverage'] == 'partial_or_insufficient')
                        )
                        
                        if coverage_correct:
                            correct_coverage += 1
                        
                        # Citation precision/recall
                        has_enough_citations = len(citations) >= test['min_citations']
                        
                        if test['requires_evidence']:
                            if has_enough_citations:
                                citation_tp += 1
                            else:
                                citation_fn += 1
                        else:
                            if has_enough_citations:
                                citation_fp += 1
                        
                        passed = coverage_correct
                        
                        self.results.append({
                            'test_id': test['id'],
                            'category': 'evidence_coverage',
                            'passed': passed,
                            'coverage': coverage,
                            'citations_count': len(citations),
                            'latency_ms': latency
                        })
                        
                        print(f"  {test['id']}: {'PASS' if passed else 'FAIL'} "
                              f"(coverage={coverage}, citations={len(citations)}, "
                              f"latency={latency:.0f}ms)")
                    
                except Exception as e:
                    print(f"  {test['id']}: ERROR - {str(e)}")
                    self.results.append({
                        'test_id': test['id'],
                        'category': 'evidence_coverage',
                        'passed': False,
                        'error': str(e)
                    })
        
        accuracy = correct_coverage / len(tests) if tests else 0
        precision = citation_tp / (citation_tp + citation_fp) if (citation_tp + citation_fp) > 0 else 0
        recall = citation_tp / (citation_tp + citation_fn) if (citation_tp + citation_fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'citation_precision': precision,
            'citation_recall': recall
        }
    
    async def _evaluate_compliance(self, tests: List[Dict]) -> Dict[str, float]:
        """Evaluate compliance and safety mechanisms"""
        
        passed_count = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for test in tests:
                try:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{self.backend_url}/query",
                        json={
                            "userId": "eval_user",
                            "conversationId": f"eval_{test['id']}",
                            "query": test['query']
                        }
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    self.latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get('response', '').lower()
                        compliance = result.get('compliance', {})
                        
                        # Check refusal when expected
                        if test.get('should_refuse', False):
                            refused = 'cannot' in response_text or 'unable' in response_text or 'not authorized' in response_text
                            passed = refused
                        # Check PII flagging
                        elif test.get('should_flag_pii', False):
                            pii_flagged = compliance.get('pii_detected', False) or 'sensitive information' in response_text
                            passed = pii_flagged
                        else:
                            # Should process normally
                            passed = len(response_text) > 50  # Got a real response
                        
                        if passed:
                            passed_count += 1
                        
                        self.results.append({
                            'test_id': test['id'],
                            'category': 'compliance',
                            'passed': passed,
                            'reason': test['reason'],
                            'latency_ms': latency
                        })
                        
                        print(f"  {test['id']}: {'PASS' if passed else 'FAIL'} ({test['reason']})")
                    
                except Exception as e:
                    print(f"  {test['id']}: ERROR - {str(e)}")
                    self.results.append({
                        'test_id': test['id'],
                        'category': 'compliance',
                        'passed': False,
                        'error': str(e)
                    })
        
        pass_rate = passed_count / len(tests) if tests else 0
        
        return {
            'pass_rate': pass_rate
        }
    
    async def _evaluate_routing(self, tests: List[Dict]) -> Dict[str, float]:
        """Evaluate routing decision accuracy"""
        
        correct_rag_decisions = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for test in tests:
                try:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{self.backend_url}/query",
                        json={
                            "userId": "eval_user",
                            "conversationId": f"eval_{test['id']}",
                            "query": test['query']
                        }
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    self.latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        evidence = result.get('evidence', {})
                        has_citations = len(evidence.get('citations', [])) > 0
                        
                        # Proxy: if citations returned, RAG was likely used
                        rag_used = has_citations
                        correct = rag_used == test['should_use_rag']
                        
                        if correct:
                            correct_rag_decisions += 1
                        
                        self.results.append({
                            'test_id': test['id'],
                            'category': 'routing',
                            'passed': correct,
                            'rag_used': rag_used,
                            'latency_ms': latency
                        })
                        
                        print(f"  {test['id']}: {'PASS' if correct else 'FAIL'} "
                              f"(RAG={'used' if rag_used else 'not used'})")
                    
                except Exception as e:
                    print(f"  {test['id']}: ERROR - {str(e)}")
                    self.results.append({
                        'test_id': test['id'],
                        'category': 'routing',
                        'passed': False,
                        'error': str(e)
                    })
        
        rag_decision_accuracy = correct_rag_decisions / len(tests) if tests else 0
        
        return {
            'rag_decision_accuracy': rag_decision_accuracy
        }
    
    async def _evaluate_grounding(self, tests: List[Dict]) -> Dict[str, float]:
        """Evaluate grounding validation accuracy"""
        
        correct_grounding = 0
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for test in tests:
                try:
                    start_time = time.time()
                    
                    response = await client.post(
                        f"{self.backend_url}/query",
                        json={
                            "userId": "eval_user",
                            "conversationId": f"eval_{test['id']}",
                            "query": test['query']
                        }
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    self.latencies.append(latency)
                    
                    if response.status_code == 200:
                        result = response.json()
                        evidence = result.get('evidence', {})
                        citations = evidence.get('citations', [])
                        
                        if test['regulatory_claim']:
                            # Should have citations from valid sources
                            has_valid_citations = any(
                                any(source.lower() in citation.get('source', '').lower() 
                                    for source in test['valid_sources'])
                                for citation in citations
                            ) if test['valid_sources'] else len(citations) > 0
                            
                            passed = has_valid_citations
                        else:
                            # General knowledge - citations optional
                            passed = True
                        
                        if passed:
                            correct_grounding += 1
                        
                        self.results.append({
                            'test_id': test['id'],
                            'category': 'grounding',
                            'passed': passed,
                            'citations_count': len(citations),
                            'latency_ms': latency
                        })
                        
                        print(f"  {test['id']}: {'PASS' if passed else 'FAIL'} "
                              f"(citations={len(citations)})")
                    
                except Exception as e:
                    print(f"  {test['id']}: ERROR - {str(e)}")
                    self.results.append({
                        'test_id': test['id'],
                        'category': 'grounding',
                        'passed': False,
                        'error': str(e)
                    })
        
        accuracy = correct_grounding / len(tests) if tests else 0
        
        return {
            'accuracy': accuracy
        }
    
    def generate_report(self, metrics: EvaluationMetrics, output_path: str):
        """Generate detailed evaluation report"""
        
        report = {
            'summary': asdict(metrics),
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\nOverall: {metrics.passed_tests}/{metrics.total_tests} tests passed "
              f"({metrics.passed_tests/metrics.total_tests*100:.1f}%)")
        
        print("\n--- RAG Retrieval Quality ---")
        print(f"Precision: {metrics.rag_precision:.2%}")
        print(f"Recall: {metrics.rag_recall:.2%}")
        print(f"Hit Rate@5: {metrics.rag_hit_rate:.2%}")
        print(f"Avg Similarity: {metrics.rag_avg_similarity:.3f}")
        
        print("\n--- Agent Performance ---")
        print(f"Routing Accuracy: {metrics.routing_accuracy:.2%}")
        print(f"RAG Decision Accuracy: {metrics.rag_decision_accuracy:.2%}")
        
        print("\n--- Response Quality ---")
        print(f"Citation Precision: {metrics.citation_precision:.2%}")
        print(f"Citation Recall: {metrics.citation_recall:.2%}")
        print(f"Evidence Coverage Accuracy: {metrics.evidence_coverage_accuracy:.2%}")
        print(f"Grounding Accuracy: {metrics.grounding_accuracy:.2%}")
        
        print("\n--- Compliance & Safety ---")
        print(f"Compliance Pass Rate: {metrics.compliance_pass_rate:.2%}")
        
        print("\n--- Performance ---")
        print(f"Avg Latency: {metrics.avg_latency_ms:.0f}ms")
        print(f"P95 Latency: {metrics.p95_latency_ms:.0f}ms")
        print(f"P99 Latency: {metrics.p99_latency_ms:.0f}ms")
        
        print("\n" + "="*80)
        print(f"\nDetailed report saved to: {output_path}")
        print("="*80)
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        
        recommendations = []
        
        if metrics.rag_precision < 0.7:
            recommendations.append("RAG Precision is low (<70%). Consider improving retrieval filters or re-ranking.")
        
        if metrics.rag_recall < 0.6:
            recommendations.append("RAG Recall is low (<60%). Consider expanding chunk size or improving embeddings.")
        
        if metrics.routing_accuracy < 0.8:
            recommendations.append("Routing accuracy is low (<80%). Review agent selection logic and keyword patterns.")
        
        if metrics.citation_recall < 0.8:
            recommendations.append("Citation recall is low (<80%). Ensure evidence extraction is comprehensive.")
        
        if metrics.compliance_pass_rate < 0.9:
            recommendations.append("Compliance pass rate is low (<90%). Review guardrails and safety filters.")
        
        if metrics.p99_latency_ms > 5000:
            recommendations.append("P99 latency > 5s. Consider caching, query optimization, or timeout adjustments.")
        
        if metrics.grounding_accuracy < 0.85:
            recommendations.append("Grounding accuracy is low (<85%). Improve citation extraction and validation.")
        
        if not recommendations:
            recommendations.append("All metrics look good! Continue monitoring in production.")
        
        return recommendations


async def main():
    """Run evaluation suite"""
    
    evaluator = FinOrbitEvaluator(
        backend_url="http://localhost:8000",
        rag_url="http://localhost:8081"
    )
    
    dataset_path = "tests/evaluation_dataset.json"
    output_path = "tests/evaluation_report.json"
    
    try:
        metrics = await evaluator.evaluate_all(dataset_path)
        evaluator.generate_report(metrics, output_path)
        
        # Print recommendations
        print("\n--- Recommendations ---")
        for i, rec in enumerate(evaluator._generate_recommendations(metrics), 1):
            print(f"{i}. {rec}")
        
        print("\n✓ Evaluation complete!")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n[START] FinOrbit Model Evaluation")
    print("\nPrerequisites:")
    print("  1. Backend server running on http://localhost:8000")
    print("  2. RAG server running on http://localhost:8081")
    print("  3. Both servers have been initialized with data")
    print("\nPress Ctrl+C to cancel...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled by user")
