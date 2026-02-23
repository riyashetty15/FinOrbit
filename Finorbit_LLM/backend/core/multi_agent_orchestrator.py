# ==============================================
# Multi-Agent Orchestrator
# ==============================================

from typing import Dict, Any, List, Optional
import asyncio
import inspect
import logging
from backend.core.query_decomposer import QueryDecomposer
from backend.core.router import RouterAgent

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents for complex queries requiring
    multiple specialists (RAG + Investment Coach + etc.)
    """
    
    def __init__(self, specialist_agents: Dict[str, Any]):
        """
        Args:
            specialist_agents: Dict of agent_name -> agent_instance
        """
        self.decomposer = QueryDecomposer()
        self.router = RouterAgent()
        self.specialist_agents = specialist_agents
        logger.info("MultiAgentOrchestrator initialized")
    
    async def process_complex_query(
        self,
        query: str,
        profile: Dict[str, Any],
        session_id: str,
        trace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process complex query requiring multiple agents.

        Independent sub-queries (no depends_on) run concurrently via asyncio.gather.
        Sub-queries with dependencies execute after their prerequisite wave completes.

        Returns:
            Dict with summary, sub_responses, agents_used, execution_order — or None
            if the query is simple and should fall back to single-agent routing.
        """
        # Step 1: Check if decomposition is needed
        needs_multi = self.decomposer.needs_decomposition(query)
        if not needs_multi:
            logger.info("Query is simple, routing to single agent")
            return None

        logger.info(f"[trace_id={trace_id}] Complex query detected, decomposing…")

        # Step 2: Decompose into sub-queries
        sub_queries = self.decomposer.decompose(query)
        if len(sub_queries) <= 1:
            logger.info("Decomposition returned single query, using single agent")
            return None

        # Step 3: Group sub-queries into dependency waves and execute in parallel per wave
        waves = self._build_execution_waves(sub_queries)
        results: List[Dict[str, Any]] = []

        for wave_idx, wave_indices in enumerate(waves):
            wave = [sub_queries[i] for i in wave_indices]
            logger.info(
                f"[trace_id={trace_id}] Wave {wave_idx + 1}/{len(waves)}: "
                f"executing {len(wave)} sub-queries in parallel — "
                f"{[sq.get('agent') for sq in wave]}"
            )

            wave_tasks = [
                self._execute_single_subquery(sq, profile, results, trace_id)
                for sq in wave
            ]
            wave_results = await asyncio.gather(*wave_tasks, return_exceptions=True)

            for sq, wave_result in zip(wave, wave_results):
                if isinstance(wave_result, Exception):
                    logger.error(
                        f"[trace_id={trace_id}] Sub-query via {sq.get('agent')} failed: {wave_result}"
                    )
                    results.append({
                        "sub_query": sq.get("sub_query", ""),
                        "agent": sq.get("agent", "unknown"),
                        "summary": "Error: Could not process this part of the query",
                        "sources": [],
                        "confidence": 0.0,
                    })
                else:
                    results.append(wave_result)

        # Step 4: Synthesize final response
        synthesized = self._synthesize_multi_agent_response(query, results)

        return {
            "summary": synthesized,
            "sub_responses": results,
            "agents_used": list(set(r["agent"] for r in results)),
            "execution_order": [r["agent"] for r in results],
            "is_multi_agent": True,
        }

    def _build_execution_waves(self, sub_queries: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Group sub-query indices into dependency waves for parallel execution.

        Wave 0: sub-queries with no dependencies.
        Wave N: sub-queries whose dependencies all appear in waves 0…N-1.

        Returns a list of waves, each wave being a list of sub-query indices.
        """
        n = len(sub_queries)
        assigned = [False] * n
        waves: List[List[int]] = []

        while not all(assigned):
            completed_indices = {i for batch in waves for i in batch}
            wave: List[int] = []

            for i, sq in enumerate(sub_queries):
                if assigned[i]:
                    continue
                deps = sq.get("depends_on", [])
                if all(d in completed_indices for d in deps):
                    wave.append(i)

            if not wave:
                # Circular or unresolvable deps — add all remaining to break the loop
                wave = [i for i in range(n) if not assigned[i]]

            waves.append(wave)
            for i in wave:
                assigned[i] = True

        return waves

    async def _execute_single_subquery(
        self,
        sub_query_info: Dict[str, Any],
        profile: Dict[str, Any],
        previous_results: List[Dict[str, Any]],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a single sub-query and return its result dict."""
        sub_query = sub_query_info.get("sub_query", "")
        agent_name = sub_query_info.get("agent", "fin_advisor")
        depends_on = sub_query_info.get("depends_on", [])

        # Build dependency context from previously completed results
        dependency_context = ""
        for dep_idx in depends_on:
            if dep_idx < len(previous_results):
                dep_result = previous_results[dep_idx]
                dependency_context += f"\nContext from previous query: {dep_result.get('summary', '')}\n"

        augmented_query = sub_query
        if dependency_context and agent_name != "rag_agent":
            augmented_query = f"{sub_query}\n\n{dependency_context}"

        agent = self.specialist_agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found in specialist_agents")

        logger.info(f"[trace_id={trace_id}] Executing sub-query via {agent_name}: {sub_query[:60]}")

        initial_state = {
            "query": sub_query if agent_name == "rag_agent" else augmented_query,
            "user_query": sub_query,
            "dependency_context": dependency_context.strip(),
            "profile": profile,
            "intent": "general",
        }

        if inspect.iscoroutinefunction(agent.run):
            result = await agent.run(initial_state)
        else:
            result = await asyncio.to_thread(agent.run, initial_state)

        return {
            "sub_query": sub_query,
            "agent": agent_name,
            "summary": result.get("summary", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("retrieval_score", result.get("confidence", 0.0)),
        }
    
    def _synthesize_multi_agent_response(
        self, 
        original_query: str, 
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize a coherent response from multiple agent outputs
        """
        if not self.decomposer.openai_client:
            # Fallback: Just concatenate responses
            sections = []
            for i, result in enumerate(results, 1):
                sections.append(f"**Part {i}:** {result['summary']}")
            return "\n\n".join(sections)
        
        try:
            # Build context from all agent responses
            context_parts = []
            for i, result in enumerate(results, 1):
                agent = result.get("agent", "unknown")
                summary = result.get("summary", "")
                context_parts.append(f"[Agent: {agent}]\n{summary}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            prompt = f"""You are synthesizing responses from multiple financial AI agents into one coherent answer.

Original User Query: "{original_query}"

Individual Agent Responses:
{context}

Instructions:
1. Create a SINGLE coherent response that addresses ALL parts of the user's query
2. Maintain the flow and logic between different parts
3. Preserve factual data (numbers, names, sources) exactly as provided
4. Use clear section headers if needed
5. Keep the user's original question structure in mind
6. Don't add information not provided by the agents

Synthesized Response:"""

            response = self.decomposer.openai_client.chat.completions.create(
                model=self.decomposer._model_name,
                messages=[
                    {"role": "system", "content": "You are a response synthesis expert. Create coherent, comprehensive answers from multiple agent outputs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            synthesized = response.choices[0].message.content.strip()
            logger.info("Successfully synthesized multi-agent response")
            return synthesized
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Fallback
            sections = []
            for i, result in enumerate(results, 1):
                sections.append(f"**Part {i}:** {result['summary']}")
            return "\n\n".join(sections)
