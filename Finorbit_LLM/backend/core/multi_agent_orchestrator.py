# ==============================================
# Multi-Agent Orchestrator
# ==============================================

from typing import Dict, Any, List, Optional
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
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process complex query requiring multiple agents
        
        Returns:
            {
                "summary": str,  # Synthesized response from all agents
                "sub_responses": List[Dict],  # Individual agent responses
                "agents_used": List[str],
                "execution_order": List[str]
            }
        """
        # Step 1: Check if decomposition is needed
        needs_multi = self.decomposer.needs_decomposition(query)
        
        if not needs_multi:
            logger.info("Query is simple, routing to single agent")
            return None  # Fall back to single-agent routing
        
        logger.info(f"Complex query detected, decomposing...")
        
        # Step 2: Decompose into sub-queries
        sub_queries = self.decomposer.decompose(query)
        
        if len(sub_queries) <= 1:
            logger.info("Decomposition returned single query, using single agent")
            return None
        
        # Step 3: Execute sub-queries in dependency order
        results = []
        context_cache = {}  # Store results for dependencies
        
        for i, sub_query_info in enumerate(sub_queries):
            sub_query = sub_query_info.get("sub_query", "")
            agent_name = sub_query_info.get("agent", "fin_advisor")
            depends_on = sub_query_info.get("depends_on", [])
            
            logger.info(f"Executing sub-query {i+1}/{len(sub_queries)}: {agent_name}")
            
            # Build context from dependencies
            dependency_context = ""
            for dep_idx in depends_on:
                if dep_idx < len(results):
                    dep_result = results[dep_idx]
                    dependency_context += f"\nContext from previous query: {dep_result.get('summary', '')}\n"
            
            # IMPORTANT: Do not mutate the user/sub-query itself.
            # Pass dependency context as a separate field so tools (e.g., RAG) don't receive
            # a rewritten query string.
            augmented_query = sub_query
            if dependency_context and agent_name != "rag_agent":
                augmented_query = f"{sub_query}\n\n{dependency_context}"
            
            # Execute agent
            agent = self.specialist_agents.get(agent_name)
            if not agent:
                logger.warning(f"Agent {agent_name} not found, skipping")
                continue
            
            try:
                # Build initial state for agent
                initial_state = {
                    # Keep the original sub-query intact for logging and tool calls.
                    "query": sub_query if agent_name == "rag_agent" else augmented_query,
                    "user_query": sub_query,
                    "dependency_context": dependency_context.strip() if dependency_context else "",
                    "profile": profile,
                    "intent": "general"
                }
                
                # Execute agent
                import inspect
                if inspect.iscoroutinefunction(agent.run):
                    result = await agent.run(initial_state)
                else:
                    result = agent.run(initial_state)
                
                results.append({
                    "sub_query": sub_query,
                    "agent": agent_name,
                    "summary": result.get("summary", ""),
                    "sources": result.get("sources", []),
                    "confidence": result.get("retrieval_score", result.get("confidence", 0.0))
                })
                
            except Exception as e:
                logger.error(f"Error executing {agent_name} for sub-query: {e}")
                results.append({
                    "sub_query": sub_query,
                    "agent": agent_name,
                    "summary": f"Error: Could not process this part of the query",
                    "sources": [],
                    "confidence": 0.0
                })
        
        # Step 4: Synthesize final response
        synthesized = self._synthesize_multi_agent_response(query, results)
        
        return {
            "summary": synthesized,
            "sub_responses": results,
            "agents_used": list(set([r["agent"] for r in results])),
            "execution_order": [r["agent"] for r in results],
            "is_multi_agent": True
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
