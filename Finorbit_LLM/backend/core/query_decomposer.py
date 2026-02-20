# ==============================================
# Query Decomposition for Multi-Agent Orchestration
# ==============================================

from typing import List, Dict, Any
import logging
from openai import OpenAI
import os
import json

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decomposes complex queries into sub-queries for different agents"""
    
    def __init__(self):
        self.openai_client = None
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self._model_name = os.getenv("CUSTOM_MODEL_NAME", "gpt-4o-mini")
                logger.info("QueryDecomposer initialized with LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
    
    def decompose(self, query: str) -> List[Dict[str, str]]:
        """
        Decompose complex query into sub-queries with agent assignments
        
        Returns:
            List of {
                "sub_query": str,
                "agent": str,  # rag_agent, investment_coach, etc.
                "depends_on": List[int]  # Indices of dependencies
            }
        """
        if not self.openai_client:
            return [{"sub_query": query, "agent": "unknown", "depends_on": []}]
        
        try:
            prompt = f"""Analyze this financial query and break it into sub-queries for different specialist agents.

Query: "{query}"

Available Agents:
- rag_agent: Specific data lookup (commission, NAV, AUM, regulations, factsheets)
- investment_coach: Investment advice, recommendations, portfolio suggestions
- tax_planner: Tax implications, deductions, tax-saving instruments
- insurance_analyzer: Insurance coverage, premium analysis
- retirement_planner: Retirement planning, corpus calculations

Instructions:
1. Break the query into ATOMIC sub-queries
2. Assign each to the most appropriate agent
3. Specify dependencies (if sub-query needs answer from another)
4. Preserve the user's original intent and wording
5. DO NOT introduce new products/entities/numbers not present in the user's query (e.g., do not rewrite "investment" into "fixed deposit" unless the user explicitly said so)
6. If the query is generic (e.g., "Which investment gives guaranteed maximum payout?"), keep sub-queries generic ("Which investments offer the highest guaranteed payout?")

Respond with JSON array:
[
  {{
    "sub_query": "What is the trail commission of Parag Parikh Flexi Cap Fund?",
    "agent": "rag_agent",
    "depends_on": []
  }},
  {{
    "sub_query": "Should I invest ₹10 lakh in Parag Parikh Flexi Cap Fund?",
    "agent": "investment_coach",
    "depends_on": [0]
  }},
  ...
]"""

            response = self.openai_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a query decomposition expert. Always respond with valid JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            sub_queries = json.loads(result_text)
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [{"sub_query": query, "agent": "unknown", "depends_on": []}]
    
    def needs_decomposition(self, query: str) -> bool:
        """Check if query is complex enough to need decomposition"""
        # Deterministic guardrail FIRST (avoids unnecessary LLM decomposition and prevents
        # query rewriting in simple single-domain questions).
        q = (query or "").strip().lower()
        if not q:
            return False

        # Heuristic: multi-part questions or explicit connectors.
        multi_part = (q.count("?") >= 2) or any(tok in q for tok in (" and ", " also ", " plus ", " along with ", " as well as "))

        # Heuristic: multi-domain signals.
        domains = 0
        if any(k in q for k in ("mutual fund", "sip", "nav", "aum", "stock", "equity", "invest", "investment", "returns")):
            domains += 1
        if any(k in q for k in ("tax", "itr", "80c", "deduction", "tds", "gst")):
            domains += 1
        if any(k in q for k in ("insurance", "premium", "claim", "policy")):
            domains += 1
        if any(k in q for k in ("retirement", "nps", "pension")):
            domains += 1
        if any(k in q for k in ("loan", "emi", "credit")):
            domains += 1

        # Short single-domain queries should NOT decompose.
        if len(q) < 140 and domains <= 1 and not multi_part:
            return False

        # If no LLM client, we can't decompose.
        if not self.openai_client:
            return False
        
        try:
            prompt = f"""Does this query need multiple different types of agents to fully answer?

Query: "{query}"

Answer YES if it needs:
- Both specific data lookup AND advice/recommendations
- Multiple different financial domains (investment + tax + insurance)
- Sequential operations (get data first, then analyze)

Answer NO if it's a simple single-agent query.

Respond with only: YES or NO"""

            response = self.openai_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a query complexity analyzer. Respond with only YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            logger.error(f"Complexity check failed: {e}")
            return False
