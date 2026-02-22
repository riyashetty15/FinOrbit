# ==============================================
# File: backend/core/router.py
# Description: Intent Router Agent with LLM-based RAG detection
# ==============================================
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import re
import logging
import os
import time
from threading import Lock
from dataclasses import dataclass, field
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class RouteIntent:
    """
    Enriched routing decision with evidence requirements.
    
    Attributes:
        module: Target agent/module (tax_planner, credit, etc.)
        needs_evidence: Whether query requires regulatory grounding
        jurisdiction: Geographic context (default: "IN" for India)
        time_sensitivity: Whether query needs latest/current rules ("high" | "low")
        query_type: Type of query ("info" | "advice" | "compliance")
    """
    module: str
    needs_evidence: bool
    jurisdiction: str = "IN"
    time_sensitivity: str = "low"
    query_type: str = "info"


@dataclass
class RouterMetrics:
    """Metrics collection for router observability"""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    circuit_breaker_opens: int = 0
    circuit_breaker_recoveries: int = 0
    rag_decisions_yes: int = 0
    rag_decisions_no: int = 0
    avg_confidence_score: float = 0.0
    confidence_scores: list = field(default_factory=list)
    total_routing_time_ms: float = 0.0
    routing_times: list = field(default_factory=list)
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def record_cache_eviction(self):
        self.cache_evictions += 1
    
    def record_llm_call(self, needs_rag: bool, confidence: float):
        self.llm_calls += 1
        if needs_rag:
            self.rag_decisions_yes += 1
        else:
            self.rag_decisions_no += 1
        
        self.confidence_scores.append(confidence)
        # Keep only last 1000 scores for memory efficiency
        if len(self.confidence_scores) > 1000:
            self.confidence_scores.pop(0)
        
        # Calculate rolling average
        if self.confidence_scores:
            self.avg_confidence_score = sum(self.confidence_scores) / len(self.confidence_scores)
    
    def record_llm_error(self):
        self.llm_errors += 1
    
    def record_circuit_breaker_open(self):
        self.circuit_breaker_opens += 1
    
    def record_circuit_breaker_recovery(self):
        self.circuit_breaker_recoveries += 1
    
    def record_routing_time(self, duration_ms: float):
        self.total_routing_time_ms += duration_ms
        self.routing_times.append(duration_ms)
        # Keep only last 1000 times for memory efficiency
        if len(self.routing_times) > 1000:
            self.routing_times.pop(0)
    
    def get_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (0.0-1.0)"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def get_llm_error_rate(self) -> float:
        """Calculate LLM error rate (0.0-1.0)"""
        return self.llm_errors / self.llm_calls if self.llm_calls > 0 else 0.0
    
    def get_avg_routing_time_ms(self) -> float:
        """Calculate average routing time in milliseconds"""
        return self.total_routing_time_ms / self.llm_calls if self.llm_calls > 0 else 0.0
    
    def get_p99_routing_time_ms(self) -> float:
        """Calculate P99 (99th percentile) routing time"""
        if not self.routing_times:
            return 0.0
        sorted_times = sorted(self.routing_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary for monitoring/logging"""
        return {
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "evictions": self.cache_evictions,
                "hit_ratio": round(self.get_cache_hit_ratio(), 3)
            },
            "llm": {
                "total_calls": self.llm_calls,
                "errors": self.llm_errors,
                "error_rate": round(self.get_llm_error_rate(), 3),
                "avg_confidence": round(self.avg_confidence_score, 3)
            },
            "rag_decisions": {
                "yes": self.rag_decisions_yes,
                "no": self.rag_decisions_no
            },
            "circuit_breaker": {
                "opens": self.circuit_breaker_opens,
                "recoveries": self.circuit_breaker_recoveries
            },
            "latency_ms": {
                "avg": round(self.get_avg_routing_time_ms(), 2),
                "p99": round(self.get_p99_routing_time_ms(), 2)
            }
        }


# Intent pattern matching for non-RAG agents (RAG handled by LLM)
INTENT_MAP = {
    r"\b(score|cibil|credit score)\b": "fin_score",
    # Include plurals and common lender terms (e.g., NBFC) to avoid falling back to fin_advisor.
    r"\b(loans?|emis?|credit|borrow|mortgage|nbfc|lender|lending)\b": "credits_loans",
    r"\b(invest|portfolio|mutual fund|stocks?)\b": "investment_coach",
    # Avoid matching generic 'coverage' (e.g., 'Liquidity Coverage Ratio'); require explicit insurance-related wording.
    r"\b(insurance|premium|term plan)\b|\binsurance coverage\b|\bcoverage amount\b": "insurance_analyzer",
    r"\b(retir(e|ement)|pension|401k|nps)\b": "retirement_planner",
    r"\b(tax|itr|deduction|exemption|section)\b": "tax_planner",
    r"\b(fraud|suspicious|chargeback|phishing)\b": "fraud_shield",
}

# Multi-domain priority rules: if multiple intents match, these take precedence.
# Key = frozenset of matched agents, value = winning agent.
MULTI_DOMAIN_PRIORITY = {
    frozenset(["credits_loans", "tax_planner"]): "tax_planner",       # "tax deduction on home loan EMI" → tax
    frozenset(["investment_coach", "tax_planner"]): "tax_planner",    # "ELSS for tax saving" → tax
    frozenset(["retirement_planner", "tax_planner"]): "tax_planner",  # "NPS tax benefit" → tax
    frozenset(["insurance_analyzer", "tax_planner"]): "tax_planner",  # "insurance premium tax deduction" → tax
}

# Query intent patterns - distinguishes general vs personalized queries
GENERAL_QUERY_PATTERNS = [
    # Simple informational yes/no questions.
    r"^(?:is|are)\b",
    r"\b(what|which|how|why|when|where)\b.*\b(is|are|do|does|did|can|could|should|would)\b",
    r"\b(tell me about|explain|define|describe)\b",
    r"\b(what are the|what is the|how does)\b",
    r"\b(difference between|types of|kinds of)\b",
    r"\b(benefits of|advantages of|disadvantages of)\b",
    r"\b(generally|typically|usually|normally)\b",
    # How much/many questions asking for guidance (not specific calculations)
    r"\bhow (much|many) (should|do|does)\b",
]

PERSONALIZED_QUERY_PATTERNS = [
    r"\b(calculate|compute|determine|find out)\b.*\b(my|for me|i)\b",
    r"\b(recommend|suggest|advise|help me)\b",
    r"\b(should i|can i|would i|shall i)\b",
    r"\b(my|mine|i have|i am|i need|i want)\b",
    r"\b(best for me|suitable for me|right for me)\b",
    r"\b(based on my|given my|considering my)\b",
]


class RouterAgent:
    """
    Context-aware intent router with conversation continuity (Module 4)

    Routes user queries to appropriate specialist agents based on:
    1. Conversation context (if follow-up, route to same agent)
    2. Keyword patterns (regex matching)
    3. Falls back to main finance advisor if no match

    **NEW**: Context-aware routing maintains conversation continuity without
    exploding token usage by only tracking last agent + extracted profile.

    USER DECISION: Removed LLM fallback to simplify architecture and reduce latency.
    Regex matching + conversation tracking is sufficient for financial domain routing.
    """

    def __init__(self):
        """Initialize router with LLM-based RAG detection, caching, and circuit breaker"""
        self._gemini_configured = False
        self._rag_decision_cache: Dict[str, Tuple[bool, float, str]] = {}
        self._cache_lock = Lock()
        self._max_cache_size = int(os.getenv("RAG_CACHE_SIZE", "1000"))
        
        # Metrics tracking
        self.metrics = RouterMetrics()
        self._metrics_lock = Lock()
        
        # Circuit breaker settings
        self._llm_failures = 0
        self._max_failures = int(os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "5"))
        self._circuit_breaker_reset_time = int(os.getenv("LLM_CIRCUIT_BREAKER_RESET_SECS", "60"))
        self._last_failure_time = None
        self._circuit_open = False
        
        # Confidence threshold (configurable)
        self._confidence_threshold = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.7"))
        
        api_key = os.getenv("LLM_API_KEY")
        self._model_name = os.getenv("CUSTOM_MODEL_NAME", "gpt-4o-mini")
        if api_key:
            try:
                self._openai_client = OpenAI(api_key=api_key)
                self._gemini_configured = True
                logger.info(f"Router initialized with OpenAI LLM-based RAG detection (confidence_threshold={self._confidence_threshold})")
            except Exception as e:
                self._openai_client = None
                logger.warning(f"Failed to initialize OpenAI: {e}, falling back to keyword routing")
        else:
            self._openai_client = None
            logger.warning("No LLM_API_KEY found, using keyword-only routing")

    def _simple_rag_heuristic(self, query: str) -> bool:
        """Keyword-only RAG trigger used when no LLM is configured.

        This is a lightweight safety net so clearly regulatory / factual
        queries (e.g., LCR for NBFCs) still go to the RAG agent even when
        the Gemini-based classifier is disabled.
        """
        text = query.lower()

        patterns = [
            r"\bliquidity coverage ratio\b",
            r"\blcr\b",
            r"\bcapital adequacy\b",
            r"\bcrar\b",
            r"\bregulatory (requirement|ratio|limit)s?\b",
            r"\b(nbfc|nbfcs)\b",
            r"\b(sebi|rbi|irdai)\b",
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True

        return False
    
    def _detect_evidence_need(self, query: str) -> bool:
        """
        Detect if query requires regulatory/factual grounding from RAG.
        
        Triggers evidence need for:
        - Regulatory questions (RBI, SEBI, IRDAI rules, circulars)
        - Compliance queries (allowed, regulated, requirements)
        - Specific fact lookups (rates, limits, procedures)
        """
        text = query.lower()
        
        evidence_patterns = [
            # Regulatory bodies
            r"\b(rbi|sebi|irdai|fiu|cbdt|customs)\b",
            # Regulatory documents
            r"\b(rule|regulation|circular|master direction|guideline|notification)\b",
            # Compliance keywords
            r"\b(allowed|permitted|regulated|required|mandatory|prohibited)\b",
            r"\b(compliance|regulatory|legal|statute)\b",
            # Specific factual queries
            r"\b(what is the|what are the|how much is|what is required)\b.*\b(limit|rate|ratio|percentage|requirement)\b",
            r"\b(npa|lcr|crar|capital adequacy|liquidity coverage)\b",
            # Authority statements
            r"\b(according to|as per|under section|under act)\b",
        ]
        
        for pattern in evidence_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_time_sensitivity(self, query: str) -> str:
        """
        Detect if query needs latest/current information.
        
        Returns:
            "high": Query explicitly asks for latest/updated info
            "low": General query, any recent info acceptable
        """
        text = query.lower()
        
        time_sensitive_patterns = [
            r"\b(latest|current|updated|new|recent|as of \d{4})\b",
            r"\b(2024|2025|2026|this year)\b",
            r"\b(changed|revised|amended)\b",
        ]
        
        for pattern in time_sensitive_patterns:
            if re.search(pattern, text):
                return "high"
        
        return "low"
    
    def _detect_query_type(self, query: str) -> str:
        """
        Classify query type for better response handling.
        
        Returns:
            "compliance": Regulatory/compliance question
            "advice": Personal recommendation request
            "info": General informational query
        """
        text = query.lower()
        
        if re.search(r"\b(allowed|permitted|legal|comply|compliant|violation|regulation)\b", text):
            return "compliance"
        
        if re.search(r"\b(should i|recommend|advise|suggest|best for me|my situation)\b", text):
            return "advice"
        
        return "info"
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (LLM failing)"""
        if not self._circuit_open:
            return False
        
        # Check if circuit should reset
        if self._last_failure_time and (time.time() - self._last_failure_time) > self._circuit_breaker_reset_time:
            logger.info("Circuit breaker reset time elapsed, attempting recovery")
            self._circuit_open = False
            self._llm_failures = 0
            with self._metrics_lock:
                self.metrics.record_circuit_breaker_recovery()
            return False
        
        return True
    
    def _record_llm_failure(self):
        """Record LLM failure and check circuit breaker threshold"""
        self._llm_failures += 1
        self._last_failure_time = time.time()
        
        with self._metrics_lock:
            self.metrics.record_llm_error()
        
        if self._llm_failures >= self._max_failures:
            self._circuit_open = True
            with self._metrics_lock:
                self.metrics.record_circuit_breaker_open()
            logger.error(f"LLM circuit breaker opened after {self._llm_failures} failures")
    
    def _record_llm_success(self):
        """Record successful LLM call"""
        if self._llm_failures > 0:
            self._llm_failures = 0
            logger.info("LLM recovered, circuit breaker reset")
    
    def _get_cached_rag_decision(self, query: str) -> Optional[Tuple[bool, float, str]]:
        """Retrieve cached RAG decision if available"""
        with self._cache_lock:
            cached = self._rag_decision_cache.get(query)
            if cached:
                with self._metrics_lock:
                    self.metrics.record_cache_hit()
            else:
                with self._metrics_lock:
                    self.metrics.record_cache_miss()
            return cached
    
    def _cache_rag_decision(self, query: str, decision: Tuple[bool, float, str]):
        """Cache RAG decision with LRU eviction if cache is full"""
        with self._cache_lock:
            # Simple LRU: clear oldest if cache exceeds max size
            if len(self._rag_decision_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._rag_decision_cache))
                del self._rag_decision_cache[oldest_key]
                with self._metrics_lock:
                    self.metrics.record_cache_eviction()
                logger.debug(f"RAG cache evicted oldest entry, size now: {len(self._rag_decision_cache)}")
            
            self._rag_decision_cache[query] = decision
    
    def _keyword_based_rag_check(self, query: str) -> bool:
        """Ask LLM if query needs RAG lookup"""
        if not self._gemini_configured:
            return (False, 0.0, "LLM not available")
        
        try:
            prompt = f"""Analyze if this financial query needs SPECIFIC DATA from a knowledge base.

Query: "{query}"

NEEDS RAG (return true) ONLY if asking for:
- Specific fund details (NAV, AUM, holdings, commission, expense ratio, trail commission)
- Regulatory info (SEBI/RBI/IRDAI guidelines, circulars, notifications)
- Document content (factsheets, disclosures, policy documents)
- Historical data or statistics from official sources
- Compliance requirements or regulatory updates
- Specific scheme/product details with names mentioned

DOES NOT NEED RAG (return false) if:
- General financial concepts or definitions
- How-to advice or recommendations  
- Personal financial planning
- Calculations or simulations
- General market information
- Personal statements sharing age, income, or profile (e.g., "I am 25 years old", "I earn 10 lakh")
- Greetings or small talk (e.g., "hi", "hello", "thank you")
- Questions asking for investment advice or recommendations
- Vague queries without specific fund/regulation names

IMPORTANT: RAG is ONLY for looking up specific factual data from documents.
Personal information sharing or advice-seeking queries should return false.

Respond ONLY with valid JSON:
{{"needs_rag": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            response = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": "You are a financial query classifier. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            
            result_text = (response.choices[0].message.content or "").strip()
            
            import json
            # Gemini usually returns '```json ... ```', strict JSON mode handles this better
            # but failsafe cleaning is good practice
            clean_json = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean_json)
            
            needs_rag = result.get("needs_rag", False)
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Cache the decision
            decision = (needs_rag, confidence, reasoning)
            self._cache_rag_decision(query, decision)
            
            # Record metrics
            with self._metrics_lock:
                self.metrics.record_llm_call(needs_rag, confidence)
            self._record_llm_success()
            
            logger.info(f"LLM routing: needs_rag={needs_rag}, confidence={confidence:.2f}, reasoning='{reasoning}'")
            return decision
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            with self._metrics_lock:
                self.metrics.record_llm_error()
            self._record_llm_failure()
            return (False, 0.0, f"LLM error: {str(e)}")
    
    def _ask_llm_needs_rag(self, query: str) -> Tuple[bool, float, str]:
        """Ask LLM if query needs RAG lookup (with caching and circuit breaker).

        Always returns a (needs_rag, confidence, reasoning) tuple.
        """
        if not self._gemini_configured:
            return (False, 0.0, "LLM not available")
        
        # Check circuit breaker
        if self._is_circuit_open():
            logger.warning("LLM circuit breaker is open, falling back to keyword routing")
            return (False, 0.0, "Circuit breaker open")
        
        # Check cache
        cached = self._get_cached_rag_decision(query)
        if cached:
            logger.debug(f"RAG decision cache hit for query (confidence={cached[1]:.2f})")
            return cached

        # No cached decision – call the LLM-based checker
        return self._keyword_based_rag_check(query)

    def classify_query_intent(self, query: str) -> str:
        """
        Classify whether query is general (informational) or personalized (needs profile)

        Args:
            query: User query text

        Returns:
            "general" or "personalized"

        Examples:
            >>> router = RouterAgent()
            >>> router.classify_query_intent("What are tax slabs?")
            'general'
            >>> router.classify_query_intent("Calculate my tax")
            'personalized'
            >>> router.classify_query_intent("Should I invest in stocks?")
            'personalized'
        """
        query_lower = query.lower()

        # Check for general patterns first (informational questions)
        for pattern in GENERAL_QUERY_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "general"

        # Check for personalized patterns (calculations, advice)
        for pattern in PERSONALIZED_QUERY_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "personalized"

        # Default to personalized if ambiguous (safer - will ask for profile if needed)
        return "personalized"

    def route_with_intent(self, query: str, hinted: Optional[str] = None) -> Tuple[str, str]:
        """
        Route query to agent AND classify intent

        Args:
            query: User query text
            hinted: Optional explicit agent hint

        Returns:
            Tuple of (agent_name, intent) where:
                - agent_name: "tax_planner", "investment_coach", etc.
                - intent: "general" or "personalized"

        Example:
            >>> router = RouterAgent()
            >>> router.route_with_intent("What are tax slabs?")
            ('tax_planner', 'general')
            >>> router.route_with_intent("Calculate my tax for ₹10 lakhs")
            ('tax_planner', 'personalized')
        """
        agent = self.route(query, hinted)
        intent = self.classify_query_intent(query)
        return agent, intent

    def route_with_context(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        hinted: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Context-aware routing that maintains conversation continuity

        Args:
            query: User query text
            conversation_context: Previous conversation state from ConversationContext.get_context()
                {
                    "last_agent": "tax_planner",
                    "profile": {"income": 2000000},
                    "turn_count": 1
                }
            hinted: Optional explicit agent hint (overrides all logic)

        Returns:
            Tuple of (agent_name, intent)

        Routing Logic:
        1. If explicit hint → use hint
        2. If conversation_context exists AND query is follow-up → route to last_agent
        3. Else → standard regex routing

        Examples:
            # Turn 1: New conversation
            >>> router.route_with_context("What is my tax?", context=None)
            ('tax_planner', 'personalized')

            # Turn 2: Follow-up response
            >>> context = {"last_agent": "tax_planner", "profile": {}, "turn_count": 1}
            >>> router.route_with_context("My income is 20 lakhs", context)
            ('tax_planner', 'personalized')  # Routes to same agent!

            # Turn 3: New topic in same conversation
            >>> router.route_with_context("What are mutual funds?", context)
            ('investment_coach', 'general')  # New topic → new routing
        """
        # Import here to avoid circular dependency
        from backend.core.conversation_context import ConversationContext

        # If explicit agent hint provided, use it
        if hinted:
            intent = self.classify_query_intent(query)
            logger.info(f"Routing with explicit hint: {hinted}, intent: {intent}")
            return hinted, intent

        # Check if this is a follow-up to previous conversation
        is_follow_up = ConversationContext.is_follow_up(query, conversation_context)

        if is_follow_up and conversation_context:
            # Route to same agent that handled previous turn
            last_agent = conversation_context.get("last_agent", "fin_advisor")
            intent = self.classify_query_intent(query)

            logger.info(f"Follow-up detected → routing to last agent: {last_agent}, intent: {intent}")
            return last_agent, intent

        # Not a follow-up or no context → standard routing
        agent = self.route(query, hinted)
        intent = self.classify_query_intent(query)

        if conversation_context:
            logger.info(f"New topic in existing conversation → routing to: {agent}, intent: {intent}")
        else:
            logger.info(f"New conversation → routing to: {agent}, intent: {intent}")

        return agent, intent

    def route(self, query: str, hinted: Optional[str] = None) -> str:
        """
        Route query with LLM-based RAG detection + keyword fallback
        
        Returns:
            Agent key
        """
        route_start_time = time.time()
        
        if hinted:
            logger.info(f"Router using hinted agent: {hinted}")
            return hinted
        
        # First, apply high-precision keyword heuristic for clear RAG cases
        needs_rag = self._simple_rag_heuristic(query)
        
        # If heuristic says RAG is needed, we can skip LLM-based routing
        if needs_rag:
            logger.info("Keyword heuristic indicates RAG is required; skipping LLM routing.")
        
        # Otherwise, check if query needs RAG using LLM (with keyword fallback)
        
        if not needs_rag and self._gemini_configured:
            needs_rag, confidence, reasoning = self._ask_llm_needs_rag(query)
            
            # If LLM has low confidence, use keyword fallback
            if confidence < self._confidence_threshold:
                logger.info(f"LLM confidence low ({confidence:.2f}), checking keywords (threshold={self._confidence_threshold})")
                keyword_rag, _, _ = self._keyword_based_rag_check(query)
                if keyword_rag:
                    needs_rag = True
                    logger.info("Keyword check suggests RAG needed, overriding LLM")
        elif not needs_rag:
            # No LLM available and heuristic didn't trigger → keywords only
            needs_rag = self._simple_rag_heuristic(query)
        
        if needs_rag:
            logger.info(f"Routing to RAG agent")
            agent = "rag_agent"
        else:
            # Pattern matching for other agents — collect ALL matches
            matched_agents = []
            for pattern, agent_match in INTENT_MAP.items():
                if re.search(pattern, query, re.IGNORECASE):
                    matched_agents.append(agent_match)

            if len(matched_agents) > 1:
                # Multi-domain query: check priority rules
                matched_set = frozenset(matched_agents)
                priority_agent = MULTI_DOMAIN_PRIORITY.get(matched_set)
                if priority_agent:
                    agent = priority_agent
                    logger.info(f"Multi-domain priority → {agent} (from {matched_agents})")
                else:
                    agent = matched_agents[0]
                    logger.info(f"Multi-domain first-match → {agent} (from {matched_agents})")
            elif matched_agents:
                agent = matched_agents[0]
                logger.info(f"Router matched pattern → {agent}")
            else:
                agent = "fin_advisor"
                logger.info("Router defaulting to fin_advisor")
        
        # Record routing latency
        route_duration_ms = (time.time() - route_start_time) * 1000
        with self._metrics_lock:
            self.metrics.record_routing_time(route_duration_ms)
        
        return agent
    
    def route_with_evidence_intent(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        hinted: Optional[str] = None
    ) -> RouteIntent:
        """
        Production-grade routing with evidence requirements detection.
        
        Returns RouteIntent with:
        - module: Target agent/specialist
        - needs_evidence: Whether RAG grounding is required
        - jurisdiction: Geographic context (extracted from user_context or default "IN")
        - time_sensitivity: Whether latest rules needed ("high" | "low")
        - query_type: "info" | "advice" | "compliance"
        
        Args:
            query: User query text
            user_context: Optional user profile/context with jurisdiction info
            hinted: Optional explicit module hint
        
        Returns:
            RouteIntent with enriched routing decision
        
        Example:
            >>> intent = router.route_with_evidence_intent("What are NBFC NPA rules?")
            >>> print(intent.module, intent.needs_evidence, intent.time_sensitivity)
            ('credit', True, 'high')
        """
        # Get base routing decision
        module = self.route(query, hinted)
        
        # Detect evidence need
        needs_evidence = self._detect_evidence_need(query)
        
        # Detect time sensitivity
        time_sensitivity = self._detect_time_sensitivity(query)
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Extract jurisdiction from user context or default to India
        jurisdiction = "IN"
        if user_context:
            jurisdiction = user_context.get("jurisdiction", "IN")
            # Also check country code
            if "country" in user_context:
                country_code = user_context["country"]
                if country_code in ["US", "UK", "SG", "AE"]:
                    jurisdiction = country_code
        
        intent = RouteIntent(
            module=module,
            needs_evidence=needs_evidence,
            jurisdiction=jurisdiction,
            time_sensitivity=time_sensitivity,
            query_type=query_type
        )
        
        logger.info(f"RouteIntent: module={module}, needs_evidence={needs_evidence}, "
                   f"jurisdiction={jurisdiction}, time_sensitivity={time_sensitivity}, query_type={query_type}")
        
        return intent
    
