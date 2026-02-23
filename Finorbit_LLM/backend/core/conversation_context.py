# ==============================================
# File: backend/core/conversation_context.py
# Description: Lightweight conversation state tracker for context-aware routing
# ==============================================

from typing import Dict, Any, Optional, Tuple
import asyncio
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# asyncpg pool — initialised lazily; silently disabled if DATABASE_URL unset
# ---------------------------------------------------------------------------
_db_pool = None
_pool_lock = asyncio.Lock() if False else __import__("threading").Lock()
_pool_ready = False


async def _get_pool():
    """Return the module-level asyncpg pool, creating it on first call."""
    global _db_pool, _pool_ready
    if _pool_ready:
        return _db_pool

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        _pool_ready = True  # disabled — do not retry
        return None

    try:
        import asyncpg
        _db_pool = await asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=5,
            command_timeout=5,
        )
        _pool_ready = True
        logger.info("[ConversationContext] asyncpg pool connected")
    except Exception as exc:
        logger.warning(f"[ConversationContext] DB pool init failed, using in-memory only: {exc}")
        _db_pool = None
        _pool_ready = True

    return _db_pool


class ProfileExtractor:
    """
    Lightweight profile information extractor from user queries

    Extracts structured information without using LLM:
    - Income: "My annual income is 20 lakhs" → income: 2000000
    - Age: "I am 35 years old" → age: 35
    - Occupation: "I work in IT" → occupation: "IT"
    """

    # Income patterns (Indian formats)
    INCOME_PATTERNS = [
        # "20 lakhs", "20 lacs", "20L"
        r'\b(\d+(?:\.\d+)?)\s*(?:lakh?s?|lacs?|L)\b',
        # "₹20,00,000", "Rs 2000000"
        r'(?:₹|Rs\.?\s*)?(\d{1,2}(?:,\d{2}){0,2}(?:,\d{3}))',
        # "2000000", "20,00,000" when talking about income
        r'\b(\d{6,9})\b',
    ]

    # Age patterns
    AGE_PATTERNS = [
        r'\b(?:age|am|is)\s+(\d{2})\s*(?:years?)?(?:\s+old)?\b',
        r'\b(\d{2})\s*(?:years?\s+old|yr|yrs)\b',
    ]

    # Occupation patterns
    OCCUPATION_KEYWORDS = {
        'software': 'IT/Software',
        'developer': 'IT/Software',
        'engineer': 'Engineer',
        'it': 'IT/Software',
        'tech': 'IT/Software',
        'doctor': 'Healthcare',
        'teacher': 'Education',
        'business': 'Business',
        'entrepreneur': 'Business',
        'salaried': 'Salaried',
        'self employed': 'Self-Employed',
        'student': 'Student',
        'retired': 'Retired',
    }

    @staticmethod
    def extract_income(text: str) -> Optional[int]:
        """
        Extract income from text in INR

        Examples:
            "20 lakhs" → 2000000
            "₹15,00,000" → 1500000
            "12L" → 1200000
        """
        text_lower = text.lower()

        for pattern in ProfileExtractor.INCOME_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(',', '')

                try:
                    value = float(value_str)

                    # Check if it's in lakhs/lacs/L format
                    if re.search(r'\b' + re.escape(value_str) + r'\s*(?:lakh?s?|lacs?|L)\b', text_lower, re.IGNORECASE):
                        value = value * 100000  # Convert lakhs to actual amount

                    # Validate reasonable income range (1 lakh to 10 crore)
                    if 100000 <= value <= 100000000:
                        return int(value)

                except (ValueError, TypeError):
                    continue

        return None

    @staticmethod
    def extract_age(text: str) -> Optional[int]:
        """
        Extract age from text

        Examples:
            "I am 35 years old" → 35
            "35 yr" → 35
        """
        for pattern in ProfileExtractor.AGE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    # Validate reasonable age range
                    if 18 <= age <= 100:
                        return age
                except (ValueError, TypeError):
                    continue

        return None

    @staticmethod
    def extract_occupation(text: str) -> Optional[str]:
        """
        Extract occupation from text

        Examples:
            "I work in IT" → "IT/Software"
            "I am a software developer" → "IT/Software"
        """
        text_lower = text.lower()

        for keyword, occupation in ProfileExtractor.OCCUPATION_KEYWORDS.items():
            if keyword in text_lower:
                return occupation

        return None

    @staticmethod
    def extract_all(text: str) -> Dict[str, Any]:
        """
        Extract all profile information from text

        Returns:
            Dict with extracted fields (only includes non-None values)
        """
        extracted = {}

        income = ProfileExtractor.extract_income(text)
        if income is not None:
            extracted['income'] = income

        age = ProfileExtractor.extract_age(text)
        if age is not None:
            extracted['age'] = age

        occupation = ProfileExtractor.extract_occupation(text)
        if occupation is not None:
            extracted['occupation'] = occupation

        return extracted


class ConversationContext:
    """
    Lightweight conversation state tracker

    Stores minimal information for context-aware routing:
    - Last agent that handled this conversation
    - Extracted profile information from conversation
    - Whether this is a follow-up query

    Uses in-memory storage (could be moved to Redis/database for production)
    """

    # In-memory storage: conversationId → context
    _contexts: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def get_context(conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation context

        Returns:
            {
                "last_agent": "tax_planner",
                "profile": {"income": 2000000, "age": 35},
                "turn_count": 2
            }
        """
        return ConversationContext._contexts.get(conversation_id)

    @staticmethod
    def update_context(
        conversation_id: str,
        agent: str,
        profile_updates: Dict[str, Any]
    ) -> None:
        """
        Update conversation context after handling a query

        Args:
            conversation_id: Conversation ID
            agent: Agent that handled this turn
            profile_updates: New profile information extracted from query
        """
        if conversation_id not in ConversationContext._contexts:
            ConversationContext._contexts[conversation_id] = {
                "last_agent": agent,
                "profile": dict(profile_updates),  # apply initial updates
                "turn_count": 1
            }
        else:
            context = ConversationContext._contexts[conversation_id]
            context["last_agent"] = agent
            context["turn_count"] = context.get("turn_count", 0) + 1

            # Merge profile updates (new values override old)
            context["profile"].update(profile_updates)

        logger.info(f"Updated context for {conversation_id}: {ConversationContext._contexts[conversation_id]}")

    @staticmethod
    def is_follow_up(query: str, context: Optional[Dict[str, Any]]) -> bool:
        """
        Determine if query is a follow-up to previous conversation

        A query is a follow-up if:
        1. Context exists (previous turns)
        2. Query doesn't start a new topic (not a new question or command)

        Examples of follow-ups:
            - "My annual income is 20 lakhs" (responding to request for info)
            - "Yes" / "No" (answering yes/no question)
            - "What about X?" (continuing same topic)

        Examples of NEW topics:
            - "What are tax slabs?" (new question)
            - "Tell me about investments" (new command)
            - "Calculate my loan EMI" (new task)
        """
        if context is None:
            logger.debug(f"is_follow_up: No context exists → False")
            return False

        query_lower = query.lower().strip()
        logger.debug(f"is_follow_up: Checking query='{query_lower}', context={context}")

        # Check if query starts a new topic (begins with question word or command)
        new_topic_patterns = [
            r'^(?:what|which|how|why|when|where|who|can|should|tell|explain|calculate|show|help|give)',
        ]

        for pattern in new_topic_patterns:
            if re.match(pattern, query_lower):
                logger.debug(f"is_follow_up: Matched new_topic pattern '{pattern}' → False")
                return False

        # Check if query is providing information (follow-up response)
        follow_up_indicators = [
            r'\bmy\s+(?:income|age|salary|occupation)\b',  # "my income is..."
            r'\b(?:i\s+am|i\'m|i\s+work)\b',  # "I am 35", "I work in IT"
            r'^\s*(?:yes|no|okay|ok|sure|fine)\b',  # Yes/no responses
            r'^\d+\s*(?:lakh?s?|lacs?|L|years?|yr)?\b',  # Numeric responses
        ]

        for pattern in follow_up_indicators:
            if re.search(pattern, query_lower):
                logger.info(f"[OK] is_follow_up: Matched follow-up pattern '{pattern}' → True")
                return True
            else:
                logger.debug(f"is_follow_up: Pattern '{pattern}' did not match")

        # If context exists and query is short (< 5 words), likely a follow-up
        word_count = len(query_lower.split())
        if word_count <= 5 and context.get("turn_count", 0) > 0:
            logger.info(f"[OK] is_follow_up: Short query ({word_count} words) with context → True")
            return True

        logger.info(f"[ERROR] is_follow_up: No patterns matched → False")
        return False

    @staticmethod
    def clear_context(conversation_id: str) -> None:
        """Clear conversation context (for testing or manual reset)"""
        if conversation_id in ConversationContext._contexts:
            del ConversationContext._contexts[conversation_id]
            logger.info(f"Cleared context for {conversation_id}")

    # -----------------------------------------------------------------------
    # Async PostgreSQL-backed methods (fall back to in-memory on DB errors)
    # -----------------------------------------------------------------------

    @staticmethod
    async def get_context_async(conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation context, preferring PostgreSQL (24h TTL).

        Falls back to in-memory dict if DATABASE_URL is not set or DB is
        temporarily unavailable.
        """
        pool = await _get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT last_agent, profile_json, turn_count
                        FROM conversation_contexts
                        WHERE conversation_id = $1
                          AND updated_at > NOW() - INTERVAL '24 hours'
                        """,
                        conversation_id,
                    )
                if row:
                    ctx = {
                        "last_agent": row["last_agent"],
                        "profile": json.loads(row["profile_json"]) if isinstance(row["profile_json"], str) else dict(row["profile_json"]),
                        "turn_count": row["turn_count"],
                    }
                    # Keep in-memory in sync
                    ConversationContext._contexts[conversation_id] = ctx
                    return ctx
                return None
            except Exception as exc:
                logger.warning(f"[ConversationContext] DB get failed, using in-memory: {exc}")

        # Fallback
        return ConversationContext.get_context(conversation_id)

    @staticmethod
    async def update_context_async(
        conversation_id: str,
        agent: str,
        profile_updates: Dict[str, Any],
    ) -> None:
        """
        Persist conversation context update to PostgreSQL and in-memory cache.

        Uses INSERT … ON CONFLICT DO UPDATE so it is idempotent and handles
        both new and returning conversations in one round-trip.
        """
        # Always update in-memory first (fast path + fallback)
        ConversationContext.update_context(conversation_id, agent, profile_updates)

        pool = await _get_pool()
        if not pool:
            return

        try:
            current = ConversationContext._contexts.get(conversation_id, {})
            profile = current.get("profile", {})
            turn_count = current.get("turn_count", 1)

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO conversation_contexts
                        (conversation_id, last_agent, profile_json, turn_count, updated_at)
                    VALUES ($1, $2, $3::jsonb, $4, NOW())
                    ON CONFLICT (conversation_id) DO UPDATE
                        SET last_agent  = EXCLUDED.last_agent,
                            profile_json = EXCLUDED.profile_json,
                            turn_count   = EXCLUDED.turn_count,
                            updated_at   = NOW()
                    """,
                    conversation_id,
                    agent,
                    json.dumps(profile),
                    turn_count,
                )
            logger.debug(f"[ConversationContext] Persisted context for {conversation_id} to DB")
        except Exception as exc:
            logger.warning(f"[ConversationContext] DB update failed (in-memory updated): {exc}")
