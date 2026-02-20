# ==============================================
# File: backend/core/compliance_engine.py
# Description: JSON-first compliance engine (no DB).
# ==============================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Set
from openai import OpenAI

from backend.core.compliance_types import ComplianceResult

logger = logging.getLogger(__name__)


PatternType = Literal["TEXT", "REGEX", "INTENT", "SEMANTIC"]
RuleType = Literal[
    "BLOCK",
    "FORCE_SAFE_ANSWER",
    "MODIFY_FORCE",
    "MODIFY_REPLACE",
    "MODIFY_APPEND",
    "WARN",
]


@dataclass(frozen=True)
class ComplianceRule:
    id: int
    regulator: str
    module: str
    pattern_type: PatternType
    pattern: str
    rule_type: RuleType
    message: str
    severity: str = "LOW"
    priority: int = 0
    language: str = "ALL"
    channel: str = "ALL"


def load_rules_from_json(path: str) -> List[ComplianceRule]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Compliance rules JSON must be a list")

    rules: List[ComplianceRule] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        rules.append(
            ComplianceRule(
                id=int(item["id"]),
                regulator=str(item.get("regulator", "GENERIC")),
                module=str(item.get("module", "GENERIC")).upper(),
                pattern_type=str(item.get("pattern_type", "REGEX")).upper(),
                pattern=str(item.get("pattern", "")),
                rule_type=str(item.get("rule_type", "WARN")).upper(),
                message=str(item.get("message", "")),
                severity=str(item.get("severity", "LOW")).upper(),
                priority=int(item.get("priority", 0)),
                language=str(item.get("language", "ALL")).lower(),
                channel=str(item.get("channel", "ALL")).upper(),
            )
        )

    # Higher priority first
    rules.sort(key=lambda r: r.priority, reverse=True)
    return rules


class ComplianceEngineService:
    """JSON-first compliance engine.

    - Loads rules from a JSON file (cached with TTL).
    - Applies deterministic TEXT/REGEX/INTENT matches.
    - No DB, no psycopg.
        - Fail-open by default: returns original text on internal errors.
        - Fail-closed for configured CRITICAL categories (e.g., OTP/CVV/password, fraud, tax evasion):
            on internal errors, returns a safe refusal instead of passing through.

    Environment:
      - COMPLIANCE_RULES_PATH: path to rules JSON
      - COMPLIANCE_CACHE_TTL_S: cache ttl seconds
            - COMPLIANCE_FAIL_CLOSED_ENABLED: enable fail-closed for critical categories (default: true)
            - COMPLIANCE_FAIL_CLOSED_MODULES: comma-separated modules to treat as critical (default: FRAUD,TAX)
    """

    def __init__(
        self,
        *,
        rules_path: Optional[str] = None,
        cache_ttl_seconds: int = 60,
    ):
        self.rules_path = rules_path or os.getenv("COMPLIANCE_RULES_PATH") or "backend/rules/compliance_rules.json"
        self.cache_ttl = timedelta(seconds=int(os.getenv("COMPLIANCE_CACHE_TTL_S", str(cache_ttl_seconds))))

        self._fail_closed_enabled = os.getenv("COMPLIANCE_FAIL_CLOSED_ENABLED", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        self._fail_closed_modules: Set[str] = {
            part.strip().upper()
            for part in os.getenv("COMPLIANCE_FAIL_CLOSED_MODULES", "FRAUD,TAX").split(",")
            if part.strip()
        }

        self._rules_cache: List[ComplianceRule] = []
        self._cache_expires_at: Optional[datetime] = None
        
        # Configure OpenAI for semantic checks if API key is present
        api_key = os.getenv("LLM_API_KEY")
        self._model_name = os.getenv("CUSTOM_MODEL_NAME", "gpt-4o-mini")
        if api_key:
            self._openai_client = OpenAI(api_key=api_key)
            self._gemini_configured = True
        else:
            self._openai_client = None
            self._gemini_configured = False

    def close(self) -> None:
        return

    def invalidate_cache(self) -> None:
        self._cache_expires_at = None
        self._rules_cache = []

    def _load_rules_cached(self) -> List[ComplianceRule]:
        if self._cache_expires_at and datetime.now(timezone.utc) < self._cache_expires_at and self._rules_cache:
            return self._rules_cache

        rules = load_rules_from_json(self.rules_path)
        self._rules_cache = rules
        self._cache_expires_at = datetime.now(timezone.utc) + self.cache_ttl
        return rules

    def _filter_rules(
        self,
        rules: Sequence[ComplianceRule],
        *,
        module: str,
        language: str,
        channel: str,
        regulator_scope: Optional[Sequence[str]],
    ) -> List[ComplianceRule]:
        module_u = (module or "GENERIC").upper()
        lang_l = (language or "en").lower()
        channel_u = (channel or "ALL").upper()
        regulator_scope_u = [r.upper() for r in regulator_scope] if regulator_scope else None

        out: List[ComplianceRule] = []
        for rule in rules:
            if rule.module not in ("GENERIC", module_u):
                continue
            if rule.language not in ("all", lang_l):
                continue
            if rule.channel not in ("ALL", channel_u):
                continue
            if regulator_scope_u and rule.regulator.upper() not in regulator_scope_u:
                continue
            out.append(rule)
        return out

    def _matches(self, rule: ComplianceRule, text: str, context: Dict[str, Any]) -> bool:
        pt = (rule.pattern_type or "").upper()

        if pt == "TEXT":
            return (rule.pattern or "").lower() in (text or "").lower()

        if pt == "REGEX":
            try:
                return re.search(rule.pattern, text or "", flags=re.IGNORECASE) is not None
            except re.error:
                logger.exception("Invalid regex in compliance rule id=%s", rule.id)
                if self._is_fail_closed_context(context):
                    raise
                return False

        if pt == "INTENT":
            tags = context.get("intent_tags") or []
            wanted = (rule.pattern or "").upper()
            return wanted in {str(t).upper() for t in tags}

        if pt == "SEMANTIC":
            return self._check_semantic_violation(rule.pattern, text)

        return False

    def _check_semantic_violation(self, concept: str, text: str) -> bool:
        """
        Check if text violates a concept using LLM.
        Returns True if violation text is detected.
        """
        if not self._gemini_configured or not text or not concept:
            return False
            
        try:
            prompt = f"""
            Task: Compliance Violation Detection.
            
            Policy Violation Concept: "{concept}"
            
            Text to Analyze:
            "{text}"
            
            Does the text contain, imply, or encourage the violation concept described above? 
            Respond with exactly "YES" or "NO".
            """
            
            response = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            result_text = (response.choices[0].message.content or "").strip()
            
            return "YES" in result_text.upper()
            
        except Exception as e:
            logger.warning(f"Semantic compliance check failed: {e}")
            return False

    def _extract_query_text(self, context: Dict[str, Any]) -> str:
        # Support multiple historical keys used across the pipeline.
        for key in ("user_query", "query", "input", "prompt"):
            val = context.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _critical_category(self, context: Dict[str, Any]) -> Optional[str]:
        """Return the critical category module name (e.g. FRAUD/TAX) if this context should be fail-closed."""
        if not self._fail_closed_enabled:
            return None

        module_u = str(context.get("module", "GENERIC") or "GENERIC").upper()
        query_text = self._extract_query_text(context)
        ql = (query_text or "").lower()

        # Module-based criticality.
        if module_u in self._fail_closed_modules:
            return module_u

        # Keyword-based criticality (protect against mis-inference).
        # Note: In India, "PIN code"/"pincode" commonly refers to postal codes. Avoid false positives.
        if re.search(r"\b(otp|cvv|password|passcode)\b", ql):
            return "FRAUD"
        if re.search(r"\bpin\b", ql) and not re.search(r"\bpin\s*code\b|\bpincode\b", ql):
            return "FRAUD"
        if re.search(r"\b(fraud|scam|phish|phishing|otp\s*scam)\b", ql):
            return "FRAUD"
        if re.search(r"\b(tax|gst|itr|tds)\b", ql) and re.search(
            r"\b(evad(e|ing)|hide|conceal|fake\s+invoice|black\s+money)\b",
            ql,
        ):
            return "TAX"

        return None

    def _is_fail_closed_context(self, context: Dict[str, Any]) -> bool:
        return self._critical_category(context) is not None

    def _fail_closed_safe_answer(self, category: Optional[str]) -> str:
        if category == "TAX":
            return (
                "I can’t help with tax evasion (e.g., hiding income, fake invoices, or concealing tax liability). "
                "If you want, I can help with compliant tax planning options or explain the relevant rules."
            )

        # Default to fraud/credential safety.
        return (
            "I can’t help with sharing or using OTP/PIN/CVV/passwords. "
            "Never share OTP/PIN/CVV/passwords with anyone—even if they claim to be bank/app support. "
            "If you suspect fraud or already shared it, contact your bank via official channels immediately."
        )

    def compliance_check(self, answer_text: str, context: Dict[str, Any]) -> ComplianceResult:
        try:
            rules_all = self._load_rules_cached()
            rules = self._filter_rules(
                rules_all,
                module=str(context.get("module", "GENERIC")),
                language=str(context.get("language", "en")),
                channel=str(context.get("channel", "ALL")),
                regulator_scope=context.get("regulator_scope"),
            )

            triggered: List[int] = []
            modified = answer_text or ""
            appended: Set[str] = set()

            query_text = self._extract_query_text(context)

            for rule in rules:
                # Evaluate compliance triggers against BOTH:
                # - the model output (what we will modify)
                # - the user's original query (to catch requests like "guarantee returns" even if
                #   the model avoided echoing that exact phrase in its response)
                if not (self._matches(rule, modified, context) or (query_text and self._matches(rule, query_text, context))):
                    continue

                triggered.append(rule.id)

                rt = (rule.rule_type or "WARN").upper()
                # Backwards/forwards-compat: treat MODIFY_FORCE as FORCE_SAFE_ANSWER.
                if rt == "MODIFY_FORCE":
                    rt = "FORCE_SAFE_ANSWER"
                sev = (rule.severity or "LOW").upper()

                if rt in ("BLOCK", "FORCE_SAFE_ANSWER") and sev in ("HIGH", "CRITICAL"):
                    # Force-safe answer behavior: do not throw; caller decides.
                    return ComplianceResult(status="BLOCKED", final_answer=rule.message or "", triggered_rule_ids=triggered)

                if rt == "MODIFY_REPLACE":
                    # Replace *matched span* with message if regex, else replace literal pattern.
                    if (rule.pattern_type or "").upper() == "REGEX":
                        try:
                            before = modified
                            modified = re.sub(rule.pattern, rule.message, modified, flags=re.IGNORECASE)
                            # If rule triggered via query text but the pattern isn't present in the answer,
                            # fall back to appending the compliance message.
                            if before == modified and rule.message and rule.message not in appended:
                                modified = modified.rstrip() + "\n\n" + rule.message
                                appended.add(rule.message)
                        except re.error:
                            logger.exception("Regex replace failed for compliance rule id=%s", rule.id)
                            if self._is_fail_closed_context(context):
                                raise
                    else:
                        target = rule.pattern
                        if target:
                            before = modified
                            modified = modified.replace(target, rule.message)
                            if before == modified and rule.message and rule.message not in appended:
                                modified = modified.rstrip() + "\n\n" + rule.message
                                appended.add(rule.message)

                elif rt == "MODIFY_APPEND":
                    if rule.message and rule.message not in appended:
                        modified = modified.rstrip() + "\n\n" + rule.message
                        appended.add(rule.message)

                elif rt == "WARN":
                    logger.warning("[WARNING] Compliance WARN triggered: rule_id=%s", rule.id)

                else:
                    logger.warning("[WARNING] Unknown compliance rule_type=%s (rule_id=%s)", rt, rule.id)

            return ComplianceResult(status="OK", final_answer=modified, triggered_rule_ids=triggered)

        except Exception as e:
            critical_category = self._critical_category(context)
            if critical_category:
                logger.error(
                    "[ERROR] Compliance check failed (fail-closed): category=%s error=%s",
                    critical_category,
                    e,
                )
                return ComplianceResult(
                    status="BLOCKED",
                    final_answer=self._fail_closed_safe_answer(critical_category),
                    triggered_rule_ids=[],
                )

            logger.error("[ERROR] Compliance check failed (fail-open): %s", e)
            return ComplianceResult(status="ERROR", final_answer=answer_text or "", triggered_rule_ids=[])

    async def compliance_check_async(self, answer_text: str, context: Dict[str, Any]) -> ComplianceResult:
        return await asyncio.to_thread(self.compliance_check, answer_text, context)
