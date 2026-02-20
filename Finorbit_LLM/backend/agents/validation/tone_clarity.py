# ==============================================
# File: mcp_server/agents/validation/tone_clarity.py
# Description: Tone and clarity validation agent
# ==============================================

from typing import Dict, Any, List
import re
from backend.agents.validation.base_validator import BaseValidator
from backend.core.validation_models import ValidationCheck, Severity


class ToneClarityAgent(BaseValidator):
    """
    Validates tone, clarity, and readability of responses (Module 1, Validation Agent #5)

    Checks:
    - Step count (max 5-7 recommended)
    - Jargon usage (flagged if unexplained)
    - Sentence length (avg < 25 words)
    - Readability score (simple language)
    - Use of examples/analogies

    **Pass Threshold**: Simple language, clear structure
    **Severity**: INFO (always non-blocking, for quality improvement)
    """

    # Financial jargon that should be explained
    JARGON_TERMS = [
        'amortization', 'annuity', 'liquidity', 'corpus', 'maturity',
        'vesting', 'appreciation', 'depreciation', 'dividend', 'portfolio',
        'diversification', 'allocation', 'rebalancing', 'volatility', 'equity'
    ]

    # Maximum recommended steps
    MAX_RECOMMENDED_STEPS = 7

    # Target sentence length
    TARGET_SENTENCE_LENGTH = 25

    def __init__(self):
        """Initialize tone and clarity agent"""
        super().__init__(name="tone_clarity", check_type="tone")

    def validate(self, response: str, context: Dict[str, Any]) -> ValidationCheck:
        """
        Validate tone and clarity of response

        Args:
            response: Agent response text
            context: Context dict (not used for tone validation)

        Returns:
            ValidationCheck with tone/clarity assessment
        """
        issues = []
        recommendations = []

        # Count steps/bullet points
        step_count = self._count_steps(response)
        if step_count > self.MAX_RECOMMENDED_STEPS:
            issues.append(f"Response has {step_count} steps (recommended: {self.MAX_RECOMMENDED_STEPS} or fewer)")
            recommendations.append("Break complex advice into smaller, focused sections")

        # Check for unexplained jargon
        jargon_used = self._find_unexplained_jargon(response)
        if jargon_used:
            issues.append(f"Jargon used without explanation: {', '.join(jargon_used[:3])}")
            recommendations.append("Explain technical terms or use simpler alternatives")

        # Analyze sentence length
        avg_sentence_length = self._calculate_avg_sentence_length(response)
        if avg_sentence_length > self.TARGET_SENTENCE_LENGTH:
            issues.append(f"Average sentence length: {avg_sentence_length} words (target: <{self.TARGET_SENTENCE_LENGTH})")
            recommendations.append("Use shorter sentences for better readability")

        # Check for examples/analogies
        has_examples = self._has_examples(response)

        # Calculate readability score (simple approximation)
        readability_score = self._calculate_readability_score(avg_sentence_length, len(jargon_used))

        # Calculate confidence (tone check is always informational)
        if readability_score >= 0.7:
            confidence = readability_score
            passed = True
        else:
            confidence = readability_score
            passed = False

        # Tone check is always INFO severity (never blocks)
        severity = Severity.INFO

        metadata = {
            "step_count": step_count,
            "jargon_count": len(jargon_used),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "readability_score": round(readability_score, 2),
            "has_examples": has_examples
        }

        return self._create_check(
            passed=passed,
            confidence=confidence,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )

    def _count_steps(self, text: str) -> int:
        """Count numbered steps or bullet points"""
        # Count numbered lists (1., 2., etc. or 1), 2), etc.)
        numbered = len(re.findall(r'^\s*\d+[\.\)]\s+', text, re.MULTILINE))

        # Count bullet points
        bullets = len(re.findall(r'^\s*[-*•]\s+', text, re.MULTILINE))

        return max(numbered, bullets)

    def _find_unexplained_jargon(self, text: str) -> List[str]:
        """Find jargon terms used without explanation"""
        text_lower = text.lower()
        unexplained = []

        for term in self.JARGON_TERMS:
            if term in text_lower:
                # Check if explained (term followed by explanation patterns)
                explanation_patterns = [
                    f'{term}.*?(?:means?|refers? to|is|are)',
                    f'(?:means?|refers? to).*?{term}'
                ]

                is_explained = any(re.search(p, text_lower) for p in explanation_patterns)

                if not is_explained:
                    unexplained.append(term)

        return unexplained

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)

    def _has_examples(self, text: str) -> bool:
        """Check if response includes examples"""
        example_indicators = ['for example', 'e.g.', 'such as', 'like', 'example:']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in example_indicators)

    def _calculate_readability_score(self, avg_sentence_length: float, jargon_count: int) -> float:
        """Simple readability score (0.0-1.0)"""
        # Score based on sentence length (shorter is better)
        length_score = max(0, 1.0 - (avg_sentence_length - 15) / 30)

        # Penalty for jargon
        jargon_penalty = min(0.3, jargon_count * 0.1)

        score = max(0, length_score - jargon_penalty)
        return min(1.0, score)
