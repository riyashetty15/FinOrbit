# ==============================================
# File: backend/agents/safety/content_risk_filter.py
# Description: Content risk filtering agent for illegal/harmful content
# ==============================================

from typing import Dict, Any, List, Tuple
import re
from backend.agents.safety.base_safety import BaseSafetyAgent


class ContentRiskFilterAgent(BaseSafetyAgent):
    """
    Filters queries containing illegal or harmful content (Module 2, Safety Agent #4)

    Blocks queries related to:
    - Tax evasion
    - Money laundering
    - Self-harm
    - Fraud/scams
    - Illegal financial activities
    - Violence/threats

    **Action**: Blocks execution if harmful content detected
    **Severity**: CRITICAL - Cannot provide assistance with illegal activities
    """

    # Keyword patterns for different risk categories
    RISK_PATTERNS = {
        "tax_evasion": [
            r'\bevade\s+tax', r'\btax\s+evasion\b', r'\bavoid\s+paying\s+tax',
            r'\bhide\s+income\b', r'\bunderreport\b', r'\bfake\s+invoice',
            r'\bblack\s+money\b', r'\bundeclared\s+income\b'
        ],
        "money_laundering": [
            r'\bmoney\s+launder', r'\blaunder\s+money\b', r'\bclean\s+money\b',
            r'\bshell\s+company\b', r'\boffshore\s+account\b', r'\bhawala\b'
        ],
        "self_harm": [
            r'\bhurt\s+myself\b', r'\bkill\s+myself\b', r'\bsuicide\b',
            r'\bself\s+harm\b', r'\bend\s+my\s+life\b'
        ],
        "fraud": [
            r'\bponzi\s+scheme\b', r'\bpyramid\s+scheme\b', r'\bscam\b',
            r'\bfake\s+company\b', r'\bfraudulent\b', r'\bforge\s+document'
        ],
        "illegal_activities": [
            r'\bdrug\s+money\b', r'\bterror\s+financing\b', r'\bbribery\b',
            r'\bkickback\b', r'\billegal\s+transaction\b'
        ]
    }

    def __init__(self):
        """Initialize content risk filter with compiled patterns"""
        super().__init__(name="content_risk_filter")

        # Compile all patterns for performance
        self.compiled_patterns = {}
        for category, patterns in self.RISK_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]

    def check(self, query: str, profile: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check query for harmful/illegal content

        Args:
            query: User query text
            profile: User profile (not used for content filtering)

        Returns:
            Tuple of (is_safe, issues, metadata):
                - is_safe: False if harmful content detected
                - issues: List of risk categories detected
                - metadata: {
                    "risk_flags": ["tax_evasion", "fraud"],
                    "matched_patterns": ["evade tax", "fake invoice"],
                    "risk_level": "critical" | "high" | "medium"
                }

        Example:
            >>> filter = ContentRiskFilterAgent()
            >>> is_safe, issues, meta = filter.check(
            ...     "How can I evade tax and hide my income?",
            ...     {}
            ... )
            >>> is_safe
            False
            >>> meta['risk_flags']
            ['tax_evasion']
        """
        risk_flags = []
        matched_patterns = []

        # Scan query for each risk category
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(query)
                if match:
                    risk_flags.append(category)
                    matched_patterns.append(match.group())
                    break  # One match per category is enough

        # Determine if safe
        is_safe = len(risk_flags) == 0

        # Create issues list
        issues = []
        if not is_safe:
            unique_flags = sorted(set(risk_flags))
            issues.append(f"Content risk detected: {', '.join(unique_flags)}")
            issues.append("I cannot provide assistance with illegal or harmful activities")

        # Determine risk level
        risk_level = "none"
        if len(risk_flags) > 0:
            # Self-harm is always critical
            if "self_harm" in risk_flags:
                risk_level = "critical"
                issues.append("If you're experiencing thoughts of self-harm, please contact a crisis helpline immediately")
            # Multiple flags = critical
            elif len(risk_flags) >= 2:
                risk_level = "critical"
            # Single flag = high
            else:
                risk_level = "high"

        # Create metadata
        metadata = {
            "risk_flags": sorted(set(risk_flags)),
            "matched_patterns": matched_patterns,
            "risk_level": risk_level
        }

        return self._create_result(is_safe, issues, metadata)

    def get_safe_response_message(self, risk_flags: List[str]) -> str:
        """
        Generate appropriate response for detected risk

        Args:
            risk_flags: List of detected risk categories

        Returns:
            Safe response message
        """
        if "self_harm" in risk_flags:
            return (
                "I'm concerned about your wellbeing. If you're experiencing thoughts of self-harm, "
                "please reach out to a crisis helpline or mental health professional immediately. "
                "In India, you can call AASRA at 91-9820466726."
            )

        return (
            "I cannot provide assistance with illegal activities or financial practices that "
            "violate laws and regulations. If you have legitimate questions about tax planning "
            "or financial management, please rephrase your question within legal boundaries."
        )
