# ==============================================
# File: mcp_server/agents/validation/regulatory_consistency.py
# Description: Regulatory compliance validation agent
# ==============================================

from typing import Dict, Any, List
import re
from backend.agents.validation.base_validator import BaseValidator
from backend.core.validation_models import ValidationCheck, Severity


class RegulatoryConsistencyAgent(BaseValidator):
    """
    Validates regulatory compliance of responses (Module 1, Validation Agent #2)

    Checks for:
    - Forbidden guarantees ("guaranteed returns", "zero risk", "100% safe")
    - Required disclaimers for investment/tax advice
    - Market risk warnings for SIP/mutual funds
    - Proper qualification of statements ("may", "typically", "generally")

    **Pass Threshold**: No violations, all required disclaimers present
    **Severity**: CRITICAL for violations, INFO for missing disclaimers
    """

    # Forbidden phrases that indicate regulatory violations
    FORBIDDEN_GUARANTEES = [
        r'\bguaranteed\s+returns?\b',
        r'\bzero\s+risk\b',
        r'\bno\s+risk\b',
        r'\b100%\s+(?:safe|secure|profit|returns?)\b',
        r'\brisk[- ]free\b',
        r'\bcompletely\s+safe\b',
        r'\balways\s+(?:profit|gain|win)\b'
    ]

    # Investment/market risk keywords requiring disclaimers
    INVESTMENT_KEYWORDS = ['sip', 'mutual fund', 'equity', 'stock', 'invest', 'market']

    # Tax advice keywords requiring disclaimers
    TAX_KEYWORDS = ['tax', 'deduction', '80c', 'itr', 'filing']

    def __init__(self):
        """Initialize regulatory consistency agent"""
        super().__init__(name="regulatory_consistency", check_type="regulatory")

        # Compile forbidden patterns
        self.forbidden_patterns = [re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_GUARANTEES]

    def validate(self, response: str, context: Dict[str, Any]) -> ValidationCheck:
        """
        Validate regulatory compliance of response

        Args:
            response: Agent response text
            context: Context dict (query used to determine if disclaimers needed)

        Returns:
            ValidationCheck with compliance results
        """
        issues = []
        recommendations = []
        response_lower = response.lower()

        # Check for forbidden guarantees (context-aware)
        violations = []
        for pattern in self.forbidden_patterns:
            match = pattern.search(response)
            if match:
                violations.append(match.group())

        # Determine whether the response is in a market-linked investing context.
        # For fixed-income products like fixed deposits, "guaranteed returns" can be a legitimate phrase
        # (though still better to qualify). We should not hard-block those.
        module = str(context.get("module", "")).upper()
        query_text = str(context.get("query", "") or "").lower()

        fixed_deposit_context = re.search(r"\b(fixed\s+deposit|fixed\s+deposits|term\s+deposit|fd|fds)\b", query_text) is not None
        market_linked_context = module == "SIP_INVESTMENT" or any(kw in response_lower for kw in self.INVESTMENT_KEYWORDS)

        if violations:
            issues.append(f"Forbidden guarantee language detected: {', '.join(set(violations))}")
            recommendations.append("Avoid absolute guarantees; qualify statements and add appropriate disclaimers")

        # Check if disclaimers are needed and present
        needs_investment_disclaimer = any(kw in response_lower for kw in self.INVESTMENT_KEYWORDS)
        needs_tax_disclaimer = any(kw in response_lower for kw in self.TAX_KEYWORDS)

        has_disclaimer = self._has_disclaimer(response)

        if needs_investment_disclaimer and not has_disclaimer:
            issues.append("Investment advice provided without proper disclaimer")
            recommendations.append("Add: 'This is educational information. Investments are subject to market risk.'")

        if needs_tax_disclaimer and not has_disclaimer:
            issues.append("Tax advice provided without proper disclaimer")
            recommendations.append("Add: 'This is general information. Consult a tax professional for your specific situation.'")

        # Calculate confidence
        if violations:
            passed = False
            if market_linked_context and not fixed_deposit_context:
                confidence = 0.0  # Critical violation in market-linked investing
                severity = Severity.CRITICAL
            else:
                confidence = 0.7  # Non-blocking: context may allow fixed returns, but wording should be qualified
                severity = Severity.WARNING
        elif issues:
            confidence = 0.6  # Missing disclaimers
            passed = False
            severity = Severity.INFO
        else:
            confidence = 1.0
            passed = True
            severity = Severity.INFO

        metadata = {
            "violations_found": len(violations),
            "needs_investment_disclaimer": needs_investment_disclaimer,
            "needs_tax_disclaimer": needs_tax_disclaimer,
            "has_disclaimer": has_disclaimer
        }

        return self._create_check(
            passed=passed,
            confidence=confidence,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )

    def _has_disclaimer(self, text: str) -> bool:
        """Check if text contains a disclaimer"""
        disclaimer_indicators = [
            'disclaimer', 'not financial advice', 'educational', 'consult',
            'subject to market risk', 'for informational purposes'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in disclaimer_indicators)
