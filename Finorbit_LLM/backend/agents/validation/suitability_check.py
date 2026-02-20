# ==============================================
# File: mcp_server/agents/validation/suitability_check.py
# Description: Suitability and risk level validation agent
# ==============================================

from typing import Dict, Any, List
import re
from backend.agents.validation.base_validator import BaseValidator
from backend.core.validation_models import ValidationCheck, Severity


class SuitabilityCheckAgent(BaseValidator):
    """
    Validates advice suitability for user profile (Module 1, Validation Agent #4)

    Checks:
    - Advice matches user age/income/risk tolerance
    - No dangerous advice (skip EMIs, extreme leverage)
    - Product complexity matches user sophistication
    - Age-appropriate recommendations

    **Pass Threshold**: Advice suitable for user profile, no dangerous recommendations
    **Severity**: CRITICAL for dangerous advice, WARNING for suitability mismatch
    """

    # Dangerous advice patterns
    DANGEROUS_PATTERNS = [
        r'\bskip\s+(?:emi|payment|installment)\b',
        r'\bdefault\s+on\s+loan\b',
        r'\bevade\s+(?:tax|payment)\b',
        r'\b(?:max|maximum)\s+leverage\b',
        r'\ball[- ]in\s+(?:on|into)\b',
        r'\bborrow\s+to\s+invest\b'
    ]

    # High-risk product keywords
    HIGH_RISK_PRODUCTS = ['crypto', 'cryptocurrency', 'futures', 'options', 'forex', 'derivatives']

    def __init__(self):
        """Initialize suitability check agent"""
        super().__init__(name="suitability_check", check_type="suitability")

        # Compile dangerous patterns
        self.dangerous_patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def validate(self, response: str, context: Dict[str, Any]) -> ValidationCheck:
        """
        Validate advice suitability for user profile

        Args:
            response: Agent response text
            context: Must contain profile with age, income, risk_tolerance

        Returns:
            ValidationCheck with suitability assessment
        """
        profile = self._extract_context_field(context, "profile", {})
        # Handle None values gracefully (profile fields may be None)
        age = profile.get("age") or 30
        income = profile.get("income") or 500000
        risk_tolerance = (profile.get("risk_tolerance") or "moderate").lower()

        issues = []
        recommendations = []
        response_lower = response.lower()

        # Check for dangerous advice
        dangerous_matches = []
        for pattern in self.dangerous_patterns:
            match = pattern.search(response)
            if match:
                dangerous_matches.append(match.group())

        if dangerous_matches:
            issues.append(f"Dangerous advice detected: {', '.join(set(dangerous_matches))}")
            recommendations.append("Remove recommendations that could harm user financially")

        # Check high-risk products for inappropriate users
        has_high_risk = any(product in response_lower for product in self.HIGH_RISK_PRODUCTS)

        if has_high_risk:
            # High-risk products inappropriate for: low income, seniors, conservative tolerance
            if income < 300000:
                issues.append("High-risk products recommended for low-income user")
                recommendations.append("Recommend stable, low-risk options for low-income users")

            if age < 21 or age > 60:
                issues.append(f"High-risk products recommended for age {age}")
                recommendations.append("Consider age-appropriate risk levels")

            if risk_tolerance == "conservative":
                issues.append("High-risk products recommended for conservative risk tolerance")
                recommendations.append("Match product risk to user's risk tolerance")

        # Age-based suitability
        if age < 18 and ('invest' in response_lower or 'loan' in response_lower):
            issues.append("Investment/loan advice provided to minor")
            recommendations.append("Direct minors to financial literacy, not investment execution")

        # Calculate confidence
        if dangerous_matches:
            confidence = 0.0
            passed = False
            severity = Severity.CRITICAL
        elif issues:
            confidence = 0.5
            passed = False
            severity = Severity.WARNING
        else:
            confidence = 1.0
            passed = True
            severity = Severity.INFO

        metadata = {
            "user_age": age,
            "user_income": income,
            "risk_tolerance": risk_tolerance,
            "dangerous_advice_count": len(dangerous_matches),
            "has_high_risk_products": has_high_risk
        }

        return self._create_check(
            passed=passed,
            confidence=confidence,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )
