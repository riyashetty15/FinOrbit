# ==============================================
# File: backend/agents/safety/age_category_guard.py
# Description: Age and category-based protection guard
# ==============================================

from typing import Dict, Any, List, Tuple
from backend.agents.safety.base_safety import BaseSafetyAgent


class AgeCategoryGuardAgent(BaseSafetyAgent):
    """
    Age and income category protection guard (Module 2, Safety Agent #3)

    Provides extra protection for vulnerable user categories:
    - Minors (age < 18): No complex products, extra caution
    - Low income (< 300k): No high-risk investments
    - Seniors (age > 60): Emphasis on safety, liquidity
    - Students: Focus on financial literacy

    **Action**: Log warning, modify context (non-blocking unless critical)
    **Severity**: WARNING - Guides response but doesn't block
    """

    # Age thresholds
    AGE_MINOR = 18
    AGE_SENIOR = 60

    # Income thresholds (annual, in INR)
    INCOME_LOW = 300000      # Below 3 lakh
    INCOME_MEDIUM = 1000000  # 3-10 lakh
    INCOME_HIGH = 2500000    # Above 25 lakh

    def __init__(self):
        """Initialize age/category guard"""
        super().__init__(name="age_category_guard")

    def check(self, query: str, profile: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check user profile for vulnerable categories

        Args:
            query: User query text
            profile: User profile with age, income, occupation, etc.

        Returns:
            Tuple of (is_safe, issues, metadata):
                - is_safe: Always True (non-blocking)
                - issues: List of category warnings
                - metadata: {
                    "age_category": "minor" | "young_adult" | "mid_career" | "senior",
                    "income_category": "low" | "medium" | "high",
                    "warnings": ["minor_detected", "low_income"],
                    "restrictions": ["no_complex_products", "extra_caution"],
                    "recommendations": ["Focus on financial literacy", ...]
                }

        Example:
            >>> guard = AgeCategoryGuardAgent()
            >>> is_safe, issues, meta = guard.check(
            ...     "Should I invest in crypto?",
            ...     {"age": 17, "income": 0}
            ... )
            >>> is_safe
            True  # Non-blocking
            >>> meta['warnings']
            ['minor_detected', 'low_income']
        """
        # Handle None values gracefully (profile fields may be None)
        age = profile.get("age") or 30  # Default to 30 if not provided or None
        income = profile.get("income") or 500000  # Default to 5 lakh if not provided or None
        occupation = (profile.get("occupation") or "").lower()  # Default to empty string if None

        warnings = []
        restrictions = []
        recommendations = []

        # Age-based categorization
        if age < self.AGE_MINOR:
            age_category = "minor"
            warnings.append("minor_detected")
            restrictions.extend([
                "no_complex_products",
                "no_investment_advice",
                "extra_caution"
            ])
            recommendations.extend([
                "Focus on financial literacy",
                "Learn basic savings concepts",
                "Discuss financial goals with parents/guardians"
            ])
        elif age < 30:
            age_category = "young_adult"
            recommendations.append("Build emergency fund first")
        elif age < 50:
            age_category = "mid_career"
        elif age >= self.AGE_SENIOR:
            age_category = "senior"
            warnings.append("senior_citizen")
            restrictions.append("emphasize_safety")
            recommendations.extend([
                "Prioritize capital preservation",
                "Focus on liquid investments",
                "Consider healthcare and insurance needs"
            ])
        else:
            age_category = "pre_retirement"

        # Income-based categorization
        if income < self.INCOME_LOW:
            income_category = "low"
            warnings.append("low_income")
            restrictions.extend([
                "no_high_risk_investments",
                "avoid_complex_derivatives"
            ])
            recommendations.extend([
                "Start with small SIP investments",
                "Build emergency fund (3-6 months expenses)",
                "Focus on increasing income first"
            ])
        elif income < self.INCOME_MEDIUM:
            income_category = "medium"
        elif income < self.INCOME_HIGH:
            income_category = "high"
        else:
            income_category = "very_high"
            recommendations.append("Consider tax planning strategies")

        # Occupation-based considerations
        if "student" in occupation:
            warnings.append("student_profile")
            restrictions.append("limited_income")
            recommendations.extend([
                "Focus on building financial knowledge",
                "Start small savings habit",
                "Avoid debt unless for education"
            ])

        # Special high-risk combinations
        if age < self.AGE_MINOR and "invest" in query.lower():
            warnings.append("minor_investment_query")
            restrictions.append("block_investment_execution")

        # Determine if truly critical (would need blocking)
        # For now, age/category guard is informational only
        is_safe = True

        # Create issues list (informational warnings)
        issues = []
        if warnings:
            issues.append(f"User category warnings: {', '.join(warnings)}")
            issues.append("Response will be tailored for user's age/income category")

        # Create metadata
        metadata = {
            "age_category": age_category,
            "income_category": income_category,
            "warnings": warnings,
            "restrictions": restrictions,
            "recommendations": recommendations,
            "age": age,
            "income": income
        }

        return self._create_result(is_safe, issues, metadata)

    def should_block_complex_products(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if complex financial products should be blocked

        Args:
            metadata: Metadata from check() call

        Returns:
            True if complex products should be avoided
        """
        return "no_complex_products" in metadata.get("restrictions", [])

    def get_category_appropriate_disclaimer(self, metadata: Dict[str, Any]) -> str:
        """
        Get category-appropriate disclaimer for response

        Args:
            metadata: Metadata from check() call

        Returns:
            Disclaimer text
        """
        age_category = metadata.get("age_category")
        income_category = metadata.get("income_category")

        if age_category == "minor":
            return (
                "Note: Financial decisions for minors should be made in consultation "
                "with parents or guardians. This information is for educational purposes only."
            )

        if age_category == "senior":
            return (
                "Note: For senior citizens, capital preservation and liquidity are important. "
                "Please consult with a financial advisor before making investment decisions."
            )

        if income_category == "low":
            return (
                "Note: Building an emergency fund and stable income should be your priority. "
                "Start with small, safe investments and increase gradually."
            )

        return (
            "This is educational information. Please assess your risk tolerance "
            "and financial situation before making investment decisions."
        )
