# ==============================================
# File: backend/agents/agent_helpers.py
# Description: Helper utilities for specialist agents (profile validation, missing field requests)
# ==============================================

from typing import Dict, Any, List, Optional


class ProfileValidator:
    """
    Helper class for validating user profiles and detecting missing fields

    Used by specialist agents to:
    1. Check if profile has required fields for personalized queries
    2. Generate user-friendly requests for missing information
    3. Determine if query can be answered with general information
    """

    @staticmethod
    def get_missing_fields(profile: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """
        Check which required fields are missing from profile

        Args:
            profile: User profile dictionary
            required_fields: List of field names that are required

        Returns:
            List of missing field names

        Example:
            >>> profile = {"age": 30, "income": None, "occupation": None}
            >>> ProfileValidator.get_missing_fields(profile, ["age", "income", "occupation"])
            ['income', 'occupation']
        """
        missing = []
        for field in required_fields:
            value = profile.get(field)
            # Consider None, empty string, or 0 as missing
            if value is None or value == "" or (isinstance(value, (int, float)) and value == 0 and field in ["income", "age"]):
                missing.append(field)
        return missing

    @staticmethod
    def has_complete_profile(profile: Dict[str, Any], required_fields: List[str]) -> bool:
        """
        Check if profile has all required fields

        Args:
            profile: User profile dictionary
            required_fields: List of required field names

        Returns:
            True if all fields present, False otherwise
        """
        return len(ProfileValidator.get_missing_fields(profile, required_fields)) == 0

    @staticmethod
    def generate_missing_fields_message(
        missing_fields: List[str],
        agent_name: str = "Finorbit",
        query_context: Optional[str] = None
    ) -> str:
        """
        Generate user-friendly message requesting missing profile fields

        Args:
            missing_fields: List of missing field names
            agent_name: Name of the agent making the request
            query_context: Optional context about what user was asking

        Returns:
            User-friendly message requesting the information

        Example:
            >>> msg = ProfileValidator.generate_missing_fields_message(
            ...     ["income", "age"],
            ...     "Tax Planner",
            ...     "calculate your tax liability"
            ... )
            >>> print(msg)
            To calculate your tax liability accurately, I'll need some additional information:

            • Annual Income: Your total yearly income
            • Age: Your age (affects tax slabs and deductions)

            Could you please provide these details so I can give you personalized tax advice?
        """
        if not missing_fields:
            return ""

        # Field descriptions
        field_descriptions = {
            "income": "Your annual income (total yearly earnings)",
            "age": "Your age (affects tax slabs, deductions, and investment recommendations)",
            "occupation": "Your occupation type (salaried, business owner, self-employed, retired)",
            "risk_tolerance": "Your risk tolerance for investments (low, moderate, high)",
            "investment_horizon": "Your investment timeline (short-term: <3 years, medium-term: 3-7 years, long-term: >7 years)",
            "dependents": "Number of people financially dependent on you",
            "cibil_score": "Your CIBIL/credit score (if known)",
            "savings": "Your current savings amount",
            "expenses": "Your monthly or annual expenses",
            "retirement_age": "The age you plan to retire",
            "life_expectancy": "Expected lifespan (for retirement planning)",
        }

        # Build message
        intro = f"To {query_context or 'provide personalized advice'}, I'll need some additional information:\n\n"

        field_list = []
        for field in missing_fields:
            field_name = field.replace("_", " ").title()
            description = field_descriptions.get(field, "")
            if description:
                field_list.append(f"• **{field_name}**: {description}")
            else:
                field_list.append(f"• **{field_name}**")

        fields_text = "\n".join(field_list)

        outro = "\n\nCould you please provide these details so I can give you personalized advice?"

        return intro + fields_text + outro


def is_general_query(query: str, intent: str) -> bool:
    """
    Determine if query should be answered with general information

    Args:
        query: User query text
        intent: Intent classification from router ("general" or "personalized")

    Returns:
        True if query should get general information, False if needs personalization

    Note:
        This relies primarily on the router's intent classification.
        Can be extended with additional agent-specific logic if needed.
    """
    return intent == "general"


def requires_profile(query: str, intent: str) -> bool:
    """
    Determine if query requires user profile for accurate response

    Args:
        query: User query text
        intent: Intent classification from router

    Returns:
        True if profile is required, False if general info is sufficient
    """
    return intent == "personalized"


def format_general_info_response(info: str, agent_name: str = "Finorbit") -> str:
    """
    Format a general information response with appropriate disclaimer

    Args:
        info: The general information to provide
        agent_name: Name of the agent providing the info

    Returns:
        Formatted response with disclaimer

    Example:
        >>> info = "Tax slabs for FY 2024-25: ₹0-3L: 0%, ₹3L-7L: 5%..."
        >>> formatted = format_general_info_response(info, "Tax Planner")
        >>> print(formatted)
        Tax slabs for FY 2024-25: ₹0-3L: 0%, ₹3L-7L: 5%...

        [NOTE] This is general information. For personalized tax calculations based on your specific income and circumstances, please provide your details.
    """
    disclaimer = (
        "\n\n[NOTE] **Note:** This is general information. "
        "For personalized advice based on your specific circumstances, "
        "please provide your details."
    )
    return info + disclaimer
