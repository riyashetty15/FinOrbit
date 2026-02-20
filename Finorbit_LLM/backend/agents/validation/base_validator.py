# ==============================================
# File: mcp_server/agents/validation/base_validator.py
# Description: Base class for all validation agents
# ==============================================

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.core.validation_models import ValidationCheck, Severity, create_validation_check


class BaseValidator(ABC):
    """
    Abstract base class for all validation agents

    All validation agents (grounding, regulatory, numeric, suitability, tone)
    inherit from this class to ensure consistent interface and behavior.

    Subclasses must implement the validate() method.
    """

    def __init__(self, name: str, check_type: str):
        """
        Initialize base validator

        Args:
            name: Human-readable name of the validator (e.g., "grounding_check")
            check_type: Type of validation (e.g., "grounding", "regulatory", "numeric")
        """
        self.name = name
        self.check_type = check_type

    @abstractmethod
    def validate(self, response: str, context: Dict[str, Any]) -> ValidationCheck:
        """
        Validate the agent response against validation criteria

        This is the main method that each validation agent must implement.
        It receives the agent's response and context, then returns a ValidationCheck
        indicating whether the response passed validation.

        Args:
            response: The agent's response text (typically the "summary" field)
            context: Additional context for validation, typically includes:
                - query: str - Original user query
                - profile: Dict[str, Any] - User profile data
                - retrieved_passages: List[str] - Sources from RAG (for grounding)
                - agent: str - Which agent generated the response
                - transactions: List[Dict] - User transaction data (if available)

        Returns:
            ValidationCheck: Result of validation with confidence score, issues, and recommendations

        Example:
            >>> validator = GroundingCheckAgent()
            >>> result = validator.validate(
            ...     response="Interest rates are 8.5% per annum",
            ...     context={
            ...         "query": "What are home loan rates?",
            ...         "retrieved_passages": ["Home loans start at 8.5%..."]
            ...     }
            ... )
            >>> result.passed
            True
            >>> result.confidence
            0.85
        """
        pass

    def _create_check(
        self,
        passed: bool,
        confidence: float,
        severity: Severity,
        issues: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationCheck:
        """
        Helper method to create ValidationCheck objects with consistent structure

        This method wraps create_validation_check() to automatically include
        the validator's name and check_type.

        Args:
            passed: Whether validation passed
            confidence: Confidence score (0.0 to 1.0)
            severity: Severity level (CRITICAL, WARNING, INFO)
            issues: List of issues found (optional)
            recommendations: List of recommended actions (optional)
            metadata: Additional metadata (optional)

        Returns:
            ValidationCheck instance

        Example:
            >>> return self._create_check(
            ...     passed=False,
            ...     confidence=0.45,
            ...     severity=Severity.WARNING,
            ...     issues=["Only 3/7 claims are grounded"],
            ...     recommendations=["Add source citations", "Verify numeric claims"]
            ... )
        """
        return create_validation_check(
            agent_name=self.name,
            check_type=self.check_type,
            passed=passed,
            confidence=confidence,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )

    def _extract_context_field(self, context: Dict[str, Any], field: str, default: Any = None) -> Any:
        """
        Safely extract a field from context with fallback

        Helper method to handle missing context fields gracefully.

        Args:
            context: Context dictionary
            field: Field name to extract
            default: Default value if field is missing

        Returns:
            Field value or default
        """
        return context.get(field, default)
