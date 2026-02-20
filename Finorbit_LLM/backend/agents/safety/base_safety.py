# ==============================================
# File: backend/agents/safety/base_safety.py
# Description: Base class for all safety agents
# ==============================================

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class BaseSafetyAgent(ABC):
    """
    Abstract base class for all safety agents

    All safety agents (PII detector, mis-selling guard, age/category guard,
    content risk filter, audit logger) inherit from this class to ensure
    consistent interface and behavior.

    Safety agents run BEFORE agent execution to catch critical issues
    like PII in queries, illegal content requests, etc.

    Subclasses must implement the check() method.
    """

    def __init__(self, name: str):
        """
        Initialize base safety agent

        Args:
            name: Human-readable name of the safety agent (e.g., "pii_detector")
        """
        self.name = name

    @abstractmethod
    def check(self, query: str, profile: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Perform safety check on user query and profile

        This is the main method that each safety agent must implement.
        It receives the user query and profile, then returns a tuple indicating
        whether it's safe to proceed.

        Args:
            query: User query text (pre-execution)
            profile: User profile data (may be empty dict if not loaded yet)

        Returns:
            Tuple of (is_safe, issues, metadata):
                - is_safe: True if no critical safety issues found
                - issues: List of issue descriptions (empty if safe)
                - metadata: Additional data about the check (e.g., detected entities, risk scores)

        Example:
            >>> agent = PIIDetectorAgent()
            >>> is_safe, issues, metadata = agent.check(
            ...     query="My Aadhaar is 1234-5678-9012",
            ...     profile={}
            ... )
            >>> is_safe
            False
            >>> issues
            ['PII detected: aadhaar']
            >>> metadata['pii_entities']
            [{'type': 'aadhaar', 'value': '12****9012', 'location': 'query'}]
        """
        pass

    def _create_result(
        self,
        is_safe: bool,
        issues: List[str],
        metadata: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Helper method to create consistent return tuples

        Args:
            is_safe: Whether the check passed
            issues: List of issues found
            metadata: Additional metadata

        Returns:
            Tuple of (is_safe, issues, metadata)
        """
        return (is_safe, issues if issues else [], metadata if metadata else {})
