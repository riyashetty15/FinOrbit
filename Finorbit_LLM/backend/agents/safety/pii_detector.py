# ==============================================
# File: backend/agents/safety/pii_detector.py
# Description: PII (Personally Identifiable Information) detection agent
# ==============================================

from typing import Dict, Any, List, Tuple
from backend.agents.safety.base_safety import BaseSafetyAgent
from backend.utils.pii_patterns import PIIPatterns, sanitize_pii, extract_pii_context


class PIIDetectorAgent(BaseSafetyAgent):
    """
    Detects PII in user queries (Module 2, Safety Agent #1)

    Scans for sensitive Indian PII including:
    - Aadhaar numbers (12 digits)
    - PAN cards (5 letters + 4 digits + 1 letter)
    - Bank account numbers (9-18 digits)
    - Credit/debit card numbers
    - IFSC codes
    - Phone numbers
    - Email addresses
    - Passport numbers
    - Voter ID
    - Driving licenses
    - UPI IDs

    **Action**: Blocks execution if critical PII detected
    **Severity**: CRITICAL - Must never process queries containing PII
    """

    def __init__(self):
        """Initialize PII detector with all pattern matchers"""
        super().__init__(name="pii_detector")

        # Get all PII patterns
        self.patterns = PIIPatterns.get_all_patterns()

        # Critical PII types that must always block
        self.critical_types = {
            "aadhaar", "pan", "bank_account", "credit_card", "passport"
        }

    def check(self, query: str, profile: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Detect PII in user query

        Args:
            query: User query text
            profile: User profile (not used for PII detection)

        Returns:
            Tuple of (is_safe, issues, metadata):
                - is_safe: False if any PII detected
                - issues: List of PII types found
                - metadata: {
                    "pii_entities": [
                        {
                            "type": "aadhaar",
                            "value": "12****9012",  # Sanitized
                            "location": "query",
                            "position": (10, 22),
                            "context": "...my aadhaar ******** please..."
                        },
                        ...
                    ],
                    "pii_types": ["aadhaar", "phone"],
                    "count": 2,
                    "has_critical_pii": True
                }

        Example:
            >>> detector = PIIDetectorAgent()
            >>> is_safe, issues, meta = detector.check(
            ...     "My Aadhaar is 1234-5678-9012 and PAN is ABCDE1234F",
            ...     {}
            ... )
            >>> is_safe
            False
            >>> issues
            ['PII detected: aadhaar, pan', 'Never share sensitive personal information']
            >>> meta['pii_types']
            ['aadhaar', 'pan']
        """
        detected_pii = []
        pii_entities = []
        has_critical_pii = False

        # Scan for each PII type
        for pii_type, pattern in self.patterns.items():
            matches = pattern.finditer(query)

            for match in matches:
                # Record detection
                detected_pii.append(pii_type)

                # Check if critical
                if pii_type in self.critical_types:
                    has_critical_pii = True

                # Create entity record with sanitized value
                pii_entities.append({
                    "type": pii_type,
                    "value": sanitize_pii(match.group(), pii_type),
                    "location": "query",
                    "position": match.span(),
                    "context": extract_pii_context(query, match.start(), match.end())
                })

        # Determine if safe
        is_safe = len(detected_pii) == 0

        # Create issues list
        issues = []
        if not is_safe:
            unique_types = sorted(set(detected_pii))
            issues.append(f"PII detected: {', '.join(unique_types)}")
            issues.append("Never share sensitive personal information in queries")

            if has_critical_pii:
                issues.append("Critical PII detected - query cannot be processed")

        # Create metadata
        metadata = {
            "pii_entities": pii_entities,
            "pii_types": sorted(set(detected_pii)),
            "count": len(detected_pii),
            "has_critical_pii": has_critical_pii
        }

        return self._create_result(is_safe, issues, metadata)

    def get_safe_query_message(self, detected_types: List[str]) -> str:
        """
        Generate a safe, user-friendly message when PII is detected

        Args:
            detected_types: List of PII types detected

        Returns:
            Safe message to return to user
        """
        pii_type_names = {
            "aadhaar": "Aadhaar number",
            "pan": "PAN card",
            "bank_account": "bank account number",
            "credit_card": "credit/debit card number",
            "passport": "passport number",
            "phone": "phone number",
            "email": "email address",
        }

        detected_names = [
            pii_type_names.get(t, t) for t in detected_types if t in pii_type_names
        ]

        if not detected_names:
            detected_names = ["sensitive personal information"]

        names_str = ", ".join(detected_names[:3])  # Limit to first 3
        if len(detected_names) > 3:
            names_str += f", and {len(detected_names) - 3} more"

        return (
            f"I cannot process queries containing {names_str}. "
            "Please remove any personal identifying information and ask your question again. "
            "For personalized advice, please speak with our human financial advisors."
        )
