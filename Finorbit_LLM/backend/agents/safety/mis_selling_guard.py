# ==============================================
# File: backend/agents/safety/mis_selling_guard.py
# Description: Mis-selling prevention guard agent
# ==============================================

from typing import Dict, Any, List, Tuple
import re
from backend.agents.safety.base_safety import BaseSafetyAgent


class MisSellingGuardAgent(BaseSafetyAgent):
    """
    Prevents mis-selling by detecting specific product recommendations (Module 2, Safety Agent #2)

    Flags queries/responses containing:
    - Specific product names (ICICI iProtect, HDFC Click2Protect)
    - Brand-specific mutual fund recommendations
    - Direct "buy X product" instructions
    - Promotional language ("best product", "guaranteed returns")

    **Generic Guidance OK**: "Consider term life insurance"
    **Specific Product NOT OK**: "Buy ICICI iProtect Plan"

    **Action**: Calculate risk score, warn if > 0.5
    **Severity**: WARNING - Informational for monitoring
    """

    # Specific insurance product names to flag
    INSURANCE_PRODUCTS = [
        r'\bICICI\s+iProtect\b', r'\bHDFC\s+Click2Protect\b',
        r'\bLIC\s+Jeevan\s+Anand\b', r'\bMax\s+Life\s+Smart\s+Secure\b',
        r'\bBajaj\s+Allianz\b', r'\bSBI\s+Life\s+Sampoorna\b'
    ]

    # Specific mutual fund names
    MUTUAL_FUND_PRODUCTS = [
        r'\bICICI\s+Prudential\s+\w+\s+Fund\b',
        r'\bHDFC\s+\w+\s+Fund\b',
        r'\bSBI\s+\w+\s+Fund\b',
        r'\bAxis\s+\w+\s+Fund\b'
    ]

    # Bank-specific products
    BANK_PRODUCTS = [
        r'\bHDFC\s+Bank\s+loan\b', r'\bICICI\s+Bank\s+loan\b',
        r'\bSBI\s+home\s+loan\b', r'\bAxis\s+Bank\s+credit\s+card\b'
    ]

    # Promotional/sales language
    PROMOTIONAL_PATTERNS = [
        r'\bbest\s+product\b', r'\btop\s+rated\s+\w+\b',
        r'\bguaranteed\s+returns\b', r'\bhighest\s+returns\b',
        r'\byou\s+should\s+buy\b', r'\bI\s+recommend\s+buying\b',
        r'\bpurchase\s+immediately\b', r'\blimited\s+time\s+offer\b'
    ]

    def __init__(self):
        """Initialize mis-selling guard with compiled patterns"""
        super().__init__(name="mis_selling_guard")

        # Compile all patterns
        self.insurance_patterns = [re.compile(p, re.IGNORECASE) for p in self.INSURANCE_PRODUCTS]
        self.mutual_fund_patterns = [re.compile(p, re.IGNORECASE) for p in self.MUTUAL_FUND_PRODUCTS]
        self.bank_patterns = [re.compile(p, re.IGNORECASE) for p in self.BANK_PRODUCTS]
        self.promotional_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROMOTIONAL_PATTERNS]

    def check(self, query: str, profile: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check query for specific product mentions (mis-selling risk)

        Args:
            query: User query text
            profile: User profile (not used for mis-selling detection)

        Returns:
            Tuple of (is_safe, issues, metadata):
                - is_safe: Always True (non-blocking, informational)
                - issues: List of warnings if risk detected
                - metadata: {
                    "risk_score": 0.0-1.0,
                    "specific_products": ["ICICI iProtect", "HDFC Click2Protect"],
                    "promotional_language": ["best product", "guaranteed returns"],
                    "category": "insurance" | "mutual_fund" | "bank" | "mixed",
                    "risk_level": "low" | "medium" | "high"
                }

        Example:
            >>> guard = MisSellingGuardAgent()
            >>> is_safe, issues, meta = guard.check(
            ...     "Should I buy ICICI iProtect? It has guaranteed returns.",
            ...     {}
            ... )
            >>> meta['risk_score']
            0.8
            >>> meta['specific_products']
            ['ICICI iProtect']
        """
        specific_products = []
        promotional_language = []
        categories = set()

        # Scan for specific insurance products
        for pattern in self.insurance_patterns:
            match = pattern.search(query)
            if match:
                specific_products.append(match.group())
                categories.add("insurance")

        # Scan for mutual fund products
        for pattern in self.mutual_fund_patterns:
            match = pattern.search(query)
            if match:
                specific_products.append(match.group())
                categories.add("mutual_fund")

        # Scan for bank products
        for pattern in self.bank_patterns:
            match = pattern.search(query)
            if match:
                specific_products.append(match.group())
                categories.add("bank")

        # Scan for promotional language
        for pattern in self.promotional_patterns:
            match = pattern.search(query)
            if match:
                promotional_language.append(match.group())

        # Calculate risk score
        risk_score = 0.0

        # Specific product mentions increase risk
        if len(specific_products) > 0:
            risk_score += 0.4 * min(len(specific_products), 3)  # Max 1.2 for 3+ products

        # Promotional language increases risk
        if len(promotional_language) > 0:
            risk_score += 0.3 * min(len(promotional_language), 2)  # Max 0.6 for 2+ phrases

        # Cap at 1.0
        risk_score = min(risk_score, 1.0)

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Determine if safe (always true for query stage - just monitoring)
        is_safe = True

        # Create issues list
        issues = []
        if risk_score >= 0.5:
            issues.append(f"Mis-selling risk detected (score: {risk_score:.2f})")
            if specific_products:
                issues.append(f"Specific products mentioned: {', '.join(specific_products[:3])}")
            if promotional_language:
                issues.append("Promotional language detected - ensure generic guidance")

        # Determine category
        if not categories:
            category = "none"
        elif len(categories) == 1:
            category = list(categories)[0]
        else:
            category = "mixed"

        # Create metadata
        metadata = {
            "risk_score": round(risk_score, 2),
            "specific_products": specific_products,
            "promotional_language": promotional_language,
            "category": category,
            "risk_level": risk_level
        }

        return self._create_result(is_safe, issues, metadata)

    def sanitize_response(self, response: str) -> str:
        """
        Remove specific product mentions from response (for output sanitization)

        Args:
            response: Agent response text

        Returns:
            Sanitized response with generic terms
        """
        sanitized = response

        # Replace specific products with generic terms
        product_replacements = {
            r'\bICICI\s+iProtect\b': 'term life insurance',
            r'\bHDFC\s+Click2Protect\b': 'term life insurance',
            r'\bLIC\s+Jeevan\s+Anand\b': 'traditional life insurance',
        }

        for pattern, replacement in product_replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized
