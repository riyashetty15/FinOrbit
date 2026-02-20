# ==============================================
# File: backend/utils/pii_patterns.py
# Description: Regex patterns for detecting Indian PII (Personally Identifiable Information)
# ==============================================

import re
from typing import Dict, Pattern


class PIIPatterns:
    """
    Regex patterns for detecting Indian PII (Aadhaar, PAN, bank accounts, etc.)

    All patterns are pre-compiled for performance.
    """

    # Aadhaar: 12 digits, optionally with spaces/dashes
    # Format: XXXX XXXX XXXX or XXXX-XXXX-XXXX or XXXXXXXXXXXX
    AADHAAR: Pattern = re.compile(
        r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
        re.IGNORECASE
    )

    # PAN: 5 letters, 4 digits, 1 letter
    # Format: ABCDE1234F
    PAN: Pattern = re.compile(
        r'\b[A-Z]{5}\d{4}[A-Z]\b',
        re.IGNORECASE
    )

    # Bank Account: 9-18 digits
    # Note: This is a simple pattern; may have false positives
    BANK_ACCOUNT: Pattern = re.compile(
        r'\b\d{9,18}\b'
    )

    # Credit/Debit Card: 13-19 digits with optional spaces/dashes
    # Format: XXXX XXXX XXXX XXXX or XXXX-XXXX-XXXX-XXXX
    CREDIT_CARD: Pattern = re.compile(
        r'\b(?:\d{4}[\s\-]?){3}\d{4,7}\b'
    )

    # IFSC Code: 4 letters, 0, 6 alphanumeric
    # Format: ABCD0123456
    IFSC: Pattern = re.compile(
        r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
        re.IGNORECASE
    )

    # Indian Phone Number: 10 digits, optionally with +91/0 prefix
    # Format: +91 9876543210, 09876543210, 9876543210
    PHONE: Pattern = re.compile(
        r'\b(?:\+91[\s\-]?|0)?[6-9]\d{9}\b'
    )

    # Email Address: Standard email pattern
    EMAIL: Pattern = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )

    # UPI ID: username@bank
    # Format: username@paytm, user@oksbi, etc.
    UPI: Pattern = re.compile(
        r'\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\b'
    )

    # Passport: 1 letter followed by 7 digits
    # Format: A1234567
    PASSPORT: Pattern = re.compile(
        r'\b[A-Z]\d{7}\b',
        re.IGNORECASE
    )

    # Voter ID: 3 letters followed by 7 digits
    # Format: ABC1234567
    VOTER_ID: Pattern = re.compile(
        r'\b[A-Z]{3}\d{7}\b',
        re.IGNORECASE
    )

    # Driving License: Varies by state, but generally 2 letters, 2 digits, 11 digits
    # Format: DL-1420110012345
    DRIVING_LICENSE: Pattern = re.compile(
        r'\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{11}\b',
        re.IGNORECASE
    )

    @classmethod
    def get_all_patterns(cls) -> Dict[str, Pattern]:
        """
        Get all PII patterns as a dictionary

        Returns:
            Dict mapping PII type name to compiled regex pattern
        """
        return {
            "aadhaar": cls.AADHAAR,
            "pan": cls.PAN,
            "bank_account": cls.BANK_ACCOUNT,
            "credit_card": cls.CREDIT_CARD,
            "ifsc": cls.IFSC,
            "phone": cls.PHONE,
            "email": cls.EMAIL,
            "upi": cls.UPI,
            "passport": cls.PASSPORT,
            "voter_id": cls.VOTER_ID,
            "driving_license": cls.DRIVING_LICENSE,
        }

    @classmethod
    def get_critical_patterns(cls) -> Dict[str, Pattern]:
        """
        Get only critical PII patterns (must always be blocked)

        Returns:
            Dict of critical PII patterns
        """
        return {
            "aadhaar": cls.AADHAAR,
            "pan": cls.PAN,
            "bank_account": cls.BANK_ACCOUNT,
            "credit_card": cls.CREDIT_CARD,
            "passport": cls.PASSPORT,
        }


def sanitize_pii(value: str, pii_type: str = "generic") -> str:
    """
    Sanitize PII value for safe logging

    Masks the middle portion of the value, keeping first 2 and last 2 characters
    visible for debugging purposes.

    Args:
        value: The PII value to sanitize
        pii_type: Type of PII (for type-specific sanitization)

    Returns:
        Sanitized string with middle characters masked

    Examples:
        >>> sanitize_pii("1234567890", "aadhaar")
        '12******90'
        >>> sanitize_pii("ABCDE1234F", "pan")
        'AB******4F'
        >>> sanitize_pii("abc@example.com", "email")
        'ab***@***le.com'
    """
    if not value or len(value) <= 4:
        return "****"

    # Special handling for email
    if pii_type == "email" and "@" in value:
        local, domain = value.split("@", 1)
        local_sanitized = local[:2] + "***" if len(local) > 2 else "***"
        domain_parts = domain.split(".")
        if len(domain_parts) >= 2:
            domain_sanitized = "***" + domain_parts[-2][-2:] + "." + domain_parts[-1]
        else:
            domain_sanitized = "***"
        return f"{local_sanitized}@{domain_sanitized}"

    # Standard masking: show first 2 and last 2 characters
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def extract_pii_context(text: str, match_start: int, match_end: int, context_chars: int = 20) -> str:
    """
    Extract surrounding context for a PII match

    Args:
        text: Full text where PII was found
        match_start: Start position of PII match
        match_end: End position of PII match
        context_chars: Number of characters to include before/after

    Returns:
        Context string with PII sanitized
    """
    start = max(0, match_start - context_chars)
    end = min(len(text), match_end + context_chars)

    context = text[start:end]
    # Replace the PII portion with asterisks
    pii_length = match_end - match_start
    context = context.replace(
        text[match_start:match_end],
        "*" * min(pii_length, 8)
    )

    return f"...{context}..."
