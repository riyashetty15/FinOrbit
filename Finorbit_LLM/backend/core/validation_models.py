# ==============================================
# File: backend/core/validation_models.py
# Description: Data structures for validation, safety, and confidence scoring
# ==============================================

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# Severity levels for validation issues
class Severity(str, Enum):
    """Severity classification for validation issues"""
    CRITICAL = "critical"      # Blocks response - illegal content, PII, severe violations
    WARNING = "warning"        # Logs but continues - missing disclaimers, informal tone
    INFO = "info"              # Informational only - metrics, analysis data


# Individual validation check result
@dataclass
class ValidationCheck:
    """
    Result from a single validation agent check

    Used by all validation agents (grounding, regulatory, numeric, suitability, tone)
    to return structured validation results.
    """
    agent_name: str                      # e.g., "grounding_check", "regulatory_consistency"
    check_type: str                      # e.g., "grounding", "regulatory", "numeric"
    passed: bool                         # True if validation passed
    confidence: float                    # 0.0 to 1.0 confidence score
    severity: Severity                   # CRITICAL, WARNING, or INFO
    issues: List[str] = field(default_factory=list)           # List of issues found
    recommendations: List[str] = field(default_factory=list)  # Recommended actions
    metadata: Dict[str, Any] = field(default_factory=dict)    # Additional context


# Pre-validation (safety) result
@dataclass
class PreValidationResult:
    """
    Aggregated results from safety agents (Module 2)

    Runs BEFORE agent execution to check for:
    - PII in queries
    - Mis-selling risk
    - Age/category warnings
    - Content risk (illegal activities, self-harm, etc.)
    """
    pii_detected: bool                                        # True if PII found
    pii_entities: List[Dict[str, str]]                        # [{type, value, location}, ...]
    mis_selling_risk: float                                   # 0.0 to 1.0 risk score
    age_category_warning: Optional[str] = None                # e.g., "minor_detected", "low_income_high_risk"
    content_risk_flags: List[str] = field(default_factory=list)  # e.g., ["tax_evasion", "self_harm"]
    blocking_issues: List[str] = field(default_factory=list)  # Critical issues that block execution
    should_block: bool = False                                # True if execution should be blocked
    safe_to_proceed: bool = True                              # False if critical safety issues found


# Post-validation result
@dataclass
class PostValidationResult:
    """
    Aggregated results from validation agents (Module 1)

    Runs AFTER agent execution to validate:
    - Grounding (claims backed by sources)
    - Regulatory compliance (no guarantees, proper disclaimers)
    - Numeric accuracy (tax slabs, limits verified)
    - Suitability (advice matches user profile)
    - Tone & clarity (simple language, clear structure)
    """
    grounding_ok: bool                                        # Grounding check passed
    grounding_score: float                                    # 0.0 to 1.0
    reg_ok: bool                                              # Regulatory check passed
    reg_score: float                                          # 0.0 to 1.0
    numeric_ok: bool                                          # Numeric validation passed
    numeric_score: float                                      # 0.0 to 1.0
    suitability_ok: bool                                      # Suitability check passed
    suitability_score: float                                  # 0.0 to 1.0
    tone_ok: bool                                             # Tone & clarity check passed
    tone_score: float                                         # 0.0 to 1.0
    validation_checks: List[ValidationCheck] = field(default_factory=list)  # Detailed check results
    blocking_issues: List[str] = field(default_factory=list)  # Critical issues that block response
    warnings: List[str] = field(default_factory=list)         # Non-blocking warnings
    should_block: bool = False                                # True if response should be blocked


# Confidence scoring (hallucination prevention)
@dataclass
class ConfidenceScore:
    """
    Multi-factor weighted confidence score (Module 3)

    Combines scores from multiple sources:
    - Retrieval relevance (30%)
    - Grounding (30%)
    - Numeric validation (20%)
    - Regulatory check (20%)

    Determines action based on threshold:
    - >= 0.85: Serve confidently
    - 0.70-0.84: Serve with disclaimer
    - 0.50-0.69: Ask clarifying questions
    - 0.30-0.49: Partial answer
    - < 0.30: Politely refuse
    """
    overall_score: float                                      # Weighted average (0.0 to 1.0)
    retrieval_score: float                                    # RAG retrieval relevance
    grounding_score: float                                    # Claim grounding score
    numeric_validation_score: float                           # Numeric accuracy score
    regulatory_score: float                                   # Regulatory compliance score
    breakdown: Dict[str, float] = field(default_factory=dict) # Detailed score breakdown
    meets_threshold: bool = False                             # True if >= 0.7
    recommended_action: Literal["serve", "clarify", "partial", "refuse"] = "serve"


# Validation metadata for state tracking
@dataclass
class ValidationMetadata:
    """
    Metadata about validation execution for performance monitoring
    """
    pre_validation_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    post_validation_timestamp: Optional[str] = None
    confidence_calculation_timestamp: Optional[str] = None
    total_validation_time_ms: float = 0.0                     # Total time spent in validation
    agents_executed: List[str] = field(default_factory=list)  # List of agents that ran


# Audit trail entry for compliance logging
@dataclass
class AuditEntry:
    """
    Single audit log entry for compliance and debugging

    Logged by AuditLoggerAgent to maintain a compliance trail
    """
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_type: str = ""                                      # e.g., "pii_detected", "low_confidence", "validation_failed"
    agent_name: str = ""                                      # Which agent generated this entry
    details: Dict[str, Any] = field(default_factory=dict)     # Event-specific details
    severity: Severity = Severity.INFO                        # Severity of the event
    action_taken: str = ""                                    # e.g., "blocked", "sanitized", "logged"


# Helper function to create ValidationCheck easily
def create_validation_check(
    agent_name: str,
    check_type: str,
    passed: bool,
    confidence: float,
    severity: Severity,
    issues: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ValidationCheck:
    """
    Helper function to create ValidationCheck objects with defaults

    Args:
        agent_name: Name of the validation agent
        check_type: Type of check (grounding, regulatory, etc.)
        passed: Whether validation passed
        confidence: Confidence score (0.0 to 1.0)
        severity: Severity level (CRITICAL, WARNING, INFO)
        issues: List of issues found (optional)
        recommendations: List of recommendations (optional)
        metadata: Additional metadata (optional)

    Returns:
        ValidationCheck instance
    """
    return ValidationCheck(
        agent_name=agent_name,
        check_type=check_type,
        passed=passed,
        confidence=confidence,
        severity=severity,
        issues=issues or [],
        recommendations=recommendations or [],
        metadata=metadata or {}
    )
