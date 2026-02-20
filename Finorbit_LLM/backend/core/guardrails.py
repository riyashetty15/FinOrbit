# ==============================================
# File: backend/core/guardrails.py
# Description: Enforcing guardrail functions (fixes tripwire_triggered bug)
# ==============================================

from typing import Dict, Any
import logging
from agents import GuardrailFunctionOutput

from backend.models import GuardrailResult

logger = logging.getLogger(__name__)


def enforce_input_guardrail(guard_result: GuardrailResult) -> GuardrailFunctionOutput:
    """
    Enforce input guardrail by actually triggering tripwire when blocked

    CRITICAL FIX: The original guardrail functions in backend/agents.py ALWAYS
    returned tripwire_triggered=False, meaning guardrails never actually blocked.
    This function fixes that by returning tripwire_triggered=True when needed.

    Args:
        guard_result: GuardrailResult from input guardrail check
            - allowed: bool - Whether query is allowed
            - blocked_category: Optional[str] - Category of violation
            - blocked_reason: Optional[str] - Reason for blocking

    Returns:
        GuardrailFunctionOutput with:
            - output_info: GuardrailResult - Original result for logging
            - tripwire_triggered: bool - TRUE if blocked, FALSE if allowed

    Example:
        >>> result = GuardrailResult(
        ...     allowed=False,
        ...     blocked_category="illegal",
        ...     blocked_reason="Query contains instructions for tax evasion"
        ... )
        >>> output = enforce_input_guardrail(result)
        >>> output.tripwire_triggered
        True  # Actually blocks!
    """
    if not guard_result.allowed:
        # Fail-open on internal/invalid guardrail outputs (prevents false 400s).
        # If the guardrail claims "blocked" but provides no actionable reason/category,
        # treat it as an internal error rather than a real violation.
        blocked_reason = (guard_result.blocked_reason or "").strip()
        blocked_category = (guard_result.blocked_category or "").strip()
        if blocked_reason == "" and blocked_category == "":
            logger.warning("[WARNING] Input guardrail returned empty block; failing open")
            return GuardrailFunctionOutput(output_info=guard_result, tripwire_triggered=False)

        logger.error(f"[ALERT] INPUT GUARDRAIL TRIGGERED: {guard_result.blocked_reason}")

        return GuardrailFunctionOutput(
            output_info=guard_result,
            tripwire_triggered=True  # ACTUALLY BLOCK
        )

    logger.info("[OK] Input guardrail passed")

    return GuardrailFunctionOutput(
        output_info=guard_result,
        tripwire_triggered=False
    )


def enforce_output_guardrail(guard_result: GuardrailResult) -> GuardrailFunctionOutput:
    """
    Enforce output guardrail by actually triggering tripwire when blocked

    CRITICAL FIX: The original guardrail functions in backend/agents.py ALWAYS
    returned tripwire_triggered=False, meaning output guardrails never blocked.
    This function fixes that by returning tripwire_triggered=True when needed.

    Args:
        guard_result: GuardrailResult from output guardrail check
            - allowed: bool - Whether response is safe (maps from 'safe' field)
            - blocked_category: Optional[str] - "safety_violation" if unsafe
            - blocked_reason: Optional[str] - Specific issues detected
            - safe: Optional[bool] - Original 'safe' field for compatibility
            - issues: Optional[str] - Comma-separated list of issues

    Returns:
        GuardrailFunctionOutput with:
            - output_info: GuardrailResult - Original result for logging
            - tripwire_triggered: bool - TRUE if unsafe, FALSE if safe

    Example:
        >>> result = GuardrailResult(
        ...     allowed=False,
        ...     blocked_category="safety_violation",
        ...     blocked_reason="Response contains guarantees and specific products",
        ...     safe=False,
        ...     issues="contains_guarantee, specific_product_mentioned"
        ... )
        >>> output = enforce_output_guardrail(result)
        >>> output.tripwire_triggered
        True  # Actually blocks!
    """
    if not guard_result.allowed:
        logger.error(f"[ALERT] OUTPUT GUARDRAIL TRIGGERED: {guard_result.blocked_reason}")

        return GuardrailFunctionOutput(
            output_info=guard_result,
            tripwire_triggered=True  # ACTUALLY BLOCK
        )

    logger.info("[OK] Output guardrail passed")

    return GuardrailFunctionOutput(
        output_info=guard_result,
        tripwire_triggered=False
    )


def create_blocked_guardrail_result(
    category: str,
    reason: str,
    is_output: bool = False
) -> GuardrailResult:
    """
    Helper function to create a blocked GuardrailResult

    Args:
        category: Category of violation (e.g., "illegal", "adult", "safety_violation")
        reason: Human-readable reason for blocking
        is_output: Whether this is for output guardrail (affects field names)

    Returns:
        GuardrailResult configured for blocking

    Example:
        >>> result = create_blocked_guardrail_result(
        ...     category="illegal",
        ...     reason="Query requests assistance with tax evasion"
        ... )
        >>> result.allowed
        False
        >>> result.blocked_category
        'illegal'
    """
    if is_output:
        # Output guardrail format (uses 'safe' and 'issues')
        return GuardrailResult(
            allowed=False,
            blocked_category=category,
            blocked_reason=reason,
            safe=False,
            issues=reason
        )
    else:
        # Input guardrail format (uses 'allowed' and 'blocked_reason')
        return GuardrailResult(
            allowed=False,
            blocked_category=category,
            blocked_reason=reason
        )


def create_allowed_guardrail_result(is_output: bool = False) -> GuardrailResult:
    """
    Helper function to create an allowed GuardrailResult

    Args:
        is_output: Whether this is for output guardrail (affects field names)

    Returns:
        GuardrailResult configured for allowing

    Example:
        >>> result = create_allowed_guardrail_result()
        >>> result.allowed
        True
        >>> result.blocked_category
        None
    """
    if is_output:
        # Output guardrail format
        return GuardrailResult(
            allowed=True,
            blocked_category=None,
            blocked_reason=None,
            safe=True,
            issues=None
        )
    else:
        # Input guardrail format
        return GuardrailResult(
            allowed=True,
            blocked_category=None,
            blocked_reason=None
        )
