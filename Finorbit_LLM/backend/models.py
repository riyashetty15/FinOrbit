"""
Pydantic models for API requests and responses

Provides type-safe request/response schemas for the FastAPI endpoints,
replacing the previous scattered model definitions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List


# ==================== Request Models ====================

class QueryRequest(BaseModel):
    """
    User query request model.

    This is the primary input to the /query endpoint in the unified backend.
    It replaces the previous scattered dict-based request handling.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User query text (1-10,000 characters)"
    )
    userId: str = Field(
        ...,
        min_length=1,
        description="Unique user identifier"
    )
    conversationId: str = Field(
        ...,
        min_length=1,
        description="Conversation thread ID for session management"
    )
    profileHint: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional natural-language profile context (age, income, etc.)"
    )

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not just whitespace"""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "query": "What are the income tax slabs for FY 2024-25?",
                "userId": "user_123",
                "conversationId": "conv_456"
            }
        }


# ==================== Response Models ====================

class QueryResponse(BaseModel):
    """
    Successful query response model with full pipeline metadata.
    """

    response: str = Field(
        ...,
        description="Agent-generated response text"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score from validation pipeline (0.0-1.0)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether the agent needs more information from the user"
    )
    agents: List[str] = Field(
        default=[],
        description="List of all specialist agents used to process the query"
    )

    # --- Extended metadata for UI ---
    recommended_action: Optional[str] = Field(
        default=None,
        description="Pipeline recommended action: approve/clarify/partial/refuse"
    )
    compliance_status: Optional[str] = Field(
        default=None,
        description="Compliance check result: OK/BLOCKED/SKIPPED"
    )
    validation_checks: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Individual validation check results"
    )
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Retrieved evidence sources / citations"
    )
    profile_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Profile fields used in this request"
    )
    pipeline_steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Pipeline execution steps for workflow indicator"
    )
    mode: Optional[str] = Field(
        default=None,
        description="Response mode: info/guidance/action"
    )
    follow_ups: Optional[List[str]] = Field(
        default=None,
        description="Smart follow-up suggestions"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "example": {
                "response": "The income tax slabs for FY 2024-25 under the new regime are...",
                "confidence_score": 0.85,
                "needs_clarification": False,
                "agents": ["rag_agent", "tax_planner"],
                "recommended_action": "approve",
                "compliance_status": "OK",
                "mode": "guidance"
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response model with detailed information.

    This provides detailed error information to help users understand
    why their query was blocked or failed validation.

    User decision: Return detailed errors with reasons (not generic messages).
    """

    error: str = Field(
        ...,
        description="High-level error message"
    )
    error_type: str = Field(
        ...,
        description="Error category (e.g., pre_validation_failure, input_guardrail_violation)"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional structured error details"
    )
    blocked_reason: Optional[str] = Field(
        None,
        description="Specific reason why the query/response was blocked"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "examples": [
                {
                    "name": "PII Detected",
                    "value": {
                        "error": "Your query cannot be processed due to safety concerns.",
                        "error_type": "pre_validation_failure",
                        "blocked_reason": "PII detected: aadhaar, pan; Never share sensitive information",
                        "details": {
                            "pii_detected": True,
                            "pii_types": ["aadhaar", "pan"],
                            "content_risk": False
                        }
                    }
                },
                {
                    "name": "Validation Failed",
                    "value": {
                        "error": "The response failed validation checks.",
                        "error_type": "post_validation_failure",
                        "blocked_reason": "Tax slab amounts incorrect; Response contains guarantees",
                        "details": {
                            "grounding_score": 0.85,
                            "numeric_score": 0.45,
                            "regulatory_score": 0.60
                        }
                    }
                }
            ]
        }


# ==================== Internal Models (for Guardrails) ====================

class GuardrailResult(BaseModel):
    """
    Result from input or output guardrail agent.

    This is used internally by the guardrail system.
    NOT exposed in API responses.
    """

    allowed: bool = Field(
        ...,
        description="Whether the query/response is allowed"
    )
    blocked_category: Optional[str] = Field(
        None,
        description="Category of violation if blocked (e.g., 'illegal', 'adult', 'politics')"
    )
    blocked_reason: Optional[str] = Field(
        None,
        description="Reason for blocking"
    )

    # For output guardrail compatibility
    safe: Optional[bool] = Field(
        None,
        description="Alternative field name for output guardrail (maps to 'allowed')"
    )
    issues: Optional[str] = Field(
        None,
        description="Comma-separated list of issues (for output guardrail)"
    )

    class Config:
        """Pydantic configuration"""
        json_schema_extra = {
            "examples": [
                {
                    "name": "Blocked Query",
                    "value": {
                        "allowed": False,
                        "blocked_category": "illegal",
                        "blocked_reason": "Query contains instructions for illegal activity"
                    }
                },
                {
                    "name": "Allowed Query",
                    "value": {
                        "allowed": True,
                        "blocked_category": None,
                        "blocked_reason": None
                    }
                }
            ]
        }


# ==================== Utility Functions ====================

def map_output_guardrail_to_result(output_result: Dict[str, Any]) -> GuardrailResult:
    """
    Convert output guardrail format to GuardrailResult.

    Output guardrail uses 'safe' and 'issues', but GuardrailResult uses 'allowed'.
    This function normalizes the format.

    Args:
        output_result: Output guardrail result with 'safe' and 'issues' fields

    Returns:
        GuardrailResult with normalized fields
    """
    safe = output_result.get("safe", True)

    return GuardrailResult(
        allowed=safe,
        blocked_category="safety_violation" if not safe else None,
        blocked_reason=output_result.get("issues") if not safe else None,
        safe=safe,
        issues=output_result.get("issues")
    )
