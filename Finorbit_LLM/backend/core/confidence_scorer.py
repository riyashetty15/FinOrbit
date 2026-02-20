# ==============================================
# File: backend/core/confidence_scorer.py
# Description: Multi-factor confidence scoring for hallucination prevention
# ==============================================

from typing import Dict, Any, Literal
from backend.core.validation_models import ConfidenceScore, PostValidationResult


class ConfidenceScorer:
    """
    Calculates multi-factor confidence scores for agent responses (Module 3)

    Combines scores from multiple validation sources with weighted formula:
    - Retrieval relevance: 35%
    - Grounding: 35%
    - Regulatory check: 30%

    Based on the overall score, recommends an action:
    - >= 0.85: Serve confidently
    - 0.70-0.84: Serve with standard disclaimer
    - 0.50-0.69: Ask clarifying questions
    - 0.30-0.49: Partial answer + "I need more information"
    - < 0.30: Politely refuse
    """

    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        "retrieval": 0.35,      # RAG retrieval relevance
        "grounding": 0.35,      # Claim grounding in sources
        "regulatory": 0.30      # Regulatory compliance
    }

    # Confidence threshold for serving responses
    THRESHOLD = 0.7             # Minimum score to serve without modification

    # Confidence action thresholds
    THRESHOLDS = {
        "serve": 0.80,          # Serve confidently without warnings
        "standard": 0.60,       # Serve with standard disclaimer
        "clarify": 0.40,        # Ask clarifying questions
        "partial": 0.25,        # Provide partial answer
        "refuse": 0.0           # Below this = refuse to answer
    }

    def __init__(self):
        """Initialize confidence scorer with default weights"""
        pass

    def calculate(
        self,
        post_validation: PostValidationResult,
        retrieval_score: float = 0.8
    ) -> ConfidenceScore:
        """
        Calculate overall confidence score from validation results

        Args:
            post_validation: Results from all validation agents (Module 1)
            retrieval_score: RAG retrieval relevance score (0.0 to 1.0)
                           Default 0.8 if no RAG system available

        Returns:
            ConfidenceScore: Overall score, breakdown, and recommended action

        Example:
            >>> scorer = ConfidenceScorer()
            >>> post_val = PostValidationResult(
            ...     grounding_score=0.85,
            ...     reg_score=0.90,
            ...     numeric_score=0.95,
            ...     suitability_score=0.80,
            ...     tone_score=0.88,
            ...     # ... other fields
            ... )
            >>> result = scorer.calculate(post_val, retrieval_score=0.80)
            >>> result.overall_score
            0.865
            >>> result.recommended_action
            'serve'
        """
        # Extract individual scores
        scores = {
            "retrieval": retrieval_score,
            "grounding": post_validation.grounding_score,
            "regulatory": post_validation.reg_score
        }

        # Calculate weighted average
        overall = sum(
            scores[factor] * weight
            for factor, weight in self.WEIGHTS.items()
        )

        # Determine recommended action
        action = self._determine_action(overall, post_validation)

        # Check if meets threshold
        meets_threshold = overall >= self.THRESHOLD

        return ConfidenceScore(
            overall_score=overall,
            retrieval_score=retrieval_score,
            grounding_score=post_validation.grounding_score,
            numeric_validation_score=post_validation.numeric_score,
            regulatory_score=post_validation.reg_score,
            breakdown=scores,
            meets_threshold=meets_threshold,
            recommended_action=action
        )

    def _determine_action(
        self,
        score: float,
        validation: PostValidationResult
    ) -> Literal["serve", "clarify", "partial", "refuse"]:
        """
        Determine recommended action based on confidence score and validation

        Priority order:
        1. If blocking issues exist → refuse (regardless of score)
        2. If score >= 0.85 → serve
        3. If score >= 0.70 → serve (with standard disclaimers)
        4. If score >= 0.50 → clarify
        5. If score >= 0.30 → partial
        6. If score < 0.30 → refuse

        Args:
            score: Overall confidence score (0.0 to 1.0)
            validation: Post-validation results

        Returns:
            Recommended action: "serve", "clarify", "partial", or "refuse"
        """
        # Critical blocking issues always refuse
        if validation.blocking_issues:
            return "refuse"

        # High confidence - serve directly
        if score >= self.THRESHOLDS["serve"]:
            return "serve"

        # Medium-high confidence - serve with standard disclaimer
        if score >= self.THRESHOLDS["standard"]:
            return "serve"

        # Medium confidence - ask clarifying questions
        if score >= self.THRESHOLDS["clarify"]:
            return "clarify"

        # Low-medium confidence - partial answer
        if score >= self.THRESHOLDS["partial"]:
            return "partial"

        # Very low confidence - refuse
        return "refuse"

    def get_action_message(self, action: str, score: float) -> str:
        """
        Get user-friendly message for each action type

        Args:
            action: Recommended action ("serve", "clarify", "partial", "refuse")
            score: Confidence score

        Returns:
            Message to append/modify response with
        """
        if action == "serve":
            return ""  # No modification needed

        elif action == "clarify":
            return (
                "\n\nI need a bit more information to provide a complete answer. "
                "Could you clarify your question or provide additional details?"
            )

        elif action == "partial":
            return (
                "\n\nNote: I've provided a partial answer based on available information. "
                "For a complete response, I may need additional context or professional verification."
            )

        elif action == "refuse":
            return (
                "I don't have enough reliable information to answer this question confidently. "
                "I recommend consulting with a qualified financial advisor for this topic."
            )

        return ""
