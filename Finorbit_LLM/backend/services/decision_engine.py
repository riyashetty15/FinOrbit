from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import json

@dataclass
class DecisionOutput:
    """
    Standardized structural output for all financial workflows.
    Encourages "decision first, explanation second" architecture.
    """
    recommendations: List[str]
    reasoning_factors: List[str]
    assumptions: List[str]
    risks: List[str]
    required_disclaimers: List[str]
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    # Optional follow-up questions for clarification
    follow_ups: List[str] = field(default_factory=list)
    
    # Metadata for UI/Frontend
    ui_components: Optional[List[Dict[str, Any]]] = None # e.g. [{"type": "chart", "data": ...}]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class DecisionEngine:
    """
    Service to validate and construct financial decisions with evidence gating.
    
    Evidence Gating Rules:
    - If needs_evidence=True and coverage != "sufficient" → Refuse and ask follow-ups
    - No compliance claims without citations
    - All regulatory statements must be backed by EvidencePack
    """
    
    @staticmethod
    def validate_schema(output: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate if a raw dictionary matches the DecisionOutput schema.
        """
        required_fields = [
            "recommendations", 
            "reasoning_factors", 
            "assumptions", 
            "risks", 
            "required_disclaimers"
        ]
        
        missing = [f for f in required_fields if f not in output]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
            
        return True, "Valid"

    @staticmethod
    def create_safe_refusal(reason: str, follow_ups: Optional[List[str]] = None) -> DecisionOutput:
        """
        Helper to create a standard "I can't help you" decision with optional follow-ups.
        
        Args:
            reason: Explanation for refusal
            follow_ups: Optional list of clarifying questions to ask user
        """
        return DecisionOutput(
            recommendations=["Cannot provide specific financial advice on this query."],
            reasoning_factors=[reason],
            assumptions=[],
            risks=[],
            required_disclaimers=["FinOrbit is an AI assistant and not a SEBI registered advisor."],
            follow_ups=follow_ups or [],
            confidence=1.0
        )
    
    @staticmethod
    def create_evidence_refusal(
        coverage: str,
        query: str,
        module: str,
        rejection_reason: Optional[str] = None
    ) -> DecisionOutput:
        """
        Create refusal when evidence is insufficient for regulatory/compliance query.
        
        Args:
            coverage: Coverage level from EvidencePack ("insufficient" or "partial")
            query: Original user query
            module: Domain/module being queried
            rejection_reason: Optional detailed rejection reason
        
        Returns:
            DecisionOutput with refusal and contextual follow-up questions
        """
        # Generate contextual follow-up questions based on module
        follow_ups = DecisionEngine._generate_follow_ups(query, module, coverage)
        
        if coverage == "insufficient":
            reason = (
                "Insufficient regulatory evidence retrieved from knowledge base. "
                "Unable to provide accurate compliance information without verified sources."
            )
            if rejection_reason:
                reason += f" Details: {rejection_reason}"
            
            return DecisionOutput(
                recommendations=[
                    "Cannot determine answer without verified regulatory sources.",
                    "Please provide more specific details about your query."
                ],
                reasoning_factors=[
                    reason,
                    "No verified citations found in knowledge base.",
                    "Regulatory queries require authoritative source documents."
                ],
                assumptions=[],
                risks=[
                    "Answering without citations could provide incorrect information.",
                    "Compliance errors can have serious legal/financial consequences."
                ],
                required_disclaimers=[
                    "This is not financial or legal advice.",
                    "Always consult a qualified professional for compliance matters."
                ],
                follow_ups=follow_ups,
                citations=[],
                confidence=0.0
            )
        
        else:  # partial coverage
            reason = (
                "Only partial evidence available. Retrieved information may be incomplete. "
                "Cannot provide comprehensive answer without sufficient verified sources."
            )
            if rejection_reason:
                reason += f" Details: {rejection_reason}"
            
            return DecisionOutput(
                recommendations=[
                    "Partial information available but insufficient for complete answer.",
                    "Recommend verifying with authoritative sources or providing more context."
                ],
                reasoning_factors=[
                    reason,
                    "Limited verified citations found.",
                    "Additional context needed for accurate response."
                ],
                assumptions=[],
                risks=[
                    "Incomplete information could lead to incorrect decisions.",
                    "Partial evidence may not reflect recent updates or full requirements."
                ],
                required_disclaimers=[
                    "This is not financial or legal advice.",
                    "Information may be incomplete."
                ],
                follow_ups=follow_ups,
                citations=[],
                confidence=0.3
            )
    
    @staticmethod
    def _generate_follow_ups(query: str, module: str, coverage: str) -> List[str]:
        """
        Generate contextual follow-up questions based on query, module, and coverage.
        
        Args:
            query: Original user query
            module: Domain module (credit, taxation, etc.)
            coverage: Coverage level
        
        Returns:
            List of 2-4 clarifying questions
        """
        query_lower = query.lower()
        follow_ups = []
        
        # Generic follow-ups applicable to all queries
        follow_ups.append("Which specific regulator or authority's rules do you need information about?")
        
        # Module-specific follow-ups
        if module in ["credit", "credits_loans"]:
            follow_ups.append("Are you asking about NBFC, banks, or specific lending institutions?")
            if "npa" in query_lower or "classification" in query_lower:
                follow_ups.append("Do you need information on NBFC-ND-SI, NBFC-D, or NBFC-NDSI-R regulations?")
        
        elif module in ["taxation", "tax_planner"]:
            follow_ups.append("Which assessment year or financial year does your query relate to?")
            follow_ups.append("Are you asking about central tax (CBDT) or state-level taxation?")
        
        elif module == "investment":
            follow_ups.append("Which type of investment instrument are you asking about (equity, mutual funds, bonds)?")
            if "sebi" in query_lower:
                follow_ups.append("Do you need SEBI regulations for RIAs, portfolio managers, or MF distributors?")
        
        elif module == "insurance":
            follow_ups.append("Which type of insurance are you asking about (life, health, vehicle, property)?")
            if "irdai" in query_lower:
                follow_ups.append("Are you asking about insurer regulations or policyholder guidelines?")
        
        elif module == "retirement":
            follow_ups.append("Are you asking about NPS, EPF, PPF, or other retirement schemes?")
        
        # Time-sensitive follow-ups if coverage is insufficient
        if coverage == "insufficient":
            follow_ups.append("Are you looking for the latest rules, or information from a specific time period?")
        
        # Return max 4 follow-ups
        return follow_ups[:4]
    
    @staticmethod
    def check_evidence_gate(
        needs_evidence: bool,
        evidence_pack: Optional[Any],  # Type: EvidencePack from retrieval_service
        query: str,
        module: str
    ) -> Optional[DecisionOutput]:
        """
        Gate check: Refuse to answer if evidence is required but insufficient.
        
        Args:
            needs_evidence: Whether query requires regulatory grounding (from RouteIntent)
            evidence_pack: EvidencePack from retrieval service (or None if not retrieved)
            query: Original user query
            module: Domain module
        
        Returns:
            DecisionOutput with refusal if gate fails, None if gate passes
        """
        # If evidence not needed, pass gate
        if not needs_evidence:
            return None
        
        # If evidence needed but not provided, refuse
        if evidence_pack is None:
            return DecisionEngine.create_evidence_refusal(
                coverage="insufficient",
                query=query,
                module=module,
                rejection_reason="No evidence retrieval was performed."
            )
        
        # If evidence coverage is insufficient or partial, refuse
        if evidence_pack.coverage != "sufficient":
            return DecisionEngine.create_evidence_refusal(
                coverage=evidence_pack.coverage,
                query=query,
                module=module,
                rejection_reason=evidence_pack.rejection_reason
            )
        
        # Evidence gate passed
        return None
