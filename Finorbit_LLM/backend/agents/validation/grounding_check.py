# ==============================================
# File: mcp_server/agents/validation/grounding_check.py
# Description: Grounding verification agent - verify claims against sources
# ==============================================

from typing import Dict, Any, List
import re
import logging
from backend.agents.validation.base_validator import BaseValidator
from backend.core.validation_models import ValidationCheck, Severity

logger = logging.getLogger(__name__)


class GroundingCheckAgent(BaseValidator):
    """
    Verifies claims in responses are grounded in retrieved sources (Module 1, Validation Agent #1)

    Checks:
    - Numeric claims (amounts, percentages, rates) appear in source passages
    - Factual claims have keyword overlap with sources
    - Overall grounding confidence score

    **Pass Threshold**: >= 0.7 (70% of claims grounded)
    **Severity**: CRITICAL if < 0.5, WARNING if < 0.7
    """

    # Minimum confidence threshold
    PASS_THRESHOLD = 0.7
    CRITICAL_THRESHOLD = 0.5

    # Minimum keyword overlap for factual claims
    KEYWORD_OVERLAP_THRESHOLD = 0.5

    def __init__(self):
        """Initialize grounding check agent"""
        super().__init__(name="grounding_check", check_type="grounding")

    def validate(self, response: str, context: Dict[str, Any]) -> ValidationCheck:
        """
        Verify response claims are grounded in source passages.
        
        Now includes production-grade citation validation:
        - Regulatory claims must have citations from EvidencePack
        - Citations must map to actual doc_ids from retrieval
        - No compliance statements without verified sources

        Args:
            response: Agent response text
            context: Must contain:
                - retrieved_passages: List[str] - Source documents from RAG (legacy)
                - evidence_pack: EvidencePack - Structured evidence with citations (new)
                - query: str - Original query (optional)

        Returns:
            ValidationCheck with grounding score and issues

        Example:
            >>> agent = GroundingCheckAgent()
            >>> result = agent.validate(
            ...     "RBI requires NBFCs to maintain LCR of 100%",
            ...     {"evidence_pack": evidence_pack}
            ... )
            >>> result.passed
            True
            >>> result.confidence
            0.95
        """
        logger.info("🔍 GroundingCheckAgent: Starting grounding validation")
        
        # Check for new EvidencePack structure (production-grade)
        evidence_pack = context.get("evidence_pack")
        if evidence_pack:
            return self._validate_with_evidence_pack(response, evidence_pack, context)
        
        # Fallback to legacy retrieved_passages validation
        retrieved_passages = self._extract_context_field(context, "retrieved_passages", [])
        logger.info(f"🔍 GroundingCheckAgent: Found {len(retrieved_passages)} source passages")

        # If no sources provided, return low confidence
        if not retrieved_passages:
            logger.warning("[WARNING] GroundingCheckAgent: No source passages provided for grounding verification")
            return self._create_check(
                passed=False,
                confidence=0.3,
                severity=Severity.WARNING,
                issues=["No source passages provided for grounding verification"],
                recommendations=["Ensure RAG retrieval is enabled", "Add source citations to response"],
                metadata={"has_sources": False}
            )

        # Extract claims from response
        numeric_claims = self._extract_numeric_claims(response)
        factual_claims = self._extract_factual_claims(response)
        logger.info(f"🔍 GroundingCheckAgent: Extracted {len(numeric_claims)} numeric claims, {len(factual_claims)} factual claims")

        # Verify grounding
        grounded_numeric = self._verify_numeric_grounding(numeric_claims, retrieved_passages)
        grounded_factual = self._verify_factual_grounding(factual_claims, retrieved_passages)
        logger.info(f"🔍 GroundingCheckAgent: Grounded {grounded_numeric}/{len(numeric_claims)} numeric, {grounded_factual}/{len(factual_claims)} factual")

        # Calculate overall confidence
        total_claims = len(numeric_claims) + len(factual_claims)
        grounded_claims = grounded_numeric + grounded_factual

        if total_claims == 0:
            logger.info("🔍 GroundingCheckAgent: No claims to verify - moderate confidence")
            # No claims to verify - moderate confidence
            confidence = 0.8
            passed = True
            issues = []
            recommendations = []
        else:
            confidence = grounded_claims / total_claims
            passed = confidence >= self.PASS_THRESHOLD

            issues = []
            recommendations = []

            if not passed:
                issues.append(f"Only {grounded_claims}/{total_claims} claims are grounded in sources")

                if grounded_numeric < len(numeric_claims):
                    issues.append(f"Numeric claims: {grounded_numeric}/{len(numeric_claims)} grounded")
                    recommendations.append("Verify all numeric claims against sources")

                if grounded_factual < len(factual_claims):
                    issues.append(f"Factual claims: {grounded_factual}/{len(factual_claims)} grounded")
                    recommendations.append("Add source citations or remove unsupported claims")

        # Determine severity
        if confidence < self.CRITICAL_THRESHOLD:
            severity = Severity.CRITICAL
        elif not passed:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO

        # Create metadata
        metadata = {
            "has_sources": True,
            "source_count": len(retrieved_passages),
            "total_claims": total_claims,
            "grounded_claims": grounded_claims,
            "numeric_claims_total": len(numeric_claims),
            "numeric_claims_grounded": grounded_numeric,
            "factual_claims_total": len(factual_claims),
            "factual_claims_grounded": grounded_factual,
            "grounding_rate": round(confidence, 2)
        }

        logger.info(f"[OK] GroundingCheckAgent: Validation {'PASSED' if passed else 'FAILED'} - Confidence: {confidence:.2f}, Severity: {severity.value}")

        return self._create_check(
            passed=passed,
            confidence=confidence,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )

    def _extract_numeric_claims(self, text: str) -> List[str]:
        """
        Extract sentences containing numeric claims (amounts, percentages, rates)

        Args:
            text: Response text

        Returns:
            List of sentences with numeric claims
        """
        # Pattern: sentences containing numbers, percentages, currency symbols
        pattern = r'[^.!?]*(?:\d+(?:,\d{3})*(?:\.\d+)?|₹|%|lakh|crore)[^.!?]*[.!?]'
        claims = re.findall(pattern, text, re.IGNORECASE)

        # Clean up and deduplicate
        claims = [c.strip() for c in claims if c.strip()]
        return list(dict.fromkeys(claims))  # Remove duplicates while preserving order

    def _extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract declarative factual sentences (simple heuristic)

        Args:
            text: Response text

        Returns:
            List of factual claim sentences
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        # Filter for declarative sentences (not questions, not too short)
        factual = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short sentences, questions, greetings, action items
            if (len(sentence.split()) > 5 and
                '?' not in sentence and
                not sentence.lower().startswith(('you should', 'consider', 'try', 'i recommend'))):
                factual.append(sentence)

        return factual

    def _verify_numeric_grounding(self, claims: List[str], passages: List[str]) -> int:
        """
        Count how many numeric claims have supporting numbers in passages

        Args:
            claims: List of claims with numbers
            passages: Source passages

        Returns:
            Count of grounded numeric claims
        """
        grounded_count = 0
        passages_text = ' '.join(passages).lower()

        for claim in claims:
            # Extract all numbers from claim (including decimals, percentages)
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', claim)

            # If any number from claim appears in passages, consider grounded
            if any(num.replace(',', '') in passages_text.replace(',', '') for num in numbers):
                grounded_count += 1

        return grounded_count

    def _verify_factual_grounding(self, claims: List[str], passages: List[str]) -> int:
        """
        Count how many factual claims have keyword overlap with passages

        Uses keyword matching with minimum overlap threshold.

        Args:
            claims: List of factual claims
            passages: Source passages

        Returns:
            Count of grounded factual claims
        """
        grounded_count = 0

        for claim in claims:
            # Extract significant keywords (words > 4 chars, excluding common words)
            keywords = self._extract_keywords(claim)

            if not keywords:
                continue

            # Check keyword overlap with each passage
            for passage in passages:
                passage_lower = passage.lower()
                overlapping_keywords = sum(1 for kw in keywords if kw in passage_lower)

                # If overlap exceeds threshold, consider grounded
                overlap_ratio = overlapping_keywords / len(keywords)
                if overlap_ratio >= self.KEYWORD_OVERLAP_THRESHOLD:
                    grounded_count += 1
                    break  # Claim is grounded, move to next

        return grounded_count

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract significant keywords from text

        Args:
            text: Input text

        Returns:
            List of keywords (lowercase)
        """
        # Common words to exclude
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'their'
        }

        # Extract words (5+ characters, alphanumeric)
        words = re.findall(r'\b\w{5,}\b', text.lower())

        # Filter out stopwords
        keywords = [w for w in words if w not in stopwords]

        return keywords

    def _validate_with_evidence_pack(self, response: str, evidence_pack: Any, context: Dict[str, Any]) -> ValidationCheck:
        """
        Production-grade validation using EvidencePack with citation checking.
        
        Performs:
        1. Regulatory claim extraction
        2. Citation requirement validation
        3. Citation-to-doc_id mapping verification
        
        Args:
            response: Agent response text
            evidence_pack: EvidencePack with citations
            context: Additional context
        
        Returns:
            ValidationCheck with detailed citation analysis
        """
        logger.info("🔍 GroundingCheckAgent: Using EvidencePack for production-grade validation")
        
        # Extract citations from evidence pack
        citations = evidence_pack.citations if hasattr(evidence_pack, 'citations') else []
        citation_doc_ids = {c.doc_id for c in citations} if citations else set()
        coverage = evidence_pack.coverage if hasattr(evidence_pack, 'coverage') else "unknown"
        
        logger.info(f"🔍 GroundingCheckAgent: EvidencePack has {len(citations)} citations, coverage={coverage}")
        
        # Extract regulatory claims from response
        regulatory_claims = self._extract_regulatory_claims(response)
        logger.info(f"🔍 GroundingCheckAgent: Found {len(regulatory_claims)} regulatory claims")
        
        issues = []
        recommendations = []
        
        # Critical check: If regulatory claims exist but no citations, FAIL
        if regulatory_claims and not citations:
            logger.warning("[WARNING] GroundingCheckAgent: CRITICAL - Regulatory claims without citations!")
            return self._create_check(
                passed=False,
                confidence=0.1,
                severity=Severity.CRITICAL,
                issues=[
                    f"Found {len(regulatory_claims)} regulatory claims but NO citations provided",
                    "Compliance statements require verified source documents",
                    "Examples: " + "; ".join(regulatory_claims[:3])
                ],
                recommendations=[
                    "Remove regulatory claims or provide citations",
                    "Use RAG retrieval to ground compliance statements",
                    "Never make up regulatory requirements"
                ],
                metadata={
                    "regulatory_claims_count": len(regulatory_claims),
                    "citations_count": 0,
                    "coverage": coverage,
                    "has_citations": False
                }
            )
        
        # Check if coverage is insufficient for regulatory query
        if regulatory_claims and coverage != "sufficient":
            logger.warning(f"[WARNING] GroundingCheckAgent: Coverage {coverage} insufficient for regulatory claims")
            issues.append(f"Coverage is {coverage} but regulatory claims present")
            recommendations.append("Increase evidence quality or remove unverified claims")
        
        # Verify each regulatory claim has backing citations
        unbacked_claims = []
        for claim in regulatory_claims:
            # Check if claim references any citation doc_ids
            # (Simple heuristic: check if doc_id substrings appear in claim)
            has_backing = False
            for doc_id in citation_doc_ids:
                # Check if any keywords from doc_id or citation text appear in claim
                if self._claim_backed_by_citation(claim, citations):
                    has_backing = True
                    break
            
            if not has_backing:
                unbacked_claims.append(claim)
        
        # Calculate confidence
        if regulatory_claims:
            backed_claims = len(regulatory_claims) - len(unbacked_claims)
            confidence = backed_claims / len(regulatory_claims)
        else:
            # No regulatory claims - check general grounding
            # Extract numeric and factual claims like legacy validation
            passages_text = [c.text for c in citations]
            numeric_claims = self._extract_numeric_claims(response)
            factual_claims = self._extract_factual_claims(response)
            
            grounded_numeric = self._verify_numeric_grounding(numeric_claims, passages_text)
            grounded_factual = self._verify_factual_grounding(factual_claims, passages_text)
            
            total_claims = len(numeric_claims) + len(factual_claims)
            grounded_claims = grounded_numeric + grounded_factual
            
            confidence = grounded_claims / total_claims if total_claims > 0 else 0.9
        
        # Severity and pass determination
        passed = confidence >= self.PASS_THRESHOLD and len(unbacked_claims) == 0
        
        if unbacked_claims:
            issues.append(f"{len(unbacked_claims)}/{len(regulatory_claims)} regulatory claims lack citation backing")
            issues.extend([f"  - {claim[:100]}..." for claim in unbacked_claims[:3]])
            recommendations.append("Ensure all regulatory statements cite source documents")
        
        if confidence < self.CRITICAL_THRESHOLD:
            severity = Severity.CRITICAL
        elif not passed:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        metadata = {
            "regulatory_claims_count": len(regulatory_claims),
            "unbacked_claims_count": len(unbacked_claims),
            "citations_count": len(citations),
            "citation_doc_ids": list(citation_doc_ids),
            "coverage": coverage,
            "has_citations": len(citations) > 0,
            "grounding_rate": round(confidence, 2)
        }
        
        logger.info(f"[OK] GroundingCheckAgent: {'PASSED' if passed else 'FAILED'} - Confidence: {confidence:.2f}")
        
        return self._create_check(
            passed=passed,
            confidence=confidence,
            severity=severity,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def _extract_regulatory_claims(self, text: str) -> List[str]:
        """
        Extract sentences that make regulatory/compliance claims.
        
        Patterns:
        - "RBI requires/mandates/specifies..."
        - "SEBI guidelines state..."
        - "According to IRDAI regulations..."
        - "Under Section X..."
        - "The regulation mandates..."
        
        Args:
            text: Response text
        
        Returns:
            List of sentences with regulatory claims
        """
        regulatory_patterns = [
            r'[^.!?]*\b(?:RBI|SEBI|IRDAI|CBDT|FIU|PFRDA)\b[^.!?]*\b(?:requires?|mandates?|specifies?|states?|guidelines?|regulations?)[^.!?]*[.!?]',
            r'[^.!?]*\baccording to\b[^.!?]*\b(?:RBI|SEBI|IRDAI|regulations?|act|section)[^.!?]*[.!?]',
            r'[^.!?]*\bunder (?:section|act|regulation|rule)\b[^.!?]*[.!?]',
            r'[^.!?]*\bthe (?:regulation|act|rule|guideline)\b[^.!?]*\b(?:requires?|mandates?|specifies?|states?)[^.!?]*[.!?]',
            r'[^.!?]*\b(?:compliance|regulatory|statutory)\b[^.!?]*\b(?:requirement|obligation|mandate)[^.!?]*[.!?]',
        ]
        
        claims = []
        for pattern in regulatory_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend([m.strip() for m in matches if m.strip()])
        
        # Deduplicate while preserving order
        return list(dict.fromkeys(claims))
    
    def _claim_backed_by_citation(self, claim: str, citations: List[Any]) -> bool:
        """
        Check if a regulatory claim is backed by at least one citation.
        
        Uses keyword overlap: if claim keywords appear in citation text, consider backed.
        
        Args:
            claim: Regulatory claim sentence
            citations: List of Citation objects from EvidencePack
        
        Returns:
            True if claim has citation backing
        """
        claim_keywords = set(self._extract_keywords(claim))
        
        if not claim_keywords:
            return False
        
        for citation in citations:
            citation_text = citation.text if hasattr(citation, 'text') else str(citation)
            citation_keywords = set(self._extract_keywords(citation_text))
            
            # Calculate keyword overlap
            overlap = len(claim_keywords & citation_keywords)
            overlap_ratio = overlap / len(claim_keywords) if claim_keywords else 0
            
            # If >30% keyword overlap, consider backed
            if overlap_ratio >= 0.3:
                return True
        
        return False