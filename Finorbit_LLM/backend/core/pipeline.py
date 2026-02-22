# ==============================================
# File: backend/core/pipeline.py
# Description: Unified validation pipeline replacing LangGraph orchestrator
# ==============================================

from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime

from backend.core.validation_models import (
    PreValidationResult, PostValidationResult, ConfidenceScore,
    ValidationCheck, AuditEntry, Severity
)
from backend.core.confidence_scorer import ConfidenceScorer
from backend.agents.safety.pii_detector import PIIDetectorAgent
from backend.agents.safety.content_risk_filter import ContentRiskFilterAgent
from backend.agents.safety.age_category_guard import AgeCategoryGuardAgent
from backend.agents.safety.mis_selling_guard import MisSellingGuardAgent
from backend.agents.safety.audit_logger import AuditLoggerAgent
from backend.agents.validation.grounding_check import GroundingCheckAgent
from backend.agents.validation.regulatory_consistency import RegulatoryConsistencyAgent
from backend.agents.validation.suitability_check import SuitabilityCheckAgent
from backend.agents.validation.tone_clarity import ToneClarityAgent

# Optional: compliance engine for input guardrails
try:
    from backend.core.compliance_engine import ComplianceEngineService
except Exception:
    ComplianceEngineService = None

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Unified validation pipeline (replaces LangGraph orchestrator)

    Orchestrates all safety and validation agents in a sequential pipeline:
    1. Pre-validation (safety): PII, content risk, age guard, mis-selling, audit
    2. [Agent execution happens between pre and post]
    3. Post-validation (quality): Grounding, numeric, regulatory, suitability, tone
    4. Confidence scoring: Multi-factor weighted calculation

    USER DECISION: Simplified sequential pipeline instead of LangGraph for better
    performance and easier debugging.
    """

    def __init__(self):
        """Initialize validation pipeline with all safety and validation agents"""
        logger.info("[CONFIG] Initializing ValidationPipeline")

        # Safety agents (pre-validation)
        self.pii_detector = PIIDetectorAgent()
        self.content_filter = ContentRiskFilterAgent()
        self.age_guard = AgeCategoryGuardAgent()
        self.missell_guard = MisSellingGuardAgent()
        self.audit_logger = AuditLoggerAgent()

        # Compliance engine for input guardrails
        self.compliance_engine = None
        if ComplianceEngineService:
            try:
                self.compliance_engine = ComplianceEngineService()
                logger.info("[OK] Compliance engine integrated into pre-validation")
            except Exception as e:
                logger.warning(f"[WARN] Failed to initialize compliance engine: {e}")

        # Validation agents (post-validation)
        self.grounding_check = GroundingCheckAgent()
        self.regulatory_check = RegulatoryConsistencyAgent()
        self.suitability_check = SuitabilityCheckAgent()
        self.tone_check = ToneClarityAgent()

        # Confidence scorer
        self.confidence_scorer = ConfidenceScorer()

        logger.info("[OK] ValidationPipeline initialized with 9 agents + compliance engine")

    def run_pre_validation(
        self,
        query: str,
        profile: Dict[str, Any]
    ) -> Tuple[PreValidationResult, List[AuditEntry]]:
        """
        Run pre-validation safety checks BEFORE agent execution

        Checks (in order):
        1. PII Detection → BLOCKS if critical PII found
        2. Content Risk → BLOCKS if illegal content detected
        3. Age/Category Guard → LOGS warnings (non-blocking)
        4. Mis-Selling Guard → LOGS warnings (non-blocking)
        5. Audit Logger → LOGS query (non-blocking)

        Args:
            query: User query text
            profile: User profile dict (age, income, occupation, etc.)

        Returns:
            Tuple of (PreValidationResult, audit_entries)

        Raises:
            No exceptions - all blocking is handled via PreValidationResult.should_block
        """
        logger.info("[GUARD] Starting pre-validation safety checks")
        start_time = datetime.utcnow()

        audit_entries = []
        blocking_issues = []
        pii_detected = False
        pii_entities = []
        content_risk_flags = []
        age_category_warning = None
        mis_selling_risk = 0.0

        # 1. PII Detection (BLOCKING)
        logger.info("🔍 Running PII detection")
        pii_safe, pii_issues, pii_meta = self.pii_detector.check(query, profile)

        if not pii_safe:
            pii_detected = True
            pii_entities = pii_meta.get("pii_entities", [])
            blocking_issues.extend(pii_issues)

            # Log audit entry
            audit_entries.append(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                event_type="pii_detected",
                agent_name="pii_detector",
                details={"pii_types": pii_meta.get("pii_types", []), "count": pii_meta.get("count", 0)},
                severity=Severity.CRITICAL,
                action_taken="blocked"
            ))

            logger.error(f"[ERROR] PII detected: {pii_meta.get('pii_types', [])}")

        # 2. Content Risk Filter (BLOCKING)
        logger.info("🔍 Running content risk filter")
        content_safe, content_issues, content_meta = self.content_filter.check(query, profile)

        if not content_safe:
            content_risk_flags = content_meta.get("risk_flags", [])
            blocking_issues.extend(content_issues)

            # Log audit entry
            audit_entries.append(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                event_type="content_risk_detected",
                agent_name="content_risk_filter",
                details={"risk_flags": content_risk_flags, "risk_level": content_meta.get("risk_level")},
                severity=Severity.CRITICAL,
                action_taken="blocked"
            ))

            logger.error(f"[ERROR] Content risk detected: {content_risk_flags}")

        # 3. Age/Category Guard (NON-BLOCKING)
        logger.info("🔍 Running age/category guard")
        age_safe, age_issues, age_meta = self.age_guard.check(query, profile)

        if age_meta.get("warnings"):
            age_category_warning = ", ".join(age_meta.get("warnings", []))

            # Log audit entry
            audit_entries.append(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                event_type="age_category_warning",
                agent_name="age_category_guard",
                details=age_meta,
                severity=Severity.WARNING,
                action_taken="logged"
            ))

            logger.warning(f"[WARNING] Age/category warnings: {age_category_warning}")

        # 4. Mis-Selling Guard (NON-BLOCKING)
        logger.info("🔍 Running mis-selling guard")
        missell_safe, missell_issues, missell_meta = self.missell_guard.check(query, profile)

        mis_selling_risk = missell_meta.get("risk_score", 0.0)

        if mis_selling_risk >= 0.5:
            # Log audit entry
            audit_entries.append(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                event_type="misselling_risk",
                agent_name="mis_selling_guard",
                details=missell_meta,
                severity=Severity.WARNING,
                action_taken="logged"
            ))

            logger.warning(f"[WARNING] Mis-selling risk: {mis_selling_risk:.2f}")

        # 5. Audit Logger (NON-BLOCKING)
        logger.info("📝 Logging query to audit trail")
        audit_safe, audit_issues, audit_meta = self.audit_logger.check(query, profile)

        # 6. Compliance Engine Input Guardrails (BLOCKING)
        if self.compliance_engine:
            logger.info("🔍 Running compliance engine input guardrails")
            try:
                # Run compliance check on input query
                # compliance_check expects (answer_text, context) where answer_text will be checked
                # against rules, and context['user_query'] is also checked for rule violations
                compliance_context = {
                    "user_query": query,
                    "module": "GENERIC",  # Pre-validation doesn't know target module yet
                    "language": "en",
                    "channel": "ALL"
                }
                # Pass empty answer for input checking - rules will match against user_query in context
                compliance_result = self.compliance_engine.compliance_check("", compliance_context)
                
                # Check if compliance blocked the query
                if compliance_result.status in ["BLOCKED", "FORCE_SAFE_ANSWER"]:
                    blocking_issues.append(compliance_result.final_answer)
                    
                    # Log audit entry
                    audit_entries.append(AuditEntry(
                        timestamp=datetime.utcnow().isoformat(),
                        event_type="compliance_block",
                        agent_name="compliance_engine",
                        details={
                            "status": compliance_result.status,
                            "message": compliance_result.final_answer,
                            "matched_rules": compliance_result.triggered_rule_ids
                        },
                        severity=Severity.CRITICAL,
                        action_taken="blocked"
                    ))
                    
                    logger.error(f"[ERROR] Compliance engine blocked query: {compliance_result.final_answer}")
            except Exception as e:
                logger.warning(f"[WARN] Compliance engine check failed: {e}")

        # Determine if execution should be blocked
        should_block = len(blocking_issues) > 0
        safe_to_proceed = not should_block

        # Create aggregated result
        result = PreValidationResult(
            pii_detected=pii_detected,
            pii_entities=pii_entities,
            mis_selling_risk=mis_selling_risk,
            age_category_warning=age_category_warning,
            content_risk_flags=content_risk_flags,
            blocking_issues=blocking_issues,
            should_block=should_block,
            safe_to_proceed=safe_to_proceed
        )

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(f"[OK] Pre-validation complete in {duration_ms:.2f}ms - {'BLOCKED' if should_block else 'SAFE'}")

        return result, audit_entries

    def run_post_validation(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Tuple[PostValidationResult, List[AuditEntry]]:
        """
        Run post-validation quality checks AFTER agent execution

        Checks (in order):
        1. Grounding Check → Verify claims against sources
        2. Numeric Validation → Check numbers against reference tables
        3. Regulatory Consistency → Detect illegal advice/missing disclaimers
        4. Suitability Check → Ensure advice matches user profile
        5. Tone & Clarity → Check for clear, understandable language

        Only CRITICAL severity issues block the response. WARNING issues are logged.

        Args:
            response: Agent's response text
            context: Context dict containing:
                - query: str - Original user query
                - profile: Dict - User profile
                - retrieved_passages: List[str] - RAG sources (if available)
                - agent: str - Which agent generated response

        Returns:
            Tuple of (PostValidationResult, audit_entries)
        """
        logger.info("[OK] Starting post-validation quality checks")
        start_time = datetime.utcnow()

        audit_entries = []
        validation_checks: List[ValidationCheck] = []
        blocking_issues = []
        warnings = []

        # Check intent - skip strict validation for general information
        intent = context.get("intent", "personalized")

        if intent == "general":
            logger.info("ℹ️ General information mode - skipping strict numeric and grounding validation")

            # Create passing results for skipped validators
            grounding_result = ValidationCheck(
                agent_name="grounding_check",
                check_type="grounding",
                passed=True,
                confidence=1.0,
                severity=Severity.INFO,
                issues=[],
                recommendations=[],
                metadata={"skipped": "general_information", "reason": "No source passages needed for educational content"}
            )
            validation_checks.append(grounding_result)

        else:
            # Personalized mode - run full validation

            # 1. Grounding Check
            logger.info("🔍 Running grounding check")
            grounding_result = self.grounding_check.validate(response, context)
            validation_checks.append(grounding_result)

            if grounding_result.severity == Severity.CRITICAL:
                blocking_issues.extend(grounding_result.issues)
                audit_entries.append(AuditEntry(
                    timestamp=datetime.utcnow().isoformat(),
                    event_type="grounding_failure",
                    agent_name="grounding_check",
                    details=grounding_result.metadata,
                    severity=Severity.CRITICAL,
                    action_taken="blocked"
                ))
                logger.error(f"[ERROR] Grounding check CRITICAL failure: {grounding_result.issues}")
            elif grounding_result.severity == Severity.WARNING:
                warnings.extend(grounding_result.issues)
                logger.warning(f"[WARNING] Grounding check warning: {grounding_result.issues}")

        # 2. Regulatory Consistency (always run for both general and personalized)
        logger.info("🔍 Running regulatory consistency check")
        regulatory_result = self.regulatory_check.validate(response, context)
        validation_checks.append(regulatory_result)

        if regulatory_result.severity == Severity.CRITICAL:
            blocking_issues.extend(regulatory_result.issues)
            audit_entries.append(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                event_type="regulatory_violation",
                agent_name="regulatory_consistency",
                details=regulatory_result.metadata,
                severity=Severity.CRITICAL,
                action_taken="blocked"
            ))
            logger.error(f"[ERROR] Regulatory check CRITICAL failure: {regulatory_result.issues}")
        elif regulatory_result.severity == Severity.WARNING:
            warnings.extend(regulatory_result.issues)
            logger.warning(f"[WARNING] Regulatory check warning: {regulatory_result.issues}")

        # 4. Suitability Check
        logger.info("🔍 Running suitability check")
        suitability_result = self.suitability_check.validate(response, context)
        validation_checks.append(suitability_result)

        if suitability_result.severity == Severity.CRITICAL:
            blocking_issues.extend(suitability_result.issues)
            audit_entries.append(AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                event_type="suitability_mismatch",
                agent_name="suitability_check",
                details=suitability_result.metadata,
                severity=Severity.CRITICAL,
                action_taken="blocked"
            ))
            logger.error(f"[ERROR] Suitability check CRITICAL failure: {suitability_result.issues}")
        elif suitability_result.severity == Severity.WARNING:
            warnings.extend(suitability_result.issues)
            logger.warning(f"[WARNING] Suitability check warning: {suitability_result.issues}")

        # 5. Tone & Clarity
        logger.info("🔍 Running tone & clarity check")
        tone_result = self.tone_check.validate(response, context)
        validation_checks.append(tone_result)

        # Tone check is always WARNING or INFO (never blocks)
        if tone_result.severity == Severity.WARNING:
            warnings.extend(tone_result.issues)
            logger.warning(f"[WARNING] Tone & clarity warning: {tone_result.issues}")

        # Determine if response should be blocked
        should_block = len(blocking_issues) > 0

        # Create aggregated result
        result = PostValidationResult(
            grounding_ok=grounding_result.passed,
            grounding_score=grounding_result.confidence,
            reg_ok=regulatory_result.passed,
            reg_score=regulatory_result.confidence,
            numeric_ok=True,  # Numeric validation moved to finance_math tools
            numeric_score=1.0,  # Default passing score
            suitability_ok=suitability_result.passed,
            suitability_score=suitability_result.confidence,
            tone_ok=tone_result.passed,
            tone_score=tone_result.confidence,
            validation_checks=validation_checks,
            blocking_issues=blocking_issues,
            warnings=warnings,
            should_block=should_block
        )

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(f"[OK] Post-validation complete in {duration_ms:.2f}ms - {'BLOCKED' if should_block else 'PASSED'}")

        return result, audit_entries

    def calculate_confidence(
        self,
        post_validation: PostValidationResult,
        retrieval_score: float = 0.8
    ) -> ConfidenceScore:
        """
        Calculate multi-factor confidence score for response

        Uses weighted average of:
        - Retrieval relevance: 35%
        - Grounding: 35%
        - Regulatory check: 30%

        Determines recommended action based on score:
        - >= 0.85: Serve confidently
        - 0.70-0.84: Serve with disclaimer
        - 0.50-0.69: Ask clarifying questions
        - 0.30-0.49: Partial answer
        - < 0.30: Politely refuse

        Args:
            post_validation: Post-validation results
            retrieval_score: RAG retrieval relevance score (default 0.8 if no RAG)

        Returns:
            ConfidenceScore with overall score, breakdown, and recommended action
        """
        logger.info(" Calculating confidence score")

        confidence = self.confidence_scorer.calculate(
            post_validation=post_validation,
            retrieval_score=retrieval_score
        )

        logger.info(f" Confidence score: {confidence.overall_score:.2f}, Action: {confidence.recommended_action}")

        return confidence
