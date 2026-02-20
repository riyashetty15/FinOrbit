# ==============================================
# File: backend/server.py
# Description: Unified backend FastAPI server (Pipeline Integrated)
# ==============================================

import os
import re
import logging
import asyncio
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Dict, Any, Optional, List

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

# Define guardrail exceptions locally to avoid import conflicts
class InputGuardrailTripwireTriggered(Exception):
    """Raised when input guardrails are triggered"""
    pass

class OutputGuardrailTripwireTriggered(Exception):
    """Raised when output guardrails are triggered"""
    pass

from backend.finance_agent import AssistantMessage, finorbit_instructions
from backend.models import QueryRequest, QueryResponse, ErrorResponse
from backend.core.pipeline import ValidationPipeline
from backend.core.validation_models import PostValidationResult, Severity
from backend.core.router import RouterAgent
from backend.core.conversation_context import ConversationContext, ProfileExtractor
from backend.core.multi_agent_orchestrator import MultiAgentOrchestrator
from backend.logging_config import setup_logging, TraceContext, log_event
from backend.config import settings

# Specialist agents
from backend.agents.specialist.tax_planner import TaxPlanner
from backend.agents.specialist.investment_coach import InvestmentCoach
from backend.agents.specialist.credits_loans import CreditsLoanAgent
from backend.agents.specialist.insurance_analyzer import InsuranceAnalyzer
from backend.agents.specialist.retirement_planner import RetirementPlanner
# from backend.agents.rag.rag_agent import RAGAgent

# MCP Tools
from backend.tools.rag_tool import knowledge_lookup
from backend.services.retrieval_service import RetrievalService

# Optional: compliance + final output formatting (fail-open if deps missing)
try:
    from backend.core.compliance_engine import ComplianceEngineService
except Exception:  # pragma: no cover
    ComplianceEngineService = None  # type: ignore

try:
    from backend.fin_fode.engine.final_output_engine import FinalOutputEngine
except Exception:  # pragma: no cover
    FinalOutputEngine = None  # type: ignore

load_dotenv()

# Setup structured logging
setup_logging(log_file="logs/backend.log", level="INFO", json_logs=True)
logger = logging.getLogger(__name__)

logger.info("Unified backend server starting up...")


def _get_openai_client() -> OpenAI:
    """Return a configured OpenAI client."""
    api_key = settings.llm_api_key
    if not api_key:
        logger.warning("LLM_API_KEY is not set; main agent will be disabled.")
        return None
    return OpenAI(api_key=api_key)


async def _run_gemini_finance_agent(user_input: str) -> str:
    """Main agent implemented on OpenAI chat completions."""
    client = _get_openai_client()
    if not client:
        return _hallucination_fallback_message()

    try:
        model_name = settings.custom_model_name or "gpt-4o-mini"
        prompt = f"{finorbit_instructions.strip()}\n\nUser question:\n{user_input}"

        def _call():
            return client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": finorbit_instructions.strip()},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

        response = await asyncio.to_thread(_call)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            logger.warning("Main agent returned empty response; using handover message.")
            return _hallucination_fallback_message()
        logger.info("OpenAI main agent responded successfully.")
        return text
    except Exception as e:  # pragma: no cover
        logger.error(f"Main agent failed: {e}")
        return _hallucination_fallback_message()


def _hallucination_fallback_message() -> str:
    # Keep this message aligned with advisor handover messaging to avoid confusing UX.
    return _low_confidence_handover_message()


def _advisor_handover_message() -> str:
    return _low_confidence_handover_message()


def _low_confidence_handover_message() -> str:
    return (
        "I can’t provide a confident, specific answer yet. Here are the top details that unblock me across most questions: "
        "1) goal/amount and timeframe, 2) monthly income and fixed expenses, 3) existing loans/EMIs and dependents. "
        "Share these and I will tailor the guidance. If you prefer, I can also give a high-level overview first. "
        "Note: this is general information, not personalized financial advice."
    )


def _should_advisor_handover(*, overall_score: float, recommended_action: str) -> bool:
    enabled = os.getenv("ADVISOR_HANDOVER_ENABLED", "true").lower() in ("1", "true", "yes")
    if not enabled:
        return False

    try:
        threshold = float(os.getenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "0.50"))
    except Exception:
        threshold = 0.50

    # Force handover for very low confidence bands.
    if (recommended_action or "").lower() in ("refuse", "partial"):
        return True

    return float(overall_score) < threshold


def _should_hallucination_fallback(post_val_result: PostValidationResult) -> bool:
    """Return True if the response was blocked for likely-hallucination reasons (grounding/numeric only).

    We do NOT auto-fallback for regulatory/suitability CRITICAL issues; those remain hard blocks.
    """
    critical_types = {
        (vc.check_type or "")
        for vc in (post_val_result.validation_checks or [])
        if vc.severity == Severity.CRITICAL
    }
    if not critical_types:
        return False
    return critical_types.issubset({"grounding", "numeric"})


def _generate_follow_ups(agent_type: str, profile: dict, needs_clarification: bool) -> List[str]:
    """Generate contextual follow-up suggestions based on agent and profile."""
    suggestions = []
    if needs_clarification:
        if not profile.get("age"):
            suggestions.append("Add my age")
        if not profile.get("income"):
            suggestions.append("Add my income details")
        if not profile.get("dependents"):
            suggestions.append("Add dependents info")
    # Domain-specific follow-ups
    follow_up_map = {
        "tax_planner": ["Compare old vs new regime", "Show tax-saving options under 80C", "Calculate HRA exemption"],
        "investment_coach": ["Show SIP calculator", "Compare mutual fund categories", "Portfolio allocation suggestion"],
        "credits_loans": ["Calculate EMI for this loan", "Check loan eligibility", "Compare interest rates"],
        "insurance_analyzer": ["Compare health plans", "Show rider options", "Calculate premium estimate"],
        "retirement_planner": ["Estimate retirement corpus", "Show NPS benefits", "Plan withdrawal strategy"],
    }
    suggestions.extend(follow_up_map.get(agent_type, ["Tell me more", "Explain in detail"])[:3])
    return suggestions[:5]


def _infer_compliance_module(
    *,
    user_input: str,
    agent_type: str,
    agents_used: List[str],
    execution_order: List[str],
) -> str:
    module_map = {
        "credits_loans": "CREDIT",
        "investment_coach": "SIP_INVESTMENT",
        "insurance_analyzer": "INSURANCE",
        "retirement_planner": "RETIREMENT",
        "tax_planner": "TAX",
        "fraud_shield": "FRAUD",
    }

    # 1) Keyword-first inference (most reliable for multi-agent, because execution_order may start
    # with unrelated specialists like insurance_analyzer).
    ql = (user_input or "").lower()
    if re.search(r"\b(otp|cvv|pin|fraud|scam|phish)\b", ql):
        return "FRAUD"
    if re.search(r"\b(tax|itr|80c|deduction|tds|gst)\b", ql):
        return "TAX"
    if re.search(r"\b(insurance|premium|claim|policy)\b", ql):
        return "INSURANCE"
    if re.search(r"\b(retirement|nps|pension)\b", ql):
        return "RETIREMENT"
    if re.search(r"\b(loan|emi|credit|card|lender)\b", ql):
        return "CREDIT"
    # Fixed deposits are fixed-income products; avoid applying market-linked "guaranteed returns" rules.
    if re.search(r"\b(fixed\s+deposit|fixed\s+deposits|term\s+deposit|fd|fds)\b", ql):
        return "GENERIC"
    if re.search(r"\b(mutual fund|mutual funds|sip|nav|aum|equity|stock|shares?|market|invest(ing)?|investment|payout|returns?)\b", ql):
        return "SIP_INVESTMENT"

    # 2) Agent-based inference
    module = module_map.get(agent_type)
    if not module:
        for name in (execution_order or agents_used or []):
            mapped = module_map.get(name)
            if mapped:
                module = mapped
                break

    return module or "GENERIC"


# ---------------------------
# Initialize Pipeline & Agents
# ---------------------------
validation_pipeline = ValidationPipeline()
router_agent = RouterAgent()

fode_engine = FinalOutputEngine() if FinalOutputEngine else None
FODE_ENABLED = os.getenv("FODE_ENABLED", "true").lower() in ("1", "true", "yes")

# Initialize specialist agents
specialist_agents: Dict[str, Any] = {
    "tax_planner": TaxPlanner(),
    "investment_coach": InvestmentCoach(),
    "credits_loans": CreditsLoanAgent(),
    "insurance_analyzer": InsuranceAnalyzer(),
    "retirement_planner": RetirementPlanner(),
    # "rag_agent": RAGAgent(),
}

# Initialize multi-agent orchestrator for complex queries
orchestrator = MultiAgentOrchestrator(specialist_agents)

# Initialize Services
retrieval_service = RetrievalService()

logger.info(f"Initialized validation pipeline, {len(specialist_agents)} specialist agents, and multi-agent orchestrator")


# ---------------------------
# Compliance init (JSON-first; fail-open)
# ---------------------------
COMPLIANCE_ENGINE: Optional[Any] = None
COMPLIANCE_ENABLED = os.getenv("COMPLIANCE_ENABLED", "true").lower() in ("1", "true", "yes")

try:
    if COMPLIANCE_ENABLED and ComplianceEngineService:
        rules_path = os.getenv("COMPLIANCE_RULES_PATH", "backend/rules/compliance_rules.json")
        COMPLIANCE_ENGINE = ComplianceEngineService(rules_path=rules_path)
        logger.info("Compliance enabled (JSON-first): rules_path=%s", rules_path)
    elif not COMPLIANCE_ENABLED:
        logger.info("Compliance disabled by COMPLIANCE_ENABLED=false")
    else:
        logger.warning("Compliance disabled: compliance engine import failed.")
except Exception as e:
    COMPLIANCE_ENGINE = None
    logger.error("Compliance init failed (disabled): %s", e)


# ---------------------------
# Lifespan (no MCP servers)
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager (MCP removed)"""
    logger.info("[PACKAGE] Server startup complete")
    yield
    if COMPLIANCE_ENGINE and hasattr(COMPLIANCE_ENGINE, "close"):
        try:
            COMPLIANCE_ENGINE.close()
        except Exception:
            logger.exception("Compliance engine shutdown failed")
    logger.info("[PACKAGE] Server shutdown complete")


# ---------------------------
# Create FastAPI App
# ---------------------------
app = FastAPI(
    title="Finorbit Unified Backend",
    description="Unified financial assistant backend with integrated validation pipeline",
    version="2.0.0",
    lifespan=lifespan
)

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/ui/static", StaticFiles(directory=STATIC_DIR), name="ui-static")

# Database connection
DATABASE_URL = re.sub(r'^postgresql:', 'postgresql+asyncpg:', os.getenv('DATABASE_URL'))


# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/query", response_model=QueryResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def process_query(request: QueryRequest):
    """
    Process user financial query with context-aware routing and validation

    Flow:
    1. Load conversation context (last agent + profile)
    2. Extract profile information from query (income, age, occupation)
    3. Pre-validation (safety checks)
    4. Context-aware routing (maintains conversation continuity)
    5. Execute agent with merged profile
    6. Post-validation (quality checks)
    7. Calculate confidence score
    8. Update conversation context
    9. Return response or detailed error

    **NEW**: Context-aware routing maintains conversation continuity without
    exploding token usage by only tracking last agent + extracted profile.

    Raises:
        HTTPException: For blocked queries/responses with detailed reasons
    """
    with TraceContext() as trace_id:
        log_event(logger, "query_received", "server", {
            "user_id": request.userId,
            "conversation_id": request.conversationId,
            "query_length": len(request.query)
        })

        # Separate user question from profile context.
        # The UI may send profile context as a separate field OR prepended to the query.
        raw_query = request.query.strip()
        profile_hint = (request.profileHint or "").strip()

        # Legacy support: if profile is embedded in query, extract the real question
        if "\nUser question:" in raw_query:
            parts = raw_query.split("\nUser question:", 1)
            if not profile_hint:
                profile_hint = parts[0].strip()
            user_question = parts[1].strip()
        elif "\n\nUser question:" in raw_query:
            parts = raw_query.split("\n\nUser question:", 1)
            if not profile_hint:
                profile_hint = parts[0].strip()
            user_question = parts[1].strip()
        else:
            user_question = raw_query

        # user_input = clean question for routing; full_context = question + profile for agents
        user_input = user_question
        full_context = (profile_hint + "\n\n" + user_question).strip() if profile_hint else user_question

        if not user_input:
            raise HTTPException(status_code=400, detail="User input cannot be empty")

        try:
            retrieved_passages: List[Dict[str, Any]] = []
            low_confidence_triggers: List[str] = []

            # ---------------------------
            # STEP 1: Pre-Validation (Safety)
            # ---------------------------
            log_event(logger, "pre_validation_start", "pipeline")

            # ---------------------------
            # Load Conversation Context
            # ---------------------------
            conversation_context = ConversationContext.get_context(request.conversationId)

            log_event(logger, "conversation_context_loaded", "context", {
                "has_context": conversation_context is not None,
                "context": conversation_context if conversation_context else {}
            })

            # Extract profile information from both profile hint and user question
            extracted_profile = ProfileExtractor.extract_all(full_context)

            if extracted_profile:
                log_event(logger, "profile_extracted", "context", {
                    "extracted_fields": list(extracted_profile.keys()),
                    "values": extracted_profile
                })

            # Build user profile (merge existing context + extracted info)
            profile = {
                "user_id": request.userId,
                "age": None,
                "income": None,
                "occupation": None
            }

            # Merge profile from conversation context (if exists)
            if conversation_context and "profile" in conversation_context:
                profile.update(conversation_context["profile"])

            # Merge newly extracted profile (overrides previous)
            profile.update(extracted_profile)

            log_event(logger, "profile_built", "context", {
                "final_profile": profile
            })

            pre_val_result, pre_audits = validation_pipeline.run_pre_validation(user_input, profile)

            if pre_val_result.should_block:
                log_event(logger, "pre_validation_blocked", "pipeline", {
                    "blocking_issues": pre_val_result.blocking_issues
                }, level="ERROR")

                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Your query cannot be processed due to safety concerns",
                        "error_type": "pre_validation_failure",
                        "blocked_reason": "; ".join(pre_val_result.blocking_issues),
                        "details": {
                            "pii_detected": pre_val_result.pii_detected,
                            "pii_types": [e["type"] for e in pre_val_result.pii_entities] if pre_val_result.pii_entities else [],
                            "content_risk_flags": pre_val_result.content_risk_flags,
                            "age_category_warning": pre_val_result.age_category_warning
                        }
                    }
                )

            log_event(logger, "pre_validation_passed", "pipeline")

            # ---------------------------
            # STEP 2: Check for Complex Multi-Agent Queries (Skip if follow-up)
            # ---------------------------
            # Check conversation context first to maintain continuity
            is_follow_up_query = conversation_context is not None and ConversationContext.is_follow_up(user_input, conversation_context)
            
            multi_result = None
            if not is_follow_up_query:
                # Only check for multi-agent orchestration if NOT a follow-up
                multi_result = await orchestrator.process_complex_query(
                    query=full_context,
                    profile=profile,
                    session_id=request.conversationId
                )
            else:
                log_event(logger, "orchestrator_skipped", "orchestrator", {
                    "reason": "follow_up_query",
                    "last_agent": conversation_context.get("last_agent") if conversation_context else None
                })

            agents_used: List[str] = []
            execution_order: List[str] = []

            if multi_result:
                # Complex query requiring multiple agents
                log_event(logger, "complex_query_processed", "orchestrator", {
                    "agents_used": multi_result["agents_used"],
                    "execution_order": multi_result["execution_order"],
                    "sub_queries_count": len(multi_result["sub_responses"])
                })

                response_text = multi_result["summary"]

                # Collect citations/sources from sub-agents (primarily rag_agent)
                try:
                    for sr in (multi_result.get("sub_responses") or []):
                        for src in (sr.get("sources") or []):
                            if isinstance(src, dict):
                                retrieved_passages.append({
                                    "agent": sr.get("agent"),
                                    "document": src.get("document"),
                                    "chunk_index": src.get("chunk_index"),
                                    "similarity_score": src.get("similarity_score"),
                                    "excerpt": src.get("excerpt"),
                                })
                except Exception:
                    logger.exception("Failed to collect retrieved passages from multi-agent sub_responses")

                agents_used = list(multi_result.get("agents_used") or [])
                execution_order = list(multi_result.get("execution_order") or [])
                
                agents_used = multi_result["agents_used"] if multi_result["agents_used"] else ["fin_advisor"]
                agent_type = agents_used[0]  # Keep for backward compatibility with context
                intent = "general"  # Multi-agent queries are typically complex, treat as general
                
                # Skip to post-validation (bypass single-agent execution)
                log_event(logger, "multi_agent_execution_complete", "orchestrator", {
                    "response_length": len(response_text)
                })
                
            else:
                # Simple query - use single-agent routing
                # ---------------------------
                # STEP 3: Context-Aware Routing with Intent Classification
                # ---------------------------
                agent_type, intent = router_agent.route_with_context(
                    query=user_input,
                    conversation_context=conversation_context
                )

                agents_used = [agent_type]
                execution_order = [agent_type]

                log_event(logger, "routing_complete", "router", {
                    "agent_type": agent_type,
                    "intent": intent,
                    "is_follow_up": conversation_context is not None and ConversationContext.is_follow_up(user_input, conversation_context)
                })

                # ---------------------------
                # STEP 3.5: Production-Grade RAG - RouteIntent with Evidence Detection
                # ---------------------------
                route_intent = router_agent.route_with_evidence_intent(
                    query=user_input,
                    user_context=profile
                )

                log_event(logger, "route_intent_enriched", "router", {
                    "module": route_intent.module,
                    "needs_evidence": route_intent.needs_evidence,
                    "jurisdiction": route_intent.jurisdiction,
                    "time_sensitivity": route_intent.time_sensitivity,
                    "query_type": route_intent.query_type
                })

                # ---------------------------
                # STEP 3.6: Evidence Retrieval (if regulatory grounding required)
                # ---------------------------
                evidence_pack = None
                if route_intent.needs_evidence:
                    log_event(logger, "evidence_retrieval_start", "rag", {
                        "module": route_intent.module,
                        "time_sensitivity": route_intent.time_sensitivity
                    })
                    
                    # Build filters from route intent
                    filters = {
                        "jurisdiction": route_intent.jurisdiction,
                    }
                    
                    # Add time sensitivity filters to get latest documents
                    if route_intent.time_sensitivity == "high":
                        filters["is_current"] = True
                        filters["year_min"] = 2023  # Last 3 years for time-sensitive queries
                    
                    # Retrieve evidence with coverage scoring
                    try:
                        evidence_pack = await retrieval_service.retrieve_evidence(
                            query=user_input,
                            module=route_intent.module,
                            top_k=8,
                            filters=filters
                        )
                        
                        log_event(logger, "evidence_retrieval_complete", "rag", {
                            "coverage": evidence_pack.coverage,
                            "citations_count": len(evidence_pack.citations),
                            "confidence": evidence_pack.confidence,
                            "rejection_reason": evidence_pack.rejection_reason
                        })
                    except Exception as e:
                        logger.error(f"Evidence retrieval failed: {e}")
                        evidence_pack = None
                
                # ---------------------------
                # STEP 3.7: Evidence Gate Check - Refuse if insufficient evidence
                # ---------------------------
                evidence_refusal = None
                if route_intent.needs_evidence:
                    from backend.services.decision_engine import DecisionEngine
                    
                    evidence_refusal = DecisionEngine.check_evidence_gate(
                        needs_evidence=route_intent.needs_evidence,
                        evidence_pack=evidence_pack,
                        query=user_input,
                        module=route_intent.module
                    )
                    
                    if evidence_refusal:
                        log_event(logger, "evidence_gate_failed", "decision_engine", {
                            "coverage": evidence_pack.coverage if evidence_pack else "none",
                            "follow_ups_count": len(evidence_refusal.follow_ups),
                            "confidence": evidence_refusal.confidence
                        })
                        
                        # Build refusal response with follow-up questions
                        response_text = evidence_refusal.recommendations[0] if evidence_refusal.recommendations else "Insufficient evidence to answer query."
                        
                        if evidence_refusal.reasoning_factors:
                            response_text += "\n\n**Why I cannot answer:**\n"
                            for reason in evidence_refusal.reasoning_factors[:2]:
                                response_text += f"• {reason}\n"
                        
                        # Add follow-up questions
                        if evidence_refusal.follow_ups:
                            response_text += "\n**To help me provide accurate information, please clarify:**\n"
                            for i, follow_up in enumerate(evidence_refusal.follow_ups, 1):
                                response_text += f"{i}. {follow_up}\n"
                        
                        # Add disclaimer
                        if evidence_refusal.required_disclaimers:
                            response_text += "\n*" + evidence_refusal.required_disclaimers[0] + "*"
                        
                        # Skip agent execution, return evidence gate refusal
                        # We'll use the existing response flow but with refusal text
                        agents_used = [agent_type]
                        execution_order = [agent_type]
                        
                        # Skip to STEP 7 (post-validation) with refusal response
                        # Set flag to bypass agent execution
                        log_event(logger, "evidence_gate_refusal_sent", "decision_engine", {
                            "response_length": len(response_text)
                        })

                # ---------------------------
                # STEP 4: Execute Single Agent
                # ---------------------------
                if agent_type in specialist_agents and not evidence_refusal:
                    # Execute specialist agent (LangGraph)
                    log_event(logger, "executing_specialist_agent", agent_type, {
                        "intent": intent,
                        "has_evidence": evidence_pack is not None if 'evidence_pack' in locals() else False
                    })

                    initial_state = {
                        "query": full_context,
                        "profile": profile,
                        "intent": intent,
                        "evidence_pack": evidence_pack if 'evidence_pack' in locals() else None  # Pass evidence to agent
                    }

                    # Check if agent.run is async (e.g., RAG agent)
                    agent = specialist_agents[agent_type]
                    if hasattr(agent.run, '__call__'):
                        import inspect
                        if inspect.iscoroutinefunction(agent.run):
                            result = await agent.run(initial_state)
                        else:
                            result = agent.run(initial_state)
                    else:
                        result = agent.run(initial_state)
                        
                    response_text = result.get("summary", "")

                    # Capture citations/sources (e.g., from rag_agent)
                    try:
                        for src in (result.get("sources") or []):
                            if isinstance(src, dict):
                                retrieved_passages.append({
                                    "agent": agent_type,
                                    "document": src.get("document"),
                                    "chunk_index": src.get("chunk_index"),
                                    "similarity_score": src.get("similarity_score"),
                                    "excerpt": src.get("excerpt"),
                                })
                    except Exception:
                        logger.exception("Failed to collect retrieved passages from specialist agent result")

                    log_event(logger, "specialist_agent_complete", agent_type, {
                        "response_length": len(response_text)
                    })

                elif not evidence_refusal:
                    # Execute main finance agent only if no evidence gate refusal
                    log_event(logger, "executing_main_agent", "finance_agent")

                    response_text = await _run_gemini_finance_agent(full_context)

                    log_event(logger, "main_agent_complete", "finance_agent", {
                        "response_length": len(response_text)
                    })
                else:
                    # Evidence gate refusal - response_text already set, skip agent execution
                    log_event(logger, "agent_execution_skipped", "decision_engine", {
                        "reason": "evidence_gate_refusal",
                        "response_length": len(response_text)
                    })

            # ---------------------------
            # STEP 4: Compliance gate (BEFORE validation) - fail-open
            # ---------------------------
            # Rationale: validators may hard-block forbidden language (e.g., "guaranteed returns");
            # compliance should deterministically rewrite/force-safe *before* those validators run.
            inferred_module_for_request = _infer_compliance_module(
                user_input=user_input,
                agent_type=agent_type,
                agents_used=agents_used,
                execution_order=execution_order,
            )

            compliance_status = "SKIPPED"
            compliance_triggered_rule_ids: List[int] = []
            if COMPLIANCE_ENABLED and COMPLIANCE_ENGINE:
                try:
                    module = _infer_compliance_module(
                        user_input=user_input,
                        agent_type=agent_type,
                        agents_used=agents_used,
                        execution_order=execution_order,
                    )

                    intent_tags: List[str] = []
                    if intent:
                        intent_tags = [str(intent).upper()]

                    user_id_hash = sha256(str(request.userId).encode("utf-8")).hexdigest()

                    engine_context = {
                        "module": module,
                        "language": "en",
                        "channel": "YOUTH_APP",
                        "intent_tags": intent_tags,
                        "regulator_scope": ["RBI", "SEBI", "IRDAI", "PFRDA", "IT", "CERT_IN", "GENERIC"],
                        "query_id": str(trace_id),
                        "user_id_hash": user_id_hash,
                        "user_query": user_input,
                    }

                    comp = await COMPLIANCE_ENGINE.compliance_check_async(response_text, engine_context)
                    compliance_status = getattr(comp, "status", None) or "OK"
                    compliance_triggered_rule_ids = getattr(comp, "triggered_rule_ids", []) or []

                    log_event(
                        logger,
                        "compliance_checked",
                        "compliance",
                        {
                            "status": compliance_status,
                            "triggered_rule_ids": compliance_triggered_rule_ids,
                            "module": module,
                            "intent_tags": intent_tags,
                            "agent_type": agent_type,
                            "agents_used": agents_used,
                        },
                    )

                    # JSON-first compliance uses "BLOCKED" to mean "force safe answer".
                    response_text = getattr(comp, "final_answer", response_text)

                except HTTPException:
                    raise
                except Exception as ce:
                    log_event(logger, "compliance_error", "compliance", {"error": str(ce)}, level="ERROR")
                    # fail-open

            # ---------------------------
            # STEP 5: Post-Validation (Quality)
            # ---------------------------
            log_event(logger, "post_validation_start", "pipeline")

            context = {
                "query": user_input,
                "profile": profile,
                "retrieved_passages": retrieved_passages,
                "evidence_pack": evidence_pack if 'evidence_pack' in locals() else None,  # Production-grade evidence
                "agent": agent_type,
                "module": inferred_module_for_request,
                "compliance_status": compliance_status,
                "compliance_triggered_rule_ids": compliance_triggered_rule_ids,
                "intent": intent  # Pass intent for validation mode (general vs personalized)
            }

            post_val_result, post_audits = validation_pipeline.run_post_validation(response_text, context)

            if post_val_result.should_block:
                if _should_hallucination_fallback(post_val_result):
                    log_event(
                        logger,
                        "hallucination_fallback_triggered",
                        "pipeline",
                        {
                            "reason": "post_validation_blocked_grounding_or_numeric",
                            "blocking_issues": post_val_result.blocking_issues,
                        },
                        level="ERROR",
                    )

                    response_text = _hallucination_fallback_message()
                    low_confidence_triggers.append("post_validation_grounding_or_numeric")
                    # Downgrade the block so the request can complete with a safe refusal.
                    post_val_result.should_block = False
                    post_val_result.blocking_issues = []
                else:
                    log_event(logger, "post_validation_blocked", "pipeline", {
                        "blocking_issues": post_val_result.blocking_issues
                    }, level="ERROR")

                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "The response failed validation checks",
                            "error_type": "post_validation_failure",
                            "blocked_reason": "; ".join(post_val_result.blocking_issues),
                            "details": {
                                "grounding_score": post_val_result.grounding_score,
                                "numeric_score": post_val_result.numeric_score,
                                "regulatory_score": post_val_result.reg_score,
                                "suitability_score": post_val_result.suitability_score,
                                "tone_score": post_val_result.tone_score,
                                "warnings": post_val_result.warnings
                            }
                        }
                    )

            log_event(logger, "post_validation_passed", "pipeline", {
                "warnings_count": len(post_val_result.warnings)
            })

            # ---------------------------
            # STEP 6: Calculate Confidence Score
            # ---------------------------
            confidence = validation_pipeline.calculate_confidence(post_val_result, retrieval_score=0.8)

            advisor_handover = _should_advisor_handover(
                overall_score=confidence.overall_score,
                recommended_action=confidence.recommended_action,
            )

            # If confidence is too low, refuse rather than risking hallucination.
            if confidence.recommended_action == "refuse":
                response_text = _hallucination_fallback_message()
                low_confidence_triggers.append("confidence_refuse")

            # If confidence is low (or confidence model recommends partial), route to a human advisor.
            # Keep this as a 200 response to avoid brittle client flows.
            if advisor_handover and compliance_status != "BLOCKED":
                response_text = _advisor_handover_message()
                low_confidence_triggers.append("advisor_handover")

            log_event(logger, "confidence_calculated", "pipeline", {
                "overall_score": confidence.overall_score,
                "recommended_action": confidence.recommended_action,
                "advisor_handover": advisor_handover,
                "low_confidence_triggers": list(dict.fromkeys(low_confidence_triggers)),
                "handover_threshold": os.getenv("ADVISOR_HANDOVER_CONFIDENCE_THRESHOLD", "0.50"),
                "handover_enabled": os.getenv("ADVISOR_HANDOVER_ENABLED", "true"),
            })

            # ---------------------------
            # STEP 7: Update Conversation Context
            # ---------------------------
            ConversationContext.update_context(
                conversation_id=request.conversationId,
                agent=agent_type,
                profile_updates=extracted_profile
            )

            log_event(logger, "conversation_context_updated", "context", {
                "agent": agent_type,
                "profile_updates": extracted_profile
            })

            # ---------------------------
            # STEP 8: FODE (final formatting) - fail-open
            # ---------------------------
            # IMPORTANT: if compliance forced a safe answer (status=BLOCKED), return it verbatim.
            # FODE tone sanitization can weaken critical safety language (e.g., replacing "never").
            # Also skip FODE for low-confidence handover responses to keep wording deterministic.
            # Skip FODE for multi-agent orchestrated responses — they're already well-structured.
            is_multi_agent = len(agents_used) > 1
            if FODE_ENABLED and fode_engine and compliance_status != "BLOCKED" and not low_confidence_triggers and not is_multi_agent:
                try:
                    module_map = {
                        "credits_loans": "CREDIT",
                        "investment_coach": "SIP_INVESTMENT",
                        "insurance_analyzer": "INSURANCE",
                        "retirement_planner": "RETIREMENT",
                        "tax_planner": "TAX",
                        "fraud_shield": "FRAUD",
                    }
                    module = module_map.get(agent_type, "GENERIC")

                    intent_tags: List[str] = []
                    if intent:
                        intent_tags = [str(intent).upper()]

                    fode_ctx = {
                        "module": module,
                        "language": "en",
                        "channel": "YOUTH_APP",
                        "intent_tags": intent_tags,
                        "user_query": user_input,
                        "agent": agent_type,
                        "retrieved_passages": retrieved_passages,
                        "compliance_status": compliance_status,
                        "recommended_action": confidence.recommended_action,
                        "needs_clarification": (
                            advisor_handover
                            or confidence.recommended_action in ["clarify", "partial", "refuse"]
                        ),
                        "advisor_handover": advisor_handover,
                        "low_confidence_triggers": list(dict.fromkeys(low_confidence_triggers)),
                    }

                    out = fode_engine.run({"raw_answer": response_text, "context": fode_ctx})
                    response_text = out.get("final_answer", response_text)

                    log_event(
                        logger,
                        "fode_applied",
                        "fode",
                        {"applied": out.get("applied", []), "flags": out.get("flags", [])},
                    )
                except Exception as fe:
                    log_event(logger, "fode_error", "fode", {"error": str(fe)}, level="ERROR")
                    # fail-open
            else:
                skip_reasons = []
                if not FODE_ENABLED or not fode_engine: skip_reasons.append("disabled")
                if compliance_status == "BLOCKED": skip_reasons.append("compliance_blocked")
                if low_confidence_triggers: skip_reasons.append("low_confidence")
                if is_multi_agent: skip_reasons.append("multi_agent")
                if skip_reasons:
                    log_event(logger, "fode_skipped", "fode", {"reasons": skip_reasons})

            # ---------------------------
            # STEP 9: Return Response with full metadata
            # ---------------------------
            needs_clarification = advisor_handover or confidence.recommended_action in ["clarify", "partial", "refuse"]

            # Build validation checks summary for UI
            validation_checks_summary = []
            for vc in (post_val_result.validation_checks or []):
                validation_checks_summary.append({
                    "check": getattr(vc, "check_type", "unknown"),
                    "passed": getattr(vc, "severity", None) != "CRITICAL",
                    "severity": str(getattr(vc, "severity", "OK")),
                    "message": getattr(vc, "message", ""),
                })
            # Always add core checks even if no violations
            check_types_seen = {c["check"] for c in validation_checks_summary}
            for default_check in ["grounding", "regulatory", "suitability", "tone", "pii"]:
                if default_check not in check_types_seen:
                    validation_checks_summary.append({
                        "check": default_check,
                        "passed": True,
                        "severity": "OK",
                        "message": "Passed",
                    })

            # Build sources list from retrieved passages and evidence pack
            sources_list = []
            
            # Add production-grade citations from EvidencePack (if available)
            if 'evidence_pack' in locals() and evidence_pack and evidence_pack.citations:
                for citation in evidence_pack.citations:
                    sources_list.append({
                        "doc_id": citation.doc_id,
                        "source": citation.source,
                        "page": citation.page,
                        "chunk_id": citation.chunk_id,
                        "document": citation.source,  # For backward compatibility
                        "excerpt": citation.text[:200] if citation.text else "",
                        "score": citation.score,
                        "metadata": citation.metadata,
                        "agent": agent_type,
                        "type": "evidence_pack"  # Mark as production-grade citation
                    })
            
            # Add legacy retrieved passages (for backward compatibility)
            for p in retrieved_passages:
                # Skip if already added from evidence_pack
                if not any(s.get("type") == "evidence_pack" and s.get("excerpt") == (p.get("excerpt") or "")[:200] for s in sources_list):
                    sources_list.append({
                        "document": p.get("document", "Unknown"),
                        "excerpt": (p.get("excerpt") or "")[:200],
                        "score": p.get("similarity_score", 0),
                        "agent": p.get("agent", ""),
                        "type": "legacy"
                    })

            # Build pipeline steps for workflow indicator
            pipeline_steps = [
                {"step": "Guardrails", "status": "passed"},
                {"step": f"Routing → {agent_type}", "status": "passed"},
            ]
            if 'evidence_pack' in locals() and evidence_pack and evidence_pack.citations:
                pipeline_steps.append({
                    "step": f"Evidence Retrieval (Coverage: {evidence_pack.coverage})", 
                    "status": "passed" if evidence_pack.coverage == "sufficient" else "partial"
                })
            elif retrieved_passages:
                pipeline_steps.append({"step": "Evidence Retrieval", "status": "passed"})
            pipeline_steps.append({"step": "Compliance Check", "status": compliance_status.lower()})
            pipeline_steps.append({"step": "Quality Validation", "status": "passed" if not post_val_result.should_block else "blocked"})
            pipeline_steps.append({"step": "Confidence Scoring", "status": confidence.recommended_action})

            # Determine response mode
            response_mode = "guidance"
            if intent == "general":
                response_mode = "info"
            elif confidence.recommended_action in ["refuse", "partial"]:
                response_mode = "action"

            # Generate follow-up suggestions based on agent type
            follow_ups = _generate_follow_ups(agent_type, profile, needs_clarification)

            return QueryResponse(
                response=response_text,
                confidence_score=confidence.overall_score,
                needs_clarification=needs_clarification,
                agents=agents_used,
                recommended_action=confidence.recommended_action,
                compliance_status=compliance_status,
                validation_checks=validation_checks_summary,
                sources=sources_list if sources_list else None,
                profile_used=profile if profile else None,
                pipeline_steps=pipeline_steps,
                mode=response_mode,
                follow_ups=follow_ups,
            )

        except InputGuardrailTripwireTriggered as e:
            log_event(logger, "input_guardrail_triggered", "guardrails", {"error": str(e)}, level="ERROR")

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Your query triggered an input guardrail",
                    "error_type": "input_guardrail_violation",
                    "blocked_reason": "Query blocked by safety guardrails",
                    "details": {"guardrail_type": "input"}
                }
            )

        except OutputGuardrailTripwireTriggered as e:
            log_event(logger, "output_guardrail_triggered", "guardrails", {"error": str(e)}, level="ERROR")

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "The response triggered an output guardrail",
                    "error_type": "output_guardrail_violation",
                    "blocked_reason": "Response blocked due to safety or compliance concerns",
                    "details": {"guardrail_type": "output"}
                }
            )

        except HTTPException:
            # Re-raise HTTPException without modification so FastAPI handles it properly
            raise

        except ValidationError as e:
            log_event(logger, "validation_error", "server", {"error": str(e)}, level="ERROR")
            raise HTTPException(status_code=400, detail=f"Validation error: {e}")

        except Exception as e:
            log_event(logger, "unexpected_error", "server", {"error": str(e)}, level="ERROR")
            logger.exception("Unexpected error occurred", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Finorbit Unified Backend API!",
        "version": "2.0.0",
        "features": [
            "Integrated validation pipeline",
            "5 specialist agents",
            "Safety guardrails",
            "Quality validation",
            "Confidence scoring"
        ]
    }


@app.get("/ui")
async def ui_home():
    """Serve the FinOrbit UI"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline": "initialized",
        "specialist_agents": len(specialist_agents),
        "database": "connected" if DATABASE_URL else "not configured"
    }


# ---------------------------
# MCP Tool Endpoints
# ---------------------------
@app.post("/tools/knowledge_lookup")
async def mcp_knowledge_lookup(
    query: str,
    module: str,
    top_k: int = 5,
    doc_type: str = None,
    year: str = None,
    issuer: str = None,
    regulator_tag: str = None,
    security: str = None,
    is_current: bool = None,
    pii: bool = None,
    compliance_tags_any: list[str] = None,
):
    """
    MCP Tool: Query the RAG knowledge base for financial information
    
    This endpoint exposes the RAG knowledge lookup as an MCP tool that can be
    called by agents to retrieve relevant financial documents and information.
    
    Parameters:
        query: The search query text
        module: Domain module (credit, investment, insurance, retirement, taxation)
        top_k: Number of chunks to retrieve (max 5, default 5)
        doc_type: Optional filter by document type
        year: Optional filter by document year
        issuer: Optional filter by document issuer
        regulator_tag: Optional filter by regulatory tag
        security: Optional filter by security level
        is_current: Optional filter for current documents only
        pii: Optional filter documents containing PII
        compliance_tags_any: Optional filter by compliance tags (any match)
        
    Returns:
        Dict with:
        - found: Whether relevant documents were found
        - results: List of document chunks with content and metadata
        - total_chunks: Number of chunks returned
        - error: Error message if any
    """
    try:
        log_event(logger, "mcp_tool_called", "rag_tool", {
            "query_length": len(query),
            "module": module,
            "top_k": top_k
        })
        
        result = await knowledge_lookup(
            query=query,
            module=module,
            top_k=top_k,
            doc_type=doc_type,
            year=year,
            issuer=issuer,
            regulator_tag=regulator_tag,
            security=security,
            is_current=is_current,
            pii=pii,
            compliance_tags_any=compliance_tags_any,
        )
        
        log_event(logger, "mcp_tool_complete", "rag_tool", {
            "found": result.get("found", False),
            "results_count": len(result.get("results", []))
        })
        
        return result
        
    except Exception as e:
        log_event(logger, "mcp_tool_error", "rag_tool", {
            "error": str(e)
        }, level="ERROR")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"RAG tool error: {str(e)}",
                "found": False,
                "results": []
            }
        )


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("[START] Starting uvicorn server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
