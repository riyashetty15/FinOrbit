# ==============================================
# File: src/agents/retirement_planner.py
# Description: LangGraph State-Based retirement_planner.py (XML schema aligned)
# ==============================================

from __future__ import annotations
from typing import Dict, Any, List, Optional, TypedDict, Tuple
from dataclasses import dataclass, field
import os
import re
import math
import xml.etree.ElementTree as ET

from langgraph.graph import StateGraph, START, END


# ----------------- State Type ------------------

class RetirementState(TypedDict, total=False):
    # Inputs
    query: str
    profile: Dict[str, Any]
    transactions: List[Dict[str, Any]]

    # user profile normalized
    user_name: str
    age: int
    income: int
    savings: int
    expenses: int
    retirement_age: int
    life_expectancy: int
    risk_tolerance: str  # "low" | "moderate" | "high"
    current_investments: Dict[str, Any]

    # derived
    annual_savings: float
    retirement_corpus_needed: float
    retirement_gap: float
    monthly_investment_required: float

    # product search
    criteria: Dict[str, Any]
    query_info: Dict[str, Any]
    products: List[Dict[str, Any]]

    # mode
    analysis_mode: str  # "product_search" | "gap_analysis" | "profile_analysis"

    # results
    summary: str
    recommendations: List[str]
    next_best_actions: List[str]
    product_search_meta: Dict[str, Any]

    # error
    error: Optional[str]


# ----------------- Config ----------------------

@dataclass
class RetirementConfig:
    default_age: int = 35
    default_income: int = 500_000
    default_savings: int = 0
    default_expenses: int = 0
    default_retirement_age: int = 60
    default_life_expectancy: int = 85
    inflation_rate: float = 0.06
    return_rate: float = 0.08
    xml_path: str = os.path.join("data", "all_retirement_offers.xml")
    max_products: int = 5


# --------------- Agent Implementation ------------

class RetirementPlannerAgent:
    """LangGraph-based Retirement Planner with optional XML-backed product search"""

    SUPPORTED_PRODUCT_TYPES = ("annuity", "nps", "pension_plan", "retirement_fund", "pension_scheme", "ppf")

    def __init__(self, config: Optional[RetirementConfig] = None):
        self.config = config or RetirementConfig()
        self.product_data: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.SUPPORTED_PRODUCT_TYPES}
        self._load_product_data()
        self.graph = self._build_graph()

    # ==================== XML LOADING ====================

    def _load_product_data(self) -> None:
        xml_path = self.config.xml_path
        if not os.path.exists(xml_path):
            print(f"[retirement_planner] XML not found at {xml_path} — product search will be empty.")
            return
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            self.product_data = {t: [] for t in self.SUPPORTED_PRODUCT_TYPES}
            for prod in root.findall("Product"):
                p = self._parse_product_node(prod)
                if not p:
                    continue
                mapped = self._map_product_type(p.get("product_type", ""))
                if not mapped:
                    continue
                p["product_type"] = mapped
                self.product_data[mapped].append(p)
        except Exception as e:
            print(f"[retirement_planner] Error loading XML: {e}")

    def _map_product_type(self, pt: str) -> Optional[str]:
        t = (pt or "").strip().lower()
        if t in ("annuity", "life annuity"):
            return "annuity"
        if t in ("nps", "national pension system"):
            return "nps"
        if t in ("pension", "pension plan", "superannuation"):
            return "pension_plan"
        if t in ("rp", "retirement fund", "retirement mutual fund", "rpf", "retirement_fund"):
            return "retirement_fund"
        if t in ("government pension", "pension scheme"):
            return "pension_scheme"
        if t == "ppf" or "public provident fund" in t:
            return "ppf"
        return t if t in self.SUPPORTED_PRODUCT_TYPES else None

    def _parse_num(self, text: Optional[str], f=float) -> float:
        if text is None:
            return 0.0
        try:
            s = re.sub(r"[^0-9.]+", "", str(text))
            return f(s) if s != "" else 0.0
        except Exception:
            return 0.0

    def _parse_product_node(self, elem: ET.Element) -> Optional[Dict[str, Any]]:
        try:
            name = elem.attrib.get("name", elem.findtext("Name", "Unknown")).strip()
            get = lambda tag, default="": (elem.findtext(tag, default) or "").strip()
            parsed = {
                "name": name,
                "product_type": get("Type", ""),
                "expected_return": float(self._parse_num(get("ExpectedReturn"), float)),
                "min_investment": float(self._parse_num(get("MinInvestment"), float)),
                "payout_frequency": get("PayoutFrequency", ""),
                "tenure_years": int(self._parse_num(get("TenureYears"), float)),
                "fees": float(self._parse_num(get("Fees"), float)),
                "features": get("Features", ""),
                "eligibility": get("Eligibility", ""),
                "notes": get("Notes", ""),
            }
            return parsed
        except Exception as e:
            print(f"[retirement_planner] Bad product node: {e}")
            return None

    # ==================== NODE: INITIALIZE ====================

    def _node_initialize(self, state: RetirementState) -> RetirementState:
        try:
            profile = state.get("profile", {}) or {}
            user_name = profile.get("name", "User")
            age = int(profile.get("age", self.config.default_age))
            income = int(profile.get("income", self.config.default_income))
            savings = int(profile.get("savings", self.config.default_savings))
            expenses = int(profile.get("expenses", self.config.default_expenses))
            retirement_age = int(profile.get("retirement_age", self.config.default_retirement_age))
            life_expectancy = int(profile.get("life_expectancy", self.config.default_life_expectancy))
            risk_tolerance = (profile.get("risk_tolerance") or "").lower() or "moderate"
            current_investments = profile.get("investments", {})

            query = state.get("query", "")
            if re.search(r"retire at (\d+)", query):
                retirement_age = int(re.search(r"retire at (\d+)", query).group(1))
            if re.search(r"with\s+₹([\d,]+)\s*\/\s*month", query):
                expenses = int(re.search(r"with\s+₹([\d,]+)\s*\/\s*month", query).group(1).replace(",", "")) * 12

            annual_savings = max(income - expenses, 0)
            
            criteria = self._extract_criteria_from_query(query)
            query_info = self._parse_query_info(query)

            new_state = dict(state)
            new_state.update({
                "user_name": user_name,
                "age": age,
                "income": income,
                "savings": savings,
                "expenses": expenses,
                "retirement_age": retirement_age,
                "life_expectancy": life_expectancy,
                "risk_tolerance": risk_tolerance,
                "current_investments": current_investments,
                "annual_savings": annual_savings,
                "criteria": criteria,
                "query_info": query_info,
                "error": None
            })
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Initialization error: {str(e)}"
            return new_state

    # ==================== NODE: DETERMINE MODE ====================

    def _node_determine_mode(self, state: RetirementState) -> RetirementState:
        try:
            criteria = state.get("criteria", {}) or {}
            query = (state.get("query") or "").lower()
            
            # Prioritize gap analysis if the query asks for a plan or savings advice
            plan_keywords = ["suggest a retirement plan", "how to retire", "sip required", "how much to save", "gap analysis"]
            if any(w in query for w in plan_keywords) or (criteria.get('product_type') and any(w in query for w in ["including", "and", "mix of"])):
                mode = "gap_analysis"
            # Fallback to product search if a specific product type is mentioned without planning keywords
            elif criteria.get("product_type"):
                mode = "product_search"
            # Default to profile analysis if no specific intent is detected
            else:
                mode = "profile_analysis"
            
            new_state = dict(state)
            new_state["analysis_mode"] = mode
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Mode determination error: {str(e)}"
            return new_state

    # ==================== NODE: PRODUCT SEARCH ====================

    def _node_product_search(self, state: RetirementState) -> RetirementState:
        try:
            criteria = state.get("criteria", {}) or {}
            products = self._search_products(criteria)

            if not products:
                summary = f"No retirement products found matching your criteria: {criteria}"
                actions = [
                    "Loosen filters (min investment / desired return / tenure)",
                    "Consider broader product types (NPS, Retirement funds, Annuities)",
                    "Contact a retirement planner for custom advice"
                ]
            else:
                lines = [f"Found {len(products)} products matching your criteria:\n"]
                for i, p in enumerate(products, 1):
                    lines.append(f"\n{i}. **{p['name']}** ({p.get('product_type')})")
                    lines.append(f"   - Expected return: {p.get('expected_return', 0):g}%")
                    if p.get("min_investment"):
                        lines.append(f"   - Min investment: ₹{int(p.get('min_investment')):,}")
                    if p.get("tenure_years") != 0:
                        lines.append(f"   - Tenure: {p.get('tenure_years')} years")
                    if p.get("payout_frequency"):
                        lines.append(f"   - Payout: {p.get('payout_frequency')}")
                    if p.get("fees"):
                        lines.append(f"   - Fees: {p.get('fees'):,}")
                    feats = (p.get("features") or "").strip()
                    if feats:
                        lines.append(f"   - Features: {feats}")
                    elig = (p.get("eligibility") or "").strip()
                    if elig:
                        lines.append(f"   - Eligibility: {elig}")
                lines.append(f"\nSearch criteria used: {criteria}")
                summary = "\n".join(lines)
                actions = [
                    "Compare expected return vs fees and payout schedule",
                    "Check product liquidity & exit rules",
                    "Confirm tax treatment and nominee rules"
                ]

            new_state = dict(state)
            new_state.update({
                "products": products,
                "summary": summary,
                "next_best_actions": actions,
                "product_search_meta": {"count": len(products), "criteria": criteria}
            })
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Product search error: {str(e)}"
            return new_state

    # ==================== NODE: GAP ANALYSIS ====================

    def _node_gap_analysis(self, state: RetirementState) -> RetirementState:
        try:
            age = state.get("age", self.config.default_age)
            retirement_age = state.get("retirement_age", self.config.default_retirement_age)
            life_expectancy = state.get("life_expectancy", self.config.default_life_expectancy)
            expenses = state.get("expenses", self.config.default_expenses)
            savings = state.get("savings", self.config.default_savings)
            
            if retirement_age <= age:
                raise ValueError("Retirement age must be greater than current age.")

            years_to_ret = max(retirement_age - age, 0)
            years_in_ret = max(life_expectancy - retirement_age, 0)
            
            future_expenses = (expenses / 12) * ((1 + self.config.inflation_rate) ** years_to_ret)

            corpus_needed = (future_expenses * 12) / (self.config.return_rate)

            fv_savings = savings * ((1 + self.config.return_rate) ** years_to_ret)

            gap = max(corpus_needed - fv_savings, 0)

            r = self.config.return_rate / 12
            n = years_to_ret * 12
            if n > 0 and r > 0:
                sip = gap * r / (((1 + r) ** n) - 1)
            elif n > 0:
                sip = gap / n
            else:
                sip = 0

            summary = (
                f"Based on your goal to retire at age {retirement_age} with a monthly income of "
                f"₹{int(expenses/12):,}, you will need a corpus of approximately "
                f"₹{int(corpus_needed):,} to sustain your lifestyle. "
                f"To bridge the gap, a **monthly SIP of ₹{int(sip):,} is required.**\n\n"
                "Here is a plan for a **moderate risk profile**:"
            )

            products_to_suggest = [
                self.product_data.get("retirement_fund", [None])[0],
                self.product_data.get("nps", [None])[0],
                self.product_data.get("ppf", [None])[0],
                self.product_data.get("annuity", [None])[0],
            ]
            
            product_lines = []
            for prod in products_to_suggest:
                if prod:
                    product_lines.append(f"- **{prod['name']}**: {prod.get('notes', prod.get('features'))}")
            
            if product_lines:
                summary += "\n" + "\n".join(product_lines)

            recs = [
                f"Start a monthly SIP of at least ₹{int(sip):,}",
                "Open an NPS account for tax-efficient savings and hybrid returns",
                "Start a PPF account for a safe, government-backed component",
                "Consult a financial advisor to create a detailed, personalized plan",
                "Regularly review your plan and adjust your contributions based on market performance"
            ]

            new_state = dict(state)
            new_state.update({
                "retirement_corpus_needed": corpus_needed,
                "retirement_gap": gap,
                "monthly_investment_required": sip,
                "summary": summary,
                "recommendations": recs
            })
            return new_state
        except ValueError as ve:
            new_state = dict(state)
            new_state["error"] = f"Calculation error: {str(ve)}. Please provide a valid retirement age."
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Gap analysis error: {str(e)}"
            return new_state

    # ==================== NODE: PROFILE ANALYSIS ====================

    def _node_profile_analysis(self, state: RetirementState) -> RetirementState:
        try:
            name = state.get("user_name", "User")
            age = state.get("age", self.config.default_age)
            income = state.get("income", self.config.default_income)
            savings = state.get("savings", self.config.default_savings)
            expenses = state.get("expenses", self.config.default_expenses)
            risk = state.get("risk_tolerance", "moderate")

            savings_rate = (savings / income * 100) if income > 0 else 0.0
            summary = (
                f"Hello {name}. At age {age} with annual income ₹{income:,}, "
                f"your current savings are ₹{savings:,} and your savings rate is approximately {savings_rate:.1f}%."
            )

            recs = []
            if savings_rate < 10:
                recs.append("Increase your savings rate to at least 15% of your income.")
            else:
                recs.append("Maintain or increase your current savings rate to meet your retirement goals.")

            if risk == "low":
                recs.append("Consider debt funds, annuities, and government schemes like PPF.")
            elif risk == "high":
                recs.append("Explore mutual funds (equity) and other market-linked options for higher growth.")
            else:
                recs.append("A balanced portfolio with a mix of equity and debt is ideal.")

            recs.extend([
                "Review your investment portfolio annually",
                "Maximize contributions to tax-advantaged retirement schemes (NPS / PPF)",
                "Build and maintain an emergency fund of 6-12 months of expenses"
            ])

            new_state = dict(state)
            new_state.update({"summary": summary, "recommendations": recs})
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Profile analysis error: {str(e)}"
            return new_state

    # ==================== HELPERS: QUERY PARSING ====================

    _UNIT_SCALE = {
        "k": 1_000, "thousand": 1_000,
        "lakh": 100_000, "lac": 100_000, "lacs": 100_000,
        "crore": 10_000_000, "cr": 10_000_000,
        "m": 1_000_000, "million": 1_000_000,
    }

    def _normalize_amount(self, number: float, unit: Optional[str]) -> int:
        if not unit:
            return int(number)
        unit = unit.lower()
        scale = self._UNIT_SCALE.get(unit, 1)
        return int(float(number) * scale)

    def _extract_amount_phrases(self, text: str) -> Dict[str, int]:
        q = text.lower()
        res: Dict[str, int] = {}
        m = re.search(r"between\s+(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac|lacs|crore|cr|m|million)?\s+and\s+(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac|lacs|crore|cr|m|million)?", q)
        if m:
            low = self._normalize_amount(float(m.group(1)), m.group(2))
            high = self._normalize_amount(float(m.group(3)), m.group(4))
            if low and high and low <= high:
                res["min_investment"] = low
                res["max_investment"] = high
                return res
        m = re.search(r"\b(under|below|up\s*to|upto|less\s*than)\s+(?:rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|lakh|lac|lacs|crore|cr|m|million)?\b", q)
        if m:
            res["max_investment"] = self._normalize_amount(float(m.group(2).replace(',', '')), m.group(3))
        m = re.search(r"\b(above|over|at\s*least|minimum|min)\s+(?:rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|lakh|lac|lacs|crore|cr|m|million)?\b", q)
        if m:
            res["min_investment"] = self._normalize_amount(float(m.group(2).replace(',', '')), m.group(3))
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", q)]
        if nums:
            res["min_return"] = min(nums)
        return res

    def _extract_tenure_years(self, text: str) -> Optional[int]:
        t = text.lower()
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(year|years|yr|yrs)\b", t)
        if m:
            return int(round(float(m.group(1))))
        return None

    def _extract_product_type(self, text: str) -> Optional[str]:
        q = text.lower()
        if any(w in q for w in ("annuity", "annuity plan", "annuit")):
            return "annuity"
        if "nps" in q or "national pension" in q:
            return "nps"
        if "pension" in q and "plan" in q:
            return "pension_plan"
        if any(w in q for w in ("retirement fund", "retirement mutual", "rpf")):
            return "retirement_fund"
        if "pension scheme" in q or "government scheme" in q:
            return "pension_scheme"
        if "ppf" in q or "public provident fund" in q:
            return "ppf"
        return None

    def _extract_payout(self, text: str) -> Optional[str]:
        q = text.lower()
        if "monthly" in q: return "monthly"
        if "quarter" in q: return "quarterly"
        if "annual" in q or "yearly" in q: return "annual"
        return None

    def _extract_criteria_from_query(self, query: str) -> Dict[str, Any]:
        q = (query or "").lower()
        c: Dict[str, Any] = {}
        pt = self._extract_product_type(q)
        if pt:
            c["product_type"] = pt
        c.update(self._extract_amount_phrases(q))
        tenure = self._extract_tenure_years(q)
        if tenure is not None:
            c["tenure_years"] = tenure
        payout = self._extract_payout(q)
        if payout:
            c["payout_frequency"] = payout
        m = re.search(r"(\d+(?:\.\d+)?)\s*%.*return", q)
        if m:
            c["min_return"] = float(m.group(1))
        if "low risk" in q or "conservative" in q:
            c["risk"] = "low"
        if "high risk" in q or "aggressive" in q:
            c["risk"] = "high"
        return c

    def _parse_query_info(self, query: str) -> Dict[str, Any]:
        info = {"type": None, "amount": None, "specific_request": False}
        pt = self._extract_product_type(query)
        if pt:
            info["type"] = pt
            info["specific_request"] = True
        return info

    # ==================== SEARCH PRODUCTS ====================

    def _search_products(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        prod_type = criteria.get("product_type")
        candidates: List[Dict[str, Any]] = []
        if prod_type:
            candidates = self.product_data.get(prod_type, [])
        else:
            for lst in self.product_data.values():
                candidates.extend(lst)
        if not candidates:
            return []

        min_inv = float(criteria.get("min_investment", 0))
        max_inv = float(criteria.get("max_investment", float('inf')))
        min_ret = float(criteria.get("min_return", 0))
        tenure = int(criteria.get("tenure_years", 0))
        payout = criteria.get("payout_frequency")

        filtered: List[Dict[str, Any]] = []
        for p in candidates:
            if p.get("min_investment", 0) > max_inv:
                continue
            if p.get("expected_return", 0) < min_ret:
                continue
            if tenure > 0 and p.get("tenure_years", 0) < tenure:
                continue
            if payout and payout not in p.get("payout_frequency", "").lower():
                continue
            filtered.append(p)

        filtered.sort(key=lambda x: (-float(x.get("expected_return", 0)), float(x.get("fees", 0)), float(x.get("min_investment", 0))))
        return filtered[: self.config.max_products]

    # ==================== CONDITIONAL ROUTING ====================

    def _should_route(self, state: RetirementState) -> str:
        if state.get("error"):
            return "end"
        return state.get("analysis_mode", "profile_analysis")

    # ==================== BUILD GRAPH ====================

    def _build_graph(self):
        graph = StateGraph(RetirementState)
        graph.add_node("initialize", self._node_initialize)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("product_search", self._node_product_search)
        graph.add_node("gap_analysis", self._node_gap_analysis)
        graph.add_node("profile_analysis", self._node_profile_analysis)

        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "determine_mode")
        graph.add_conditional_edges(
            "determine_mode",
            self._should_route,
            {
                "product_search": "product_search",
                "gap_analysis": "gap_analysis",
                "profile_analysis": "profile_analysis",
                "end": END
            }
        )
        graph.add_edge("product_search", END)
        graph.add_edge("gap_analysis", END)
        graph.add_edge("profile_analysis", END)
        return graph.compile()

    # ==================== PUBLIC RUN ====================

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            init_state: RetirementState = {
                "query": state.get("query", ""),
                "profile": state.get("profile", {}),
                "transactions": state.get("transactions", []),
            }
            final = self.graph.invoke(init_state)
            result: Dict[str, Any] = {
                "summary": final.get("summary", "Retirement analysis complete"),
                "recommendations": final.get("recommendations", final.get("next_best_actions", [])),
                "mode": final.get("analysis_mode", "unknown")
            }
            if final.get("analysis_mode") == "product_search":
                result.update({
                    "products": final.get("products", []),
                    "product_search_meta": final.get("product_search_meta", {})
                })
            else:
                result.update({
                    "retirement_gap": final.get("retirement_gap"),
                    "corpus_needed": final.get("retirement_corpus_needed"),
                    "monthly_investment_required": final.get("monthly_investment_required"),
                })
            return result
        except Exception as e:
            return {
                "summary": f"Retirement analysis error: {str(e)}",
                "recommendations": ["Please try again with a different query"],
                "mode": "error"
            }


# --------------- Wrapper --------------------

class RetirementPlanner:
    """Orchestrator-compatible wrapper for RetirementPlannerAgent"""

    def __init__(self, config: Optional[RetirementConfig] = None):
        self.agent = RetirementPlannerAgent(config=config)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run retirement planning with intent-aware routing

        Flow:
        1. Check query intent (general vs personalized)
        2. For general queries: Provide educational retirement planning information
        3. For personalized queries: Validate profile completeness
        4. If profile incomplete: Request missing fields
        5. If profile complete: Run LangGraph analysis
        """
        from backend.agents.agent_helpers import (
            ProfileValidator,
            is_general_query,
            format_general_info_response
        )

        query = state.get("query", "")
        intent = state.get("intent", "personalized")
        profile = state.get("profile", {})

        # Handle general queries - provide retirement planning education without personalization
        if is_general_query(query, intent):
            return self._provide_general_retirement_info(query)

        # For personalized queries, check profile completeness
        required_fields = ["age", "income", "retirement_age"]
        missing_fields = ProfileValidator.get_missing_fields(profile, required_fields)

        if missing_fields:
            # Generate request for missing information
            message = ProfileValidator.generate_missing_fields_message(
                missing_fields,
                agent_name="Retirement Planner",
                query_context="create a personalized retirement plan and estimate your retirement corpus needs"
            )
            return {
                "summary": message,
                "next_best_actions": [f"Provide your {field.replace('_', ' ')}" for field in missing_fields],
                "mode": "awaiting_profile",
                "requires_input": True,
                "missing_fields": missing_fields
            }

        # Profile is complete, proceed with personalized analysis using LangGraph
        return self.agent.run(state)

    def _provide_general_retirement_info(self, query: str) -> Dict[str, Any]:
        """Provide general retirement planning information without personalization"""
        from backend.agents.agent_helpers import format_general_info_response

        query_lower = query.lower()

        # Determine what type of information to provide
        if any(word in query_lower for word in ["nps", "national pension", "tier 1", "tier 2"]):
            info = """**NPS (National Pension System): Government-Backed Retirement Savings**

**What is NPS?**
NPS is a voluntary retirement savings scheme regulated by PFRDA (Pension Fund Regulatory and Development Authority). It helps build a retirement corpus through systematic contributions.

**Key Features:**
• **Minimum Investment**: ₹500/year to keep account active
• **No Maximum Limit** on contributions
• **Lock-in**: Till age 60 (partial withdrawal allowed)
• **Tax Benefits**: Triple tax benefit (EEE for Tier I)
• **Low Cost**: Fund management fees ~0.09% only

**Two Account Types:**

**Tier I (Retirement Account):**
• Lock-in till 60 years
• Tax benefits under Sections 80C, 80CCD(1B), 80CCD(2)
• Cannot withdraw before retirement (except specific cases)
• Mandatory for government employees

**Tier II (Voluntary Savings):**
• No lock-in period
• Withdraw anytime
• No tax benefits
• Requires active Tier I account

**Investment Options:**

**Asset Classes:**
• **Equity (E)**: Stock market (max 75%)
• **Corporate Bonds (C)**: Debt instruments (max 100%)
• **Government Securities (G)**: G-secs (max 100%)
• **Alternative Assets (A)**: REITs, InvITs (max 5%)

**Auto Choice:**
• **Aggressive**: Higher equity allocation (up to 75%)
• **Moderate**: Balanced approach
• **Conservative**: Minimal equity (up to 25%)
• Auto-rebalancing based on age

**Active Choice:**
• Decide your own asset allocation
• Can change twice a year

**Tax Benefits:**

**Section 80C**: Up to ₹1.5 lakhs
**Section 80CCD(1B)**: Additional ₹50,000 (ONLY for NPS)
**Section 80CCD(2)**: Employer contribution up to 10-14% of salary
**Total max benefit**: ₹2 lakhs + employer contribution

**At Maturity (Age 60):**
• **Withdraw**: Up to 60% as lump sum (tax-free)
• **Mandatory Annuity**: Minimum 40% must buy annuity (provides monthly pension)

**Annuity Options:**
1. Life annuity with return of purchase price
2. Life annuity without return
3. Joint life annuity
4. Annuity for specific period

**Premature Exit (Before 60):**
• **Allowed after 3 years** for specific reasons
• **Withdraw**: Only 20% lump sum
• **Mandatory Annuity**: 80% must buy annuity

**Pension Fund Managers:**
• SBI Pension Fund
• HDFC Pension Management
• ICICI Prudential Pension Fund
• UTI Retirement Solutions
• LIC Pension Fund
• Kotak Mahindra Pension Fund
• Aditya Birla Sun Life Pension Management

**Historical Returns (10-year average):**
• Equity: ~12-14% p.a.
• Corporate Bonds: ~8-10% p.a.
• Government Securities: ~8-9% p.a.

**Pros:**
• Excellent tax benefits (extra ₹50k under 80CCD(1B))
• Low cost (cheapest pension product)
• Portable across jobs
• Government-regulated
• Market-linked returns

**Cons:**
• Mandatory annuity purchase (40%)
• Lock-in till 60
• Annuity income is taxable
• Equity exposure capped at 75%

**Who Should Invest?**
• Salaried employees (especially for 80CCD(1B) benefit)
• Self-employed individuals
• Those seeking additional retirement corpus
• Individuals exhausting 80C limit"""

        elif any(word in query_lower for word in ["ppf", "public provident fund"]):
            info = """**PPF (Public Provident Fund): Safe Long-Term Savings**

**What is PPF?**
A government-backed savings scheme offering guaranteed returns and complete safety. Ideal for risk-averse investors seeking long-term wealth creation with tax benefits.

**Key Features:**
• **Minimum Deposit**: ₹500/year
• **Maximum Deposit**: ₹1.5 lakhs/year
• **Interest Rate**: 7.1% p.a. (Q4 FY 2024-25, revised quarterly)
• **Tenure**: 15 years (extendable in blocks of 5 years)
• **Tax Status**: EEE (Exempt-Exempt-Exempt)
• **Deposit Frequency**: Lump sum or installments (max 12/year)

**Account Opening:**
• Open at Post Office or authorized banks
• One account per person (joint account not allowed)
• Minor account allowed (by parent/guardian)
• Can be opened online via net banking

**Tax Benefits:**

**Section 80C**: Deposit up to ₹1.5 lakhs deductible
**Interest**: Completely tax-free
**Maturity**: Withdrawal tax-free

**Withdrawal Rules:**

**Partial Withdrawal:**
• Allowed from 7th year onwards
• Maximum: 50% of balance at end of 4th preceding year
• Or 50% of balance at end of preceding year
• Only one withdrawal per year

**Full Withdrawal:**
• After 15 years
• Can extend in 5-year blocks
  - **With contribution**: Continue deposits
  - **Without contribution**: No new deposits, earn interest

**Premature Closure:**
• Allowed after 5 years in specific cases:
  - Medical emergency (life-threatening illness)
  - Higher education of self/children
  - Change of residency status
• Penalty may apply

**Loan Facility:**
• Available from 3rd to 6th year
• Maximum: 25% of balance at end of 2nd preceding year
• Interest: 2% above PPF interest rate
• Must be repaid within 36 months

**Example Calculation:**

**Investment**: ₹1.5 lakhs/year for 15 years
**Interest Rate**: 7.1% p.a. (compounded annually)
**Total Investment**: ₹22.5 lakhs
**Maturity Value**: ~₹40.5 lakhs
**Tax-free Gains**: ~₹18 lakhs

**Returns Comparison:**
• 7.1% tax-free = ~10% taxable return (for 30% tax bracket)
• Better than FD post-tax (~5-6%)
• Lower than equity long-term (~12-15%)

**Pros:**
• Government-backed (100% safe)
• Tax-free returns (EEE status)
• Attractive interest rate
• Disciplined long-term savings
• Loan facility available
• No TDS deduction

**Cons:**
• Long lock-in (15 years)
• Limited liquidity (withdrawal from 7th year)
• Returns lower than equity
• Maximum ₹1.5L/year cap
• Interest rate subject to quarterly revision

**PPF vs NPS vs EPF:**

| Feature | PPF | NPS | EPF |
|---------|-----|-----|-----|
| **Returns** | 7.1% guaranteed | 10-12% market-linked | 8.25% guaranteed |
| **Tax on maturity** | Tax-free | 60% tax-free | Tax-free |
| **Lock-in** | 15 years | Till 60 years | Till retirement |
| **Contribution** | Up to ₹1.5L | Unlimited | 12% of basic |
| **Risk** | Zero | Market risk | Zero |

**Who Should Invest?**
• Risk-averse investors
• Those seeking guaranteed returns
• Long-term wealth creators
• Retirement planners (supplement to NPS/EPF)
• Individuals wanting to exhaust 80C limit"""

        elif any(word in query_lower for word in ["epf", "pf", "provident fund", "employee provident"]):
            info = """**EPF (Employees' Provident Fund): Retirement Savings for Salaried**

**What is EPF?**
A mandatory retirement savings scheme for salaried employees in India. Both employee and employer contribute 12% of basic salary monthly to build retirement corpus.

**Key Features:**
• **Contribution**: 12% of (Basic + DA) by employee + employer
• **Interest Rate**: 8.25% p.a. (FY 2023-24)
• **Tax-Free**: Interest and maturity amount
• **Managed by**: EPFO (Employees' Provident Fund Organisation)
• **Mandatory**: For companies with 20+ employees

**Contribution Breakdown:**

**Employee Contribution (12% of Basic):**
• Entire 12% goes to EPF

**Employer Contribution (12% of Basic):**
• 3.67% to EPF
• 8.33% to EPS (Employee Pension Scheme, capped at ₹1,250/month)

**Example:**
Basic Salary: ₹40,000/month

**Employee**: ₹4,800/month to EPF
**Employer**: ₹1,468 to EPF + ₹3,332 to EPS
**Total EPF**: ₹6,268/month
**Annual EPF**: ₹75,216

**Interest Calculation:**
• Compounded annually
• Credited to account yearly
• Interest on contributions made during year
• Rate revised annually by EPFO

**Withdrawal Rules:**

**Full Withdrawal:**
• After retirement (58 years)
• 2 months of unemployment (after leaving job)
• Migration abroad permanently
• Medical emergency

**Partial Withdrawal:**
Allowed for specific purposes:

**50% withdrawal:**
• Purchase/construction of house (after 5 years)
• Repayment of home loan (after 10 years)
• Renovation/repairs (after 5 years)

**90% withdrawal:**
• 1 year before retirement (for advance)

**6 months salary:**
• Marriage of self/children/siblings
• Higher education of self/children

**Medical:**
• Serious illness of self/family members

**Advance:**
• After 1 month of unemployment

**Tax Implications:**

**Contribution:**
• Employee: Deduction under Section 80C (up to ₹1.5 lakhs)
• Employer: Deduction under Section 80C (up to ₹1.5 lakhs)

**Interest:**
• Tax-free if employed for minimum 5 years
• Taxable if withdrawn before 5 years (continuous service)

**Withdrawal:**
• **Tax-free** if continuous service ≥5 years
• **Taxable** if withdrawn before 5 years (added to income)

**EPS (Employee Pension Scheme):**
• Provides monthly pension after 58 years
• Minimum 10 years service required
• Pension amount based on pensionable salary and years of service
• Formula: (Pensionable Salary × Service Years) ÷ 70

**EPF vs VPF (Voluntary Provident Fund):**

**VPF:**
• Additional voluntary contribution beyond 12%
• Same interest as EPF (8.25%)
• Same tax benefits
• Lock-in till retirement/withdrawal conditions
• Can contribute up to 100% of basic

**UAN (Universal Account Number):**
• Single account number for life
• Portable across jobs
• Link Aadhaar for easy transfers
• Check balance online via UMANG app/EPFO portal

**Example Long-Term Growth:**

**Assumptions:**
• Starting Basic: ₹40,000/month
• Annual increment: 8%
• EPF contribution: 12% employee + 3.67% employer = 15.67%
• Interest: 8.25% p.a.
• Career span: 30 years

**Result:**
• Total contributions: ~₹1.6 crores
• Maturity value: ~₹3.5-4 crores (tax-free)

**Pros:**
• Forced savings (mandatory)
• Employer contribution (free money)
• High guaranteed returns (8.25%)
• Tax-free at maturity
• Portable across jobs
• Loan facility (against EPF)

**Cons:**
• Limited liquidity (lock-in)
• Capped contribution (12% only)
• TDS on early withdrawal
• Paperwork for transfers
• Returns lower than equity

**Tips:**
• Link Aadhaar for seamless transfers
• Check balance annually
• Transfer to new employer within 6 months
• Don't withdraw prematurely (tax + loss of compounding)
• Consider VPF for extra savings"""

        else:
            # General retirement planning overview
            info = """**Retirement Planning: Secure Your Golden Years**

**Why Retirement Planning?**
• Life expectancy increasing (75-80 years)
• Inflation erodes savings (~6% annually)
• Medical costs rising faster than inflation
• Pension insufficient for comfortable lifestyle
• 20-30 years of expenses post-retirement

**Key Retirement Principles:**

**1. Start Early**
• Power of compounding magnifies small contributions
• ₹5,000/month from age 25 → ₹2+ crores by 60
• ₹15,000/month from age 40 → ₹1.2 crores by 60

**2. Estimate Corpus Needed**

**Simple Formula:**
Retirement Corpus = (Current Monthly Expenses × 12 × Life Expectancy Post-Retirement) / (1 - Inflation Rate)

**Example:**
• Current expenses: ₹50,000/month
• Retirement age: 60
• Life expectancy: 85 years
• Required corpus: ₹2-3 crores (inflation-adjusted)

**3. Multiple Income Sources**
Don't rely on single source:
• EPF/PPF
• NPS
• Equity investments
• Real estate rental
• Pension/annuity
• Part-time work

**Retirement Investment Options:**

**Government Schemes:**

**EPF (Employees' Provident Fund)**
• Mandatory for salaried
• 12% employee + 12% employer contribution
• 8.25% interest
• Tax-free maturity
• Lock-in till retirement

**NPS (National Pension System)**
• Voluntary for all
• Extra ₹50,000 tax benefit under 80CCD(1B)
• Market-linked returns (10-12%)
• Mandatory annuity (40%)
• Lock-in till 60

**PPF (Public Provident Fund)**
• 7.1% tax-free returns
• 15-year lock-in
• Max ₹1.5 lakhs/year
• Government-backed safety

**Senior Citizen Savings Scheme (SCSS)**
• For 60+ years only
• 8.2% interest (quarterly payout)
• Max ₹30 lakhs investment
• 5-year tenure (extendable by 3 years)

**Post Office Monthly Income Scheme (POMIS)**
• Regular monthly income
• 7.4% interest
• Max ₹9 lakhs (single), ₹15 lakhs (joint)
• 5-year maturity

**Market-Linked Options:**

**Equity Mutual Funds**
• Long-term: 12-15% returns
• SIP recommended
• Equity for growth (till 50 years)
• Shift to debt post-50

**Debt Mutual Funds**
• 7-9% returns
• Lower risk than equity
• Tax-efficient vs FDs

**National Savings Certificate (NSC)**
• 7.7% interest
• 5-year lock-in
• Tax benefit under 80C

**Retirement Investment Strategy by Age:**

**20s-30s:**
• **Equity**: 70-80%
• **Debt**: 20-30%
• Focus: Growth via SIP in equity funds
• Instruments: NPS (equity), ELSS, index funds

**30s-40s:**
• **Equity**: 60-70%
• **Debt**: 30-40%
• Increase SIP amounts with salary hikes
• Add NPS, increase EPF contribution (VPF)

**40s-50s:**
• **Equity**: 50-60%
• **Debt**: 40-50%
• Start building debt portfolio
• Continue NPS, PPF
• Review and rebalance annually

**50s-60s (Pre-retirement):**
• **Equity**: 30-40%
• **Debt**: 60-70%
• Shift to safer instruments
• SCSS, POMIS, G-Secs, debt funds
• Lock-in guaranteed income sources

**Post-Retirement (60+):**
• **Equity**: 20-30% (for inflation protection)
• **Debt**: 70-80% (for stability)
• Focus: Regular monthly income
• Instruments: Annuity, SCSS, FDs, debt funds with SWP

**Retirement Corpus Calculation:**

**Example:**
• Current age: 30 years
• Retirement age: 60 years
• Current monthly expenses: ₹40,000
• Inflation: 6%
• Life expectancy: 85 years
• Expected returns: 10% (pre-retirement), 7% (post-retirement)

**Step 1:** Future monthly expenses at 60
= ₹40,000 × (1.06)^30 = ₹2,30,000/month

**Step 2:** Corpus required at 60
= ₹2,30,000 × 12 × 25 years / (1 - 0.06) = ₹4-5 crores

**Step 3:** Monthly SIP required
= ₹15,000-20,000/month for 30 years @ 12% return

**Common Retirement Planning Mistakes:**

[ERROR] **Starting too late**
   (Compounding needs time)

[ERROR] **Underestimating inflation**
   (6% doubles costs every 12 years)

[ERROR] **Over-reliance on EPF only**
   (Diversification needed)

[ERROR] **No health insurance**
   (Medical costs biggest risk)

[ERROR] **Withdrawing EPF/PPF early**
   (Breaks compounding)

[ERROR] **Mixing insurance with investment**
   (ULIPs/endowment give poor returns)

[ERROR] **Ignoring spouse's retirement**
   (Both need separate corpus)

**Health Insurance in Retirement:**
• Critical for post-retirement
• Buy before 50 (pre-existing conditions)
• ₹10-25 lakhs coverage minimum
• Top-up/super top-up for major illnesses
• Factor premiums in retirement budget

**Tax Benefits:**

**Section 80C** (₹1.5 lakhs):
• EPF, PPF, ELSS, NSC, life insurance premium

**Section 80CCD(1B)** (₹50,000):
• Additional benefit ONLY for NPS

**Section 80D** (₹50,000):
• Health insurance for senior citizens

**Total max deduction**: ₹2.5 lakhs

**Retirement Income Sources:**

1. **EPF/NPS corpus** (lump sum + annuity)
2. **Annuity/pension** (monthly income)
3. **SWP from mutual funds** (systematic withdrawal)
4. **Rental income** (if real estate investment)
5. **Senior Citizen FD** (interest income)
6. **SCSS** (quarterly payout)
7. **Part-time work/consulting** (active income)

**Action Plan:**

[OK] **Step 1**: Calculate retirement corpus needed
[OK] **Step 2**: Maximize EPF/PPF contributions
[OK] **Step 3**: Open NPS (for 80CCD(1B) benefit)
[OK] **Step 4**: Start SIP in equity funds (₹5,000-10,000/month)
[OK] **Step 5**: Get health insurance (₹10L+ family floater)
[OK] **Step 6**: Increase contributions with salary hikes (50% rule)
[OK] **Step 7**: Review portfolio annually
[OK] **Step 8**: Rebalance towards debt post-50"""

        formatted = format_general_info_response(info, "Retirement Planner")

        return {
            "summary": formatted,
            "next_best_actions": [
                "Provide your age, current income, and planned retirement age for personalized retirement planning",
                "Ask specific questions about retirement schemes (EPF, NPS, PPF)",
                "Estimate your retirement corpus needs"
            ],
            "mode": "general_information"
        }
