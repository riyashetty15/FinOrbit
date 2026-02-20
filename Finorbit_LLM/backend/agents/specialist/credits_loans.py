# ==============================================
# File: src/agents/credits_loans_langgraph.py
# Description: LangGraph State-Based Credits & Loans Analyzer (XML schema aligned)
# ==============================================

from __future__ import annotations
from typing import Dict, Any, List, TypedDict, Optional
from dataclasses import dataclass
import os
import re
import xml.etree.ElementTree as ET
import logging

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

# ------------- Define Credits/Loans-specific State -------------

class LoanState(TypedDict, total=False):
    # Input data
    query: str
    profile: Dict[str, Any]
    transactions: List[Dict[str, Any]]

    # User info
    user_name: str
    age: int
    income: int
    cibil_score: int

    # Analysis results
    credit_spending: Dict[str, Any]
    criteria: Dict[str, Any]
    query_info: Dict[str, Any]
    loan_needs: List[str]
    risk_profile: str

    # Loan search
    loans: List[Dict[str, Any]]

    # Mode
    analysis_mode: str  # "loan_search" or "profile_analysis"

    # Final output
    summary: str
    next_best_actions: List[str]
    loan_analysis: Dict[str, Any]

    # Error
    error: Optional[str]


@dataclass
class LoanConfig:
    max_loans: int = 5
    default_age: int = 35
    default_income: int = 500_000
    default_cibil: int = 700
    xml_path: str = os.path.join("data", "all_loan_offers.xml")


class CreditsLoanAgent:
    """LangGraph-based Credits & Loans Analyzer (aligned to your XML schema)."""

    SUPPORTED_TYPES = ("personal", "home", "auto", "education", "gold")

    # ---------- lifecycle ----------

    def __init__(self, config: Optional[LoanConfig] = None):
        self.config = config or LoanConfig()
        self.loan_data: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.SUPPORTED_TYPES}
        self._load_loan_data()
        self.graph = self._build_graph()

    # ==================== DATA LOADING ====================

    def _load_loan_data(self) -> None:
        """
        Expected XML:
        <LoanOffers>
          <LoanOffer name="PNB Mortgage Loan">
            <Type>mortgage</Type>
            <InterestRate>12.94</InterestRate>
            <MinAmount>385694</MinAmount>
            <MaxAmount>1714911</MaxAmount>
            <MinTenure>17</MinTenure>
            <MaxTenure>184</MaxTenure>
            <ProcessingFee>3354</ProcessingFee>
            <Eligibility>...</Eligibility>
            <Features>...</Features>
          </LoanOffer>
          ...
        </LoanOffers>
        """
        xml_path = self.config.xml_path
        if not os.path.exists(xml_path):
            logger.warning("XML not found at %s — loan search will be empty.", xml_path)
            return

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # reset
            self.loan_data = {t: [] for t in self.SUPPORTED_TYPES}

            for offer in root.findall("LoanOffer"):
                parsed = self._parse_loans_xml(offer)
                if not parsed:
                    continue
                mapped = self._map_loan_type(parsed["loan_type"])
                if not mapped:
                    continue
                parsed["loan_type"] = mapped  # normalize
                self.loan_data[mapped].append(parsed)
        except Exception as e:
            logger.exception("Error loading XML: %s", e)

    def _map_loan_type(self, loan_type: str) -> Optional[str]:
        """Normalize loan types to canonical buckets."""
        lt = (loan_type or "").strip().lower()
        if lt in ("gold", "jewel", "jewell"):
            return "gold"
        if lt in ("home", "housing", "mortgage"):
            return "home"
        if lt in ("auto", "car", "vehicle"):
            return "auto"
        if lt in ("education", "student"):
            return "education"
        if lt in ("personal", "insta"):
            return "personal"
        # fallback: leave unknowns out of index
        return lt if lt in self.SUPPORTED_TYPES else None

    def _parse_num(self, text: Optional[str], f=float) -> float:
        if text is None:
            return 0.0
        try:
            s = re.sub(r"[^0-9.]+", "", str(text))
            if s == "":
                return 0.0
            return f(s)
        except Exception:
            return 0.0

    def _parse_loans_xml(self, offer_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse one <LoanOffer> node to dict aligned with our filters."""
        try:
            name = offer_elem.attrib.get("name", "Unknown Loan").strip()
            get = lambda tag, default="": (offer_elem.findtext(tag, default) or "").strip()

            loan_type = get("Type", "").lower()

            parsed = {
                "name": name,
                "loan_type": loan_type,  # will be normalized later
                "interest_rate": float(self._parse_num(get("InterestRate"), float)),
                "processing_fee": float(self._parse_num(get("ProcessingFee"), float)),
                "min_amount": float(self._parse_num(get("MinAmount"), float)),
                "max_amount": float(self._parse_num(get("MaxAmount"), float)),
                "min_tenure": int(self._parse_num(get("MinTenure"), float)),
                "max_tenure": int(self._parse_num(get("MaxTenure"), float)),
                "eligibility": get("Eligibility", ""),
                "features": get("Features", ""),
            }
            return parsed
        except Exception as e:
            print(f"[credits_loans] Bad offer node: {e}")
            return None

    # ==================== LANGGRAPH NODES ====================

    def _node_initialize_analysis(self, state: LoanState) -> LoanState:
        try:
            profile = state.get("profile", {}) or {}
            normalized = self._normalize_profile(profile)
            user_name = normalized["name"]
            age = normalized["age"]
            income = normalized["income"]
            cibil_score = normalized["cibil_score"]

            transactions = state.get("transactions", []) or []
            credit_spending = self._analyze_loan_transactions(transactions)

            query = (state.get("query", "") or "").strip()
            criteria = self._extract_criteria_from_query(query)
            query_info = self._parse_loan_query(query.lower())

            new_state = dict(state)
            new_state.update({
                "user_name": user_name,
                "age": age,
                "income": income,
                "cibil_score": cibil_score,
                "credit_spending": credit_spending,
                "criteria": criteria,
                "query_info": query_info,
                "error": None,
            })
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Initialization error: {str(e)}"
            return new_state

    def _node_determine_mode(self, state: LoanState) -> LoanState:
        try:
            criteria = state.get("criteria", {})
            numeric_keys = {"min_amount", "max_interest", "min_cibil", "max_tenure_months"}
            if criteria.get("loan_type") and any(k in criteria for k in numeric_keys):
                mode = "loan_search"
            else:
                mode = "profile_analysis"
            new_state = dict(state)
            new_state["analysis_mode"] = mode
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Mode determination error: {str(e)}"
            return new_state

    def _node_loan_search(self, state: LoanState) -> LoanState:
        try:
            criteria = state.get("criteria", {})
            loans = self._search_loans(criteria)

            if not loans:
                summary = f"No {criteria.get('loan_type', 'loan')} products found matching your criteria: {criteria}"
                actions = [
                    "Adjust your loan amount, interest, tenure, or CIBIL requirements",
                    "Consider different loan types (e.g., gold, personal, home)",
                    "Contact a loan advisor"
                ]
            else:
                lt = criteria.get('loan_type', 'loan')
                lines = [f"Found {len(loans)} {lt} loan products matching your criteria:\n"]
                for i, loan in enumerate(loans, 1):
                    lines.append(f"\n{i}. **{loan['name']}**")
                    lines.append(f"   - Interest: {loan.get('interest_rate', 0):g}%")
                    lines.append(f"   - Amount Range: ₹{int(loan.get('min_amount', 0)):,} – ₹{int(loan.get('max_amount', 0)):,}")
                    lines.append(f"   - Tenure Range: {loan.get('min_tenure', 0)} – {loan.get('max_tenure', 0)} months")
                    pf = loan.get("processing_fee", 0)
                    if pf:
                        lines.append(f"   - Processing Fee: ₹{int(pf):,}")
                    feats = (loan.get("features") or "").strip()
                    if feats:
                        lines.append(f"   - Features: {feats}")
                    elig = (loan.get("eligibility") or "").strip()
                    if elig:
                        lines.append(f"   - Eligibility: {elig}")
                lines.append(f"\nSearch criteria used: {criteria}")
                summary = "".join(lines)
                actions = [
                    "Compare total cost (interest + processing fee) across lenders",
                    "Check foreclosure/prepayment charges",
                    "Confirm documentation & eligibility before applying",
                    "Verify if special schemes (PMAY/subsidies) apply"
                ]

            new_state = dict(state)
            new_state.update({"loans": loans, "summary": summary, "next_best_actions": actions})
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Loan search error: {str(e)}"
            return new_state

    def _node_profile_analysis(self, state: LoanState) -> LoanState:
        try:
            age = state.get("age", self.config.default_age)
            income = state.get("income", self.config.default_income)
            cibil_score = state.get("cibil_score", self.config.default_cibil)
            user_name = state.get("user_name", "user")
            credit_spending = state.get("credit_spending", {})
            query_info = state.get("query_info", {})

            loan_needs = self._assess_loan_needs(age, income, cibil_score)
            risk_profile = self._get_risk_profile(age, income, cibil_score)
            recommendations = self._generate_recommendations(loan_needs, credit_spending, query_info, income, cibil_score)
            summary = self._create_summary(user_name, age, income, cibil_score, loan_needs, credit_spending, query_info)

            loan_analysis = {
                "current_spending": credit_spending,
                "recommended_types": loan_needs,
                "risk_profile": risk_profile
            }
            new_state = dict(state)
            new_state.update({
                "loan_needs": loan_needs,
                "risk_profile": risk_profile,
                "summary": summary,
                "next_best_actions": recommendations,
                "loan_analysis": loan_analysis
            })
            return new_state
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Profile analysis error: {str(e)}"
            return new_state

    # ==================== HELPERS ====================

    def _normalize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and sanitize profile fields."""
        name = (profile.get("name") or "user").strip() or "user"
        try:
            age = int(profile.get("age", self.config.default_age))
        except Exception:
            age = self.config.default_age
        try:
            income = int(profile.get("income", self.config.default_income))
        except Exception:
            income = self.config.default_income
        try:
            cibil_score = int(profile.get("cibil_score", self.config.default_cibil))
        except Exception:
            cibil_score = self.config.default_cibil

        age = max(0, age)
        income = max(0, income)
        cibil_score = max(0, cibil_score)

        return {"name": name, "age": age, "income": income, "cibil_score": cibil_score}

    def _analyze_loan_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        credit_spend = {"personal": 0, "home": 0, "auto": 0, "education": 0, "gold": 0, "other": 0, "total": 0}
        loan_keywords = {
            "personal": ["personal loan", "insta loan"],
            "home": ["home loan", "housing loan", "mortgage"],
            "auto": ["car loan", "auto loan", "vehicle loan"],
            "education": ["education loan", "student loan"],
            "gold": ["gold loan", "jewel loan", "jewell loan"],
            "other": ["loan", "emi"]
        }
        for txn in transactions or []:
            description = (txn.get("description", "") or "").lower()
            category = (txn.get("category", "") or "").lower()
            amount = float(txn.get("amount", 0) or 0)
            if "loan" in description or "emi" in description or "loan" in category:
                categorized = False
                for loan_type, keywords in loan_keywords.items():
                    if loan_type != "other" and any(k in description for k in keywords):
                        credit_spend[loan_type] += amount
                        categorized = True
                        break
                if not categorized:
                    credit_spend["other"] += amount
                credit_spend["total"] += amount
        return credit_spend

    _UNIT_SCALE = {
        "k": 1_000,
        "thousand": 1_000,
        "lakh": 100_000,
        "lac": 100_000,
        "crore": 10_000_000,
        "cr": 10_000_000,
        "m": 1_000_000,
        "million": 1_000_000,
    }

    def _extract_amount_candidates(self, text: str) -> List[int]:
        t = text.lower()
        candidates: List[int] = []

        # with units
        unit_pat = re.compile(r"(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac|crore|cr|m|million)")
        for m in unit_pat.finditer(t):
            val = float(m.group(1)); unit = m.group(2)
            candidates.append(int(val * self._UNIT_SCALE[unit]))

        # plain numbers with commas
        for m in re.finditer(r"\b(\d{1,3}(?:,\d{2}){1,}|\d{1,3}(?:,\d{3})+|\d+)\b", t):
            end = m.end()
            tail = t[end:end+10]
            if re.match(r"\s*(year|years|yr|yrs|month|months|mo)s?\b", tail):
                continue
            num = int(m.group(1).replace(",", ""))
            candidates.append(num)

        return candidates

    def _extract_tenure(self, text: str) -> Optional[int]:
        t = text.lower()
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(year|years|yr|yrs)\b", t)
        if m:
            years = float(m.group(1))
            return int(round(years * 12))
        m = re.search(r"\b(\d+)\s*(month|months|mo)\b", t)
        if m:
            return int(m.group(1))
        return None

    def _extract_interest(self, text: str) -> Optional[float]:
        t = text.lower()
        for pat in [
            r"under\s+(\d+(?:\.\d+)?)\s*%",
            r"below\s+(\d+(?:\.\d+)?)\s*%",
            r"max(?:imum)?\s+interest\s+(\d+(?:\.\d+)?)\s*%",
            r"interest\s+rate\s+below\s+(\d+(?:\.\d+)?)\s*%",
        ]:
            m = re.search(pat, t)
            if m:
                return float(m.group(1))
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", t)]
        return min(nums) if nums else None

    def _extract_cibil(self, text: str) -> Optional[int]:
        t = text.lower()
        m = re.search(r"cibil\s*(?:score)?\s*(?:above|over|greater\s+than|>=?)\s*(\d{3,4})", t)
        if m:
            return int(m.group(1))
        m = re.search(r"cibil\s*(?:is|=)\s*(\d{3,4})", t)
        if m:
            return int(m.group(1))
        return None

    def _extract_criteria_from_query(self, query: str) -> Dict[str, Any]:
        criteria: Dict[str, Any] = {}
        q = (query or "").lower()

        # type
        if any(term in q for term in ["gold", "jewel", "jewell"]):
            criteria["loan_type"] = "gold"
        elif any(term in q for term in ["home", "housing", "mortgage"]):
            criteria["loan_type"] = "home"
        elif any(term in q for term in ["auto", "car", "vehicle"]):
            criteria["loan_type"] = "auto"
        elif "education" in q or "student" in q:
            criteria["loan_type"] = "education"
        elif "personal" in q or "insta" in q:
            criteria["loan_type"] = "personal"
        elif "loan" in q:
            criteria["loan_type"] = "personal"

        # amount (take largest number user mentions as required minimum capacity)
        cands = self._extract_amount_candidates(q)
        if cands:
            criteria["min_amount"] = max(cands)

        # tenure
        tenure = self._extract_tenure(q)
        if tenure is not None:
            criteria["max_tenure_months"] = tenure

        # interest
        interest = self._extract_interest(q)
        if interest is not None:
            criteria["max_interest"] = interest

        # cibil (not present in XML; still capture for advice)
        cibil = self._extract_cibil(q)
        if cibil is not None:
            criteria["min_cibil"] = cibil

        return criteria

    def _parse_loan_query(self, query: str) -> Dict[str, Any]:
        info = {"type": None, "amount": None, "interest": None, "specific_request": False}
        q = (query or "").lower()
        if any(t in q for t in ["gold", "jewel", "jewell"]):
            info["type"] = "gold"; info["specific_request"] = True
        elif any(t in q for t in ["home", "housing", "mortgage"]):
            info["type"] = "home"; info["specific_request"] = True
        elif any(t in q for t in ["auto", "car", "vehicle"]):
            info["type"] = "auto"; info["specific_request"] = True
        elif "education" in q or "student" in q:
            info["type"] = "education"; info["specific_request"] = True
        elif "personal" in q or "insta" in q:
            info["type"] = "personal"; info["specific_request"] = True
        return info

    def _search_loans(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        loan_type = criteria.get("loan_type", "personal")
        offers = self.loan_data.get(loan_type, [])
        if not offers:
            return []

        filtered: List[Dict[str, Any]] = []
        for loan in offers:
            # Amount capacity: lender's max must cover user's required min
            if "min_amount" in criteria and float(loan.get("max_amount", 0)) < float(criteria["min_amount"]):
                continue

            # Interest ceiling
            if "max_interest" in criteria and float(loan.get("interest_rate", 10**9)) > float(criteria["max_interest"]):
                continue

            # Tenure: interpret user's number as desired tenure in months.
            # Keep loans whose tenure range includes that desired tenure.
            if "max_tenure_months" in criteria:
                desired = int(criteria["max_tenure_months"])
                if not (int(loan.get("min_tenure", 0)) <= desired <= int(loan.get("max_tenure", 10**9))):
                    continue

            # (Optional) CIBIL not in XML; we don't filter by it.

            filtered.append(loan)

        # Sort by interest asc, then max_amount desc
        filtered.sort(key=lambda x: (float(x.get("interest_rate", 1e9)), -float(x.get("max_amount", 0))))
        return filtered[: self.config.max_loans]

    def _assess_loan_needs(self, age: int, income: int, cibil_score: int) -> List[str]:
        needs: List[str] = []
        if income < 400_000:
            needs.append("education")
        if age >= 28 and income > 500_000:
            needs.append("home")
        if 21 <= age <= 60 and cibil_score > 700:
            needs.append("personal")
        if age > 24 and income > 300_000:
            needs.append("auto")
        # Gold loans are collateralized; broadly available for many profiles
        if cibil_score >= 600:
            needs.append("gold")
        return list(dict.fromkeys(needs))

    def _get_risk_profile(self, age: int, income: int, cibil_score: int) -> str:
        risk = 0
        if cibil_score < 650: risk += 3
        elif cibil_score < 700: risk += 2
        else: risk += 1
        if income < 300_000: risk += 2
        elif income < 700_000: risk += 1
        if age < 23: risk += 2
        elif age > 50: risk += 1
        if risk <= 2: return "Low Risk"
        if risk <= 4: return "Moderate Risk"
        return "High Risk"

    def _create_summary(self, name: str, age: int, income: int, cibil_score: int,
                        needs: List[str], spending: Dict[str, Any], query_info: Dict[str, Any]) -> str:
        total_spending = float(spending.get("total", 0) or 0)
        s = f"Hello {name}, "
        if query_info.get("specific_request"):
            s += f"I've analyzed your {query_info['type']} loan requirements. "
        else:
            s += f"I've analyzed your credit profile (age: {age}, income: ₹{income:,}, CIBIL: {cibil_score}). "
        if total_spending > 0:
            s += f"You're currently paying ~₹{int(total_spending):,} annually towards EMIs. "
        else:
            s += "You don't have significant ongoing loans. "
        if needs:
            s += f"Based on your profile, likely fits: {', '.join(needs[:3])}. "
        else:
            s += "No strong loan requirements detected. "
        if cibil_score < 700:
            s += "Consider improving your CIBIL score for better loan offers."
        else:
            s += "Your credit profile is healthy for most loan products."
        return s

    def _generate_recommendations(self, needs: List[str], current_spending: Dict[str, Any],
                                  query_info: Dict[str, Any], income: int, cibil_score: int) -> List[str]:
        recs: List[str] = []
        if cibil_score < 700:
            recs.append("Improve your CIBIL score towards 700+")
        if "home" in needs and float(current_spending.get("home", 0) or 0) < 100_000:
            recs.append("Evaluate home loan options (check subsidy eligibility like PMAY)")
        if "auto" in needs and float(current_spending.get("auto", 0) or 0) < 50_000:
            recs.append("Consider auto loans for vehicle purchase")
        if "personal" in needs and float(current_spending.get("personal", 0) or 0) < 30_000:
            recs.append("Personal loans can help with short-term needs")
        if "gold" in needs and float(current_spending.get("gold", 0) or 0) < 25_000:
            recs.append("Gold loans can unlock liquidity at competitive rates")
        recs.extend([
            "Compare interest + processing fee for true total cost",
            "Check foreclosure/prepayment charges before signing",
            "Maintain low credit utilization"
        ])
        return recs[:5]

    # ==================== CONDITIONAL EDGES ====================

    def _should_route_to_mode(self, state: LoanState) -> str:
        if state.get("error"):
            return "end"
        return state.get("analysis_mode", "profile_analysis")

    # ==================== GRAPH CONSTRUCTION ====================

    def _build_graph(self):
        graph = StateGraph(LoanState)
        graph.add_node("initialize_analysis", self._node_initialize_analysis)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("loan_search", self._node_loan_search)
        graph.add_node("profile_analysis", self._node_profile_analysis)

        graph.add_edge(START, "initialize_analysis")
        graph.add_edge("initialize_analysis", "determine_mode")
        graph.add_conditional_edges(
            "determine_mode",
            self._should_route_to_mode,
            {"loan_search": "loan_search", "profile_analysis": "profile_analysis", "end": END}
        )
        graph.add_edge("loan_search", END)
        graph.add_edge("profile_analysis", END)
        return graph.compile()

    # ==================== PUBLIC INTERFACE ====================

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run loan/credit analysis with intent-aware routing

        Flow:
        1. Check query intent (general vs personalized)
        2. For general queries: Provide educational loan/credit information
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

        # Handle general queries - provide loan/credit education without personalization
        if is_general_query(query, intent):
            return self._provide_general_loan_info(query)

        # For personalized queries, check profile completeness
        required_fields = ["income", "cibil_score"]
        missing_fields = ProfileValidator.get_missing_fields(profile, required_fields)

        if missing_fields:
            # Generate request for missing information
            message = ProfileValidator.generate_missing_fields_message(
                missing_fields,
                agent_name="Credits & Loans Advisor",
                query_context="recommend suitable loan options or analyze your creditworthiness"
            )
            return {
                "summary": message,
                "next_best_actions": [f"Provide your {field.replace('_', ' ')}" for field in missing_fields],
                "mode": "awaiting_profile",
                "requires_input": True,
                "missing_fields": missing_fields
            }

        # Profile is complete, proceed with personalized analysis using LangGraph
        try:
            loan_state: LoanState = {
                "query": query,
                "profile": profile,
                "transactions": state.get("transactions", []),
            }
            final_state = self.graph.invoke(loan_state)
            result: Dict[str, Any] = {
                "summary": final_state.get("summary", "Analysis completed"),
                "next_best_actions": final_state.get("next_best_actions", []),
                "mode": final_state.get("analysis_mode", "unknown"),
            }
            if final_state.get("analysis_mode") == "loan_search":
                result.update({
                    "loans": final_state.get("loans", []),
                    "criteria": final_state.get("criteria", {}),
                })
            else:
                result["loan_analysis"] = final_state.get("loan_analysis", {})
            return result
        except Exception as e:
            return {
                "summary": f"Loan analysis error: {str(e)}",
                "next_best_actions": ["Please try again with a different query"],
                "mode": "error",
            }

    def _provide_general_loan_info(self, query: str) -> Dict[str, Any]:
        """Provide general loan/credit information without personalization"""
        from backend.agents.agent_helpers import format_general_info_response

        query_lower = query.lower()

        # Determine what type of information to provide
        if any(word in query_lower for word in ["emi", "calculate emi", "equated monthly"]):
            info = """**EMI (Equated Monthly Installment): Complete Guide**

**What is EMI?**
EMI is a fixed payment amount made by a borrower to a lender at a specified date each month. It consists of both principal and interest components.

**EMI Formula:**
EMI = [P × r × (1+r)^n] / [(1+r)^n - 1]

Where:
• **P** = Principal loan amount
• **r** = Monthly interest rate (annual rate ÷ 12 ÷ 100)
• **n** = Loan tenure in months

**EMI Composition:**
• **Initial months**: Higher interest, lower principal
• **Later months**: Lower interest, higher principal
• This is because interest is calculated on the outstanding principal

**Example Calculation:**
Loan: ₹10 lakhs, Rate: 10% p.a., Tenure: 5 years (60 months)
• Monthly rate: 10 ÷ 12 ÷ 100 = 0.00833
• EMI = ₹21,247 approx
• Total payment = ₹12,74,820
• Total interest = ₹2,74,820

**Factors Affecting EMI:**
1. **Loan Amount**: Higher amount → Higher EMI
2. **Interest Rate**: Higher rate → Higher EMI
3. **Tenure**: Longer tenure → Lower EMI (but higher total interest)

**Tips to Reduce EMI:**
• Make a larger down payment (reduce principal)
• Negotiate for lower interest rate
• Extend loan tenure (increases total interest though)
• Consider part-prepayment to reduce principal
• Maintain good credit score for better rates"""

        elif any(word in query_lower for word in ["cibil", "credit score", "credit rating"]):
            info = """**CIBIL Score: Your Credit Health Report**

**What is CIBIL Score?**
CIBIL (Credit Information Bureau India Limited) score is a 3-digit number (300-900) that represents your creditworthiness based on your credit history.

**Score Ranges:**
• **300-549**: Poor (loan rejection likely)
• **550-649**: Fair (high interest rates, limited options)
• **650-749**: Good (moderate interest rates, decent approval chances)
• **750-900**: Excellent (best rates, high approval probability)

**What Affects Your CIBIL Score?**

**Positive Factors:**
• Timely payment of EMIs and credit card bills (35% weightage)
• Low credit utilization ratio (<30% of limit)
• Long credit history with good track record
• Mix of secured and unsecured loans
• Few credit inquiries

**Negative Factors:**
• Late payments or defaults (biggest impact)
• High credit utilization (>50%)
• Multiple loan applications in short time
• Settling loans (marked as "settled" not "closed")
• Frequent loan rejections

**How to Check CIBIL Score:**
• Visit www.cibil.com
• One free report per year
• Paid plans for unlimited access
• Some banks offer free score in net banking

**How to Improve CIBIL Score:**
1. **Pay on time**: Set auto-debit for EMIs and credit cards
2. **Keep utilization low**: Use <30% of credit limit
3. **Don't close old cards**: Credit history length matters
4. **Mix of credit**: Have both secured (home loan) and unsecured (personal loan)
5. **Check report regularly**: Dispute errors immediately
6. **Avoid multiple applications**: Each inquiry reduces score slightly

**Time to Improve:**
• Minor improvements: 3-6 months
• Significant improvements: 6-12 months
• Recovery from default: 2-3 years

**Impact on Loans:**
• Score >750: Interest rate 1-2% lower than average
• Score <650: Interest rate 2-4% higher, or loan rejection"""

        elif any(word in query_lower for word in ["personal loan", "unsecured loan"]):
            info = """**Personal Loans: Quick Funding Solution**

**What is a Personal Loan?**
An unsecured loan provided for various purposes without requiring collateral. Based purely on creditworthiness and income.

**Key Features:**
• **Loan Amount**: ₹50,000 to ₹40 lakhs (varies by lender)
• **Interest Rate**: 10-24% p.a. (depends on credit score)
• **Tenure**: 1-5 years
• **Processing Fee**: 1-3% of loan amount
• **No Collateral Required**
• **Quick Disbursal**: 24-48 hours

**Eligibility Criteria:**
• Age: 21-60 years
• Income: Minimum ₹15,000-25,000/month (varies by bank)
• CIBIL Score: >650 (preferably >750)
• Employment: Salaried or self-employed with stable income
• Work Experience: Minimum 1-2 years

**Documents Required:**
• Identity proof (Aadhaar, PAN, Passport)
• Address proof (utility bills, rent agreement)
• Income proof (salary slips, ITR, bank statements)
• Employment proof (offer letter, appointment letter)

**Common Uses:**
• Medical emergencies
• Wedding expenses
• Home renovation
• Debt consolidation
• Travel/vacation
• Education fees

**Pros:**
• No collateral needed
• Fast approval and disbursal
• Flexible usage
• Fixed interest rate (most cases)

**Cons:**
• Higher interest compared to secured loans
• Processing fees and prepayment charges
• Affects CIBIL if not managed properly
• Strict eligibility criteria

**Tips for Best Personal Loan:**
• Maintain CIBIL score >750 for lower rates
• Compare offers from multiple lenders
• Check hidden charges (processing, prepayment, late payment)
• Borrow only what you need
• Choose tenure wisely (shorter = less interest)"""

        elif any(word in query_lower for word in ["home loan", "housing loan", "mortgage"]):
            info = """**Home Loans: Financing Your Dream Home**

**What is a Home Loan?**
A secured loan provided for purchasing, constructing, or renovating residential property. The property serves as collateral.

**Types of Home Loans:**
1. **Home Purchase Loan**: Buy ready-to-move or under-construction property
2. **Home Construction Loan**: Build house on owned land
3. **Home Improvement Loan**: Renovate/repair existing property
4. **Land Purchase Loan**: Buy plot for future construction
5. **Balance Transfer**: Transfer loan to another bank for better rates
6. **Top-up Loan**: Additional loan on existing home loan

**Key Features:**
• **Loan Amount**: Up to 90% of property value (₹75 lakhs for affordable housing)
• **Interest Rate**: 8.5-10.5% p.a. (varies by bank and credit profile)
• **Tenure**: Up to 30 years
• **Processing Fee**: 0.25-1% of loan amount
• **Tax Benefits**: Section 80C (principal) and 24(b) (interest)

**Eligibility Criteria:**
• Age: 18-70 years (loan should end before 65-70)
• Income: Minimum ₹25,000/month (varies by city and bank)
• CIBIL Score: >650 (preferably >750)
• Employment: Stable salaried/self-employed
• LTV Ratio: 75-90% (Loan-to-Value)

**Documents Required:**
• KYC documents (Aadhaar, PAN, passport)
• Income proof (salary slips, ITR, Form 16)
• Bank statements (6-12 months)
• Property documents (sale deed, approved plan)
• Employment proof

**Interest Rate Types:**
• **Fixed Rate**: Same rate throughout tenure (rare, higher rate)
• **Floating Rate**: Rate changes with market (most common)
• **Hybrid**: Fixed for initial years, then floating

**Tax Benefits (FY 2024-25):**
• **Section 80C**: Principal repayment up to ₹1.5 lakhs
• **Section 24(b)**: Interest up to ₹2 lakhs (self-occupied)
• **First-time buyers**: Additional ₹50,000 under Section 80EEA

**EMI Calculation:**
For ₹30 lakhs at 9% for 20 years:
• EMI ≈ ₹27,000/month
• Total payment ≈ ₹64.8 lakhs
• Total interest ≈ ₹34.8 lakhs

**Tips:**
• Save for 20-25% down payment
• Compare rates from multiple banks
• Consider prepayment options
• Check foreclosure and part-payment charges
• Maintain CIBIL >750 for best rates
• Opt for shorter tenure if affordable (saves interest)"""

        elif any(word in query_lower for word in ["car loan", "auto loan", "vehicle loan"]):
            info = """**Auto/Car Loans: Drive Your Dream Car**

**What is an Auto Loan?**
A secured loan for purchasing new or used cars, two-wheelers, or commercial vehicles. The vehicle serves as collateral.

**Key Features:**
• **Loan Amount**: 80-90% of on-road price (10-20% down payment)
• **Interest Rate**:
  - New cars: 8.5-12% p.a.
  - Used cars: 13-16% p.a.
• **Tenure**: 1-7 years
• **Processing Fee**: ₹3,000-10,000 or 2-3% of loan amount

**Eligibility:**
• Age: 21-65 years
• Income: Minimum ₹3-5 lakhs/year (varies by car price)
• CIBIL Score: >650 (preferably >700)
• Employment: Salaried/self-employed with stable income

**Documents Required:**
• Identity and address proof
• Income proof (salary slips, ITR)
• Bank statements
• Quotation/proforma invoice from dealer
• RC copy (for used cars)

**New Car vs Used Car Loan:**

**New Car:**
• Lower interest rates (8.5-11%)
• Higher loan amount (up to 90%)
• Longer tenure (up to 7 years)
• Easier approval

**Used Car:**
• Higher interest (13-16%)
• Lower loan amount (70-80%)
• Shorter tenure (up to 5 years)
• Age of car matters (usually <5 years)

**Types:**
1. **Simple Car Loan**: Standard financing
2. **Balloon Payment**: Lower EMIs, large final payment
3. **Dealer Financing**: Arranged through dealership
4. **Manufacturer Financing**: Lower rates during offers

**EMI Example:**
₹8 lakhs car, 85% loan (₹6.8 lakhs), 10% interest, 5 years:
• Down payment: ₹1.2 lakhs
• EMI ≈ ₹14,500/month
• Total payment ≈ ₹8.7 lakhs
• Total interest ≈ ₹1.9 lakhs

**Tips:**
• Make higher down payment (20-30%) to reduce EMI
• Check end-to-end cost (processing, insurance, registration)
• Compare bank loans vs dealer financing
• Avoid add-ons unless necessary
• Consider resale value before long tenure
• Insurance should be comprehensive, not third-party only"""

        else:
            # General loan overview
            info = """**Loans & Credits: Complete Overview**

**Types of Loans in India:**

**Secured Loans** (Backed by collateral, lower interest):
1. **Home Loan**: 8.5-10.5% p.a., up to 30 years
2. **Auto Loan**: 8.5-12% p.a., up to 7 years
3. **Gold Loan**: 7-12% p.a., short-term
4. **Loan Against Property**: 9-12% p.a., up to 15 years
5. **Loan Against FD/Securities**: 1-2% above FD rate

**Unsecured Loans** (No collateral, higher interest):
1. **Personal Loan**: 10-24% p.a., 1-5 years
2. **Credit Card**: 36-48% p.a. on outstanding
3. **Education Loan**: 8-14% p.a., up to 15 years
4. **Business Loan**: 12-24% p.a., varies

**Key Loan Terms:**

**Principal**: Original loan amount
**Interest**: Cost of borrowing
**EMI**: Fixed monthly payment
**Tenure**: Loan duration
**Processing Fee**: Upfront charges (1-3%)
**Foreclosure**: Closing loan before tenure ends
**Prepayment**: Partial early payment
**APR**: Actual cost including all fees

**Factors Affecting Loan Approval:**

1. **Credit Score (CIBIL)**: Most critical factor
   - >750: Excellent (best rates)
   - 650-750: Good (moderate rates)
   - <650: Poor (rejection likely)

2. **Income Stability**:
   - Salaried: Work experience, salary continuity
   - Self-employed: Business vintage, ITR

3. **Debt-to-Income Ratio**:
   - Total EMIs shouldn't exceed 50-60% of income
   - Lower ratio = better approval chances

4. **Employment History**:
   - Minimum 1-2 years in current job
   - Frequent job changes = red flag

5. **Age**:
   - 21-55 years ideal
   - Loan should mature before retirement age

**Smart Borrowing Tips:**

**Before Taking Loan:**
• Assess actual need vs want
• Calculate affordable EMI (max 40-50% of income)
• Compare offers from 3-4 lenders
• Read fine print (hidden charges, penalties)
• Maintain CIBIL >750

**During Loan:**
• Set auto-debit for EMI (avoid late payment)
• Consider prepayment if you get windfall
• Keep loan documents safe
• Monitor CIBIL score regularly

**Red Flags to Avoid:**
• Too-good-to-be-true rates
• Upfront fee demands
• Unregistered lenders
• Loans without documentation
• Pressure to borrow more

**Debt Management:**
• Emergency fund before loan
• Avoid multiple loans simultaneously
• Prioritize high-interest debt repayment
• Don't borrow for depreciating assets (unless necessary)
• Consider debt consolidation if overwhelmed"""

        formatted = format_general_info_response(info, "Credits & Loans Advisor")

        return {
            "summary": formatted,
            "next_best_actions": [
                "Provide your income and CIBIL score for personalized loan recommendations",
                "Ask specific questions about loan types or eligibility",
                "Compare loan offers from different banks"
            ],
            "mode": "general_information"
        }


# ------------- Main class for orchestrator compatibility -------------

class CreditsLoansAnalyzer:
    """LangGraph-based Credits & Loans Analyzer - Compatible with existing orchestrator"""

    def __init__(self):
        self.analyzer_graph = CreditsLoanAgent()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run credits/loans analysis using LangGraph"""
        return self.analyzer_graph.run(state)
