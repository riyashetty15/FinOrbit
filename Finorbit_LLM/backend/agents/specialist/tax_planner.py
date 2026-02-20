# ==============================================
# File: src/agents/tax_planner_langgraph.py
# Description: LangGraph State-Based Tax Planner
# ==============================================

from __future__ import annotations
from typing import Dict, Any, List, TypedDict, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import os
import re
import logging
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

# Define Tax-specific State
class TaxState(TypedDict, total=False):
    # Input data
    query: str
    profile: Dict[str, Any]
    transactions: List[Dict[str, Any]]

    # User information
    user_name: str
    age: int
    income: int
    occupation: str

    # Analysis results
    income_sources: Dict[str, Any]
    tax_components: Dict[str, Any]
    criteria: Dict[str, Any]
    query_info: Dict[str, Any]
    tax_regime: str  # "old" or "new"

    # Tax calculation results
    tax_breakdown: Dict[str, Any]
    deductions: List[Dict[str, Any]]
    investments: List[Dict[str, Any]]

    # Mode selection
    analysis_mode: str  # "tax_calculation" or "tax_planning"

    # Final output
    summary: str
    next_best_actions: List[str]
    tax_analysis: Dict[str, Any]

    # Error handling
    error: Optional[str]

@dataclass
class TaxConfig:
    max_investment_options: int = 5
    default_age: int = 35
    default_income: int = 500000
    default_occupation: str = "salaried"

class TaxPlannerGraph:
    """LangGraph-based Tax Planner with State Management"""

    def __init__(self, config: Optional[TaxConfig] = None):
        self.config = config or TaxConfig()

        # Initialize tax data and analysis parameters
        self.tax_data = {
            "deductions": [],
            "exemptions": [],
            "rebates": [],
            "investment_options": []
        }
        self._load_tax_data()

        # Tax regime parameters
        self.tax_slabs_old = {
            "salaried": [
                {"min": 0, "max": 250000, "rate": 0},
                {"min": 250001, "max": 500000, "rate": 5},
                {"min": 500001, "max": 1000000, "rate": 20},
                {"min": 1000001, "max": float("inf"), "rate": 30}
            ],
            "senior_citizen": [
                {"min": 0, "max": 300000, "rate": 0},
                {"min": 300001, "max": 500000, "rate": 5},
                {"min": 500001, "max": 1000000, "rate": 20},
                {"min": 1000001, "max": float("inf"), "rate": 30}
            ],
            "super_senior": [
                {"min": 0, "max": 500000, "rate": 0},
                {"min": 500001, "max": 1000000, "rate": 20},
                {"min": 1000001, "max": float("inf"), "rate": 30}
            ]
        }

        self.tax_slabs_new = {
            "all": [
                {"min": 0, "max": 250000, "rate": 0},
                {"min": 250001, "max": 500000, "rate": 5},
                {"min": 500001, "max": 750000, "rate": 10},
                {"min": 750001, "max": 1000000, "rate": 15},
                {"min": 1000001, "max": 1250000, "rate": 20},
                {"min": 1250001, "max": 1500000, "rate": 25},
                {"min": 1500001, "max": float("inf"), "rate": 30}
            ]
        }

        # Standard deduction amounts
        self.standard_deduction = {
            "salaried": 50000,
            "pensioner": 50000
        }

        # Build the graph
        self.graph = self._build_graph()

    def _load_tax_data(self):
        """Load tax data from XML files"""
        possible_paths = ["data/"]
        files = {
            "deductions": "taxDeductions.xml",
            "exemptions": "taxExemptions.xml",
            "rebates": "taxRebates.xml",
            "investment_options": "taxInvestments.xml"
        }

        for data_type, filename in files.items():
            loaded = False
            for base_path in possible_paths:
                file_path = os.path.join(base_path, filename)
                try:
                    if os.path.exists(file_path):
                        tree = ET.parse(file_path)
                        if data_type == "deductions":
                            self.tax_data[data_type] = self._parse_deductions_xml(tree.getroot())
                        elif data_type == "exemptions":
                            self.tax_data[data_type] = self._parse_exemptions_xml(tree.getroot())
                        elif data_type == "rebates":
                            self.tax_data[data_type] = self._parse_rebates_xml(tree.getroot())
                        elif data_type == "investment_options":
                            self.tax_data[data_type] = self._parse_investments_xml(tree.getroot())
                        logger.info("Loaded %s %s", len(self.tax_data[data_type]), data_type)
                        loaded = True
                        break
                except Exception:
                    continue
            if not loaded:
                logger.warning("No %s data found (expected %s)", data_type, filename)

    def _parse_deductions_xml(self, root):
        """Parse tax deductions XML data"""
        deductions = []
        for deduction in root.findall("Deduction"):
            d = {
                "section": deduction.get("section", ""),
                "name": deduction.get("name", ""),
                "max_amount": int(self._get_element_text(deduction, "MaxAmount", "0")),
                "description": self._get_element_text(deduction, "Description", ""),
                "applicable_for": self._get_element_text(deduction, "ApplicableFor", "all").split(","),
                "conditions": self._get_element_text(deduction, "Conditions", "")
            }
            deductions.append(d)
        return deductions

    def _parse_exemptions_xml(self, root):
        """Parse tax exemptions XML data"""
        exemptions = []
        for exemption in root.findall("Exemption"):
            e = {
                "section": exemption.get("section", ""),
                "name": exemption.get("name", ""),
                "description": self._get_element_text(exemption, "Description", ""),
                "conditions": self._get_element_text(exemption, "Conditions", "")
            }
            exemptions.append(e)
        return exemptions

    def _parse_rebates_xml(self, root):
        """Parse tax rebates XML data"""
        rebates = []
        for rebate in root.findall("Rebate"):
            r = {
                "section": rebate.get("section", ""),
                "name": rebate.get("name", ""),
                "max_amount": int(self._get_element_text(rebate, "MaxAmount", "0")),
                "income_limit": int(self._get_element_text(rebate, "IncomeLimit", "0")),
                "description": self._get_element_text(rebate, "Description", ""),
                "conditions": self._get_element_text(rebate, "Conditions", "")
            }
            rebates.append(r)
        return rebates

    def _parse_investments_xml(self, root):
        """Parse tax-saving investment options XML data"""
        investments = []
        for investment in root.findall("Investment"):
            i = {
                "name": investment.get("name", ""),
                "section": investment.get("section", ""),
                "type": self._get_element_text(investment, "Type", ""),
                "min_amount": int(self._get_element_text(investment, "MinAmount", "0")),
                "max_amount": int(self._get_element_text(investment, "MaxAmount", "0")),
                "lock_in_period": self._get_element_text(investment, "LockInPeriod", ""),
                "risk_level": self._get_element_text(investment, "RiskLevel", ""),
                "returns": self._get_element_text(investment, "Returns", ""),
                "description": self._get_element_text(investment, "Description", "")
            }
            investments.append(i)
        return investments

    def _get_element_text(self, parent, tag_name, default=""):
        """Safely get text from XML element"""
        element = parent.find(tag_name)
        return element.text if element is not None and element.text is not None else default

    # ==================== LANGGRAPH NODES ====================
    def _node_initialize_analysis(self, state: TaxState) -> TaxState:
        """Initialize tax analysis - extract user data and query info"""
        try:
            # Extract user information
            profile = state.get("profile", {}) or {}
            normalized = self._normalize_profile(profile)
            user_name = normalized["name"]
            age = normalized["age"]
            income = normalized["income"]
            occupation = normalized["occupation"]

            # Analyze income sources from transactions
            transactions = state.get("transactions", [])
            income_sources = self._analyze_income_transactions(transactions)

            # Extract criteria from query
            query = state.get("query", "")
            criteria = self._extract_criteria_from_query(query)

            # Parse query for tax-related information
            query_info = self._parse_tax_query(query.lower())

            # Update state
            new_state = dict(state)
            new_state.update({
                "user_name": user_name,
                "age": age,
                "income": income,
                "occupation": occupation,
                "income_sources": income_sources,
                "criteria": criteria,
                "query_info": query_info,
                "error": None
            })

            return new_state

        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Initialization error: {str(e)}"
            return new_state

    def _node_determine_mode(self, state: TaxState) -> TaxState:
        """Determine analysis mode: tax calculation vs tax planning"""
        try:
            criteria = state.get("criteria", {})
            query_info = state.get("query_info", {})

            # Tax Calculation Mode: Specific queries with income/regime details
            if criteria.get('income') or criteria.get('tax_regime') or query_info.get("calculation_request"):
                analysis_mode = "tax_calculation"
            else:
                # Tax Planning Mode: General tax planning advice
                analysis_mode = "tax_planning"

            new_state = dict(state)
            new_state["analysis_mode"] = analysis_mode
            return new_state

        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Mode determination error: {str(e)}"
            return new_state

    def _node_tax_calculation(self, state: TaxState) -> TaxState:
        """Calculate tax based on income and selected regime"""
        try:
            # Extract state variables
            income = state.get("income", self.config.default_income)
            occupation = state.get("occupation", self.config.default_occupation)
            criteria = state.get("criteria", {})
            user_name = state.get("user_name", "user")

            # Determine tax regime
            tax_regime = criteria.get('tax_regime', 'old')  # Default to old regime

            # Calculate tax based on regime
            tax_breakdown = self._calculate_tax(income, occupation, tax_regime)

            # Get applicable deductions and exemptions
            deductions = self._get_applicable_deductions(income, occupation, tax_regime)

            # Create summary
            summary = self._create_tax_calculation_summary(
                user_name, income, tax_regime, tax_breakdown, deductions
            )

            # Generate recommendations
            next_best_actions = self._generate_tax_calculation_recommendations(
                tax_breakdown, deductions, tax_regime
            )

            # Create tax analysis object
            tax_analysis = {
                "income": income,
                "tax_regime": tax_regime,
                "tax_breakdown": tax_breakdown,
                "applicable_deductions": deductions
            }

            new_state = dict(state)
            new_state.update({
                "tax_regime": tax_regime,
                "tax_breakdown": tax_breakdown,
                "deductions": deductions,
                "summary": summary,
                "next_best_actions": next_best_actions,
                "tax_analysis": tax_analysis
            })

            return new_state

        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Tax calculation error: {str(e)}"
            return new_state

    def _node_tax_planning(self, state: TaxState) -> TaxState:
        """Provide tax planning recommendations based on user profile"""
        try:
            # Extract state variables
            income = state.get("income", self.config.default_income)
            occupation = state.get("occupation", self.config.default_occupation)
            age = state.get("age", self.config.default_age)
            user_name = state.get("user_name", "user")
            query_info = state.get("query_info", {})

            # Determine optimal tax regime
            old_regime_tax = self._calculate_tax(income, occupation, 'old')
            new_regime_tax = self._calculate_tax(income, occupation, 'new')

            # Compare regimes
            if old_regime_tax["total_tax"] < new_regime_tax["total_tax"]:
                recommended_regime = "old"
                tax_savings = new_regime_tax["total_tax"] - old_regime_tax["total_tax"]
            else:
                recommended_regime = "new"
                tax_savings = old_regime_tax["total_tax"] - new_regime_tax["total_tax"]

            # Get applicable deductions for old regime
            deductions = self._get_applicable_deductions(income, occupation, 'old')

            # Get investment recommendations
            investments = self._get_investment_recommendations(income, age, occupation)

            # Create summary
            summary = self._create_tax_planning_summary(
                user_name, income, recommended_regime, tax_savings, deductions, investments
            )

            # Generate recommendations
            next_best_actions = self._generate_tax_planning_recommendations(
                income, recommended_regime, deductions, investments
            )

            # Create tax analysis object
            tax_analysis = {
                "income": income,
                "recommended_regime": recommended_regime,
                "potential_tax_savings": tax_savings,
                "applicable_deductions": deductions,
                "recommended_investments": investments,
                "old_regime_tax": old_regime_tax["total_tax"],
                "new_regime_tax": new_regime_tax["total_tax"]
            }

            new_state = dict(state)
            new_state.update({
                "tax_regime": recommended_regime,
                "deductions": deductions,
                "investments": investments,
                "summary": summary,
                "next_best_actions": next_best_actions,
                "tax_analysis": tax_analysis
            })

            return new_state

        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Tax planning error: {str(e)}"
            return new_state

    # ==================== HELPER METHODS ====================

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

        occupation = (profile.get("occupation") or self.config.default_occupation).strip().lower()
        if occupation not in {"salaried", "business", "senior_citizen", "super_senior", "pensioner"}:
            occupation = self.config.default_occupation

        age = max(0, age)
        income = max(0, income)

        return {"name": name, "age": age, "income": income, "occupation": occupation}
    def _analyze_income_transactions(self, transactions):
        """Analyze transactions to identify income sources"""
        income_sources = {
            "salary": 0,
            "business": 0,
            "rental": 0,
            "interest": 0,
            "capital_gains": 0,
            "other": 0,
            "total": 0
        }

        income_keywords = {
            "salary": ["salary", "income", "payroll"],
            "business": ["business", "freelance", "consulting"],
            "rental": ["rent", "lease"],
            "interest": ["interest", "dividend"],
            "capital_gains": ["capital gain", "stocks", "securities"]
        }

        for txn in transactions:
            description = txn.get("description", "").lower()
            category = txn.get("category", "").lower()
            amount = txn.get("amount", 0)

            if amount > 0:  # Income transactions have positive amounts
                categorized = False
                for income_type, keywords in income_keywords.items():
                    if any(keyword in description for keyword in keywords):
                        income_sources[income_type] += amount
                        categorized = True
                        break

                if not categorized:
                    income_sources["other"] += amount

                income_sources["total"] += amount

        return income_sources

    def _extract_criteria_from_query(self, query):
        """Extract tax criteria from natural language query"""
        criteria = {}
        query_lower = query.lower()

        # Extract income amount
        income_patterns = [
            r'income\s+(?:of\s+)?(?:rs\.?\s*)?(\d+(?:,\d+)*)\s*(?:lakh|lakhs?|crore|crores?)?',
            r'earning\s+(?:rs\.?\s*)?(\d+(?:,\d+)*)\s*(?:lakh|lakhs?|crore|crores?)?',
            r'salary\s+(?:of\s+)?(?:rs\.?\s*)?(\d+(?:,\d+)*)\s*(?:lakh|lakhs?|crore|crores?)?'
        ]

        for pattern in income_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = int(amount_str)
                if 'lakh' in query_lower:
                    amount *= 100000
                elif 'crore' in query_lower:
                    amount *= 10000000
                criteria['income'] = amount
                break

        # Extract tax regime preference
        if 'old regime' in query_lower or 'old tax regime' in query_lower:
            criteria['tax_regime'] = 'old'
        elif 'new regime' in query_lower or 'new tax regime' in query_lower:
            criteria['tax_regime'] = 'new'

        # Extract occupation
        if any(term in query_lower for term in ['salaried', 'employee', 'job']):
            criteria['occupation'] = 'salaried'
        elif any(term in query_lower for term in ['business', 'self-employed', 'freelancer']):
            criteria['occupation'] = 'business'
        elif any(term in query_lower for term in ['senior citizen', 'retired', 'pension']):
            criteria['occupation'] = 'senior_citizen'

        return criteria

    def _parse_tax_query(self, query):
        """Parse user query for specific tax requests"""
        query_info = {
            "type": None,
            "calculation_request": False,
            "planning_request": False,
            "specific_section": None
        }

        # Check for calculation requests
        if any(term in query for term in ['calculate', 'compute', 'how much tax']):
            query_info["calculation_request"] = True

        # Check for planning requests
        if any(term in query for term in ['plan', 'save tax', 'reduce tax', 'investment']):
            query_info["planning_request"] = True

        # Check for specific section queries
        section_match = re.search(r'section\s+(\d+)([a-z]*)', query.lower())
        if section_match:
            query_info["specific_section"] = f"{section_match.group(1)}{section_match.group(2)}"

        return query_info

    def _calculate_tax(self, income, occupation, regime):
        """Calculate tax based on income, occupation, and regime"""
        # Determine applicable tax slabs
        if regime == 'old':
            if occupation == 'senior_citizen':
                slabs = self.tax_slabs_old["senior_citizen"]
            elif occupation == 'super_senior':
                slabs = self.tax_slabs_old["super_senior"]
            else:
                slabs = self.tax_slabs_old["salaried"]
        else:
            slabs = self.tax_slabs_new["all"]

        # Apply standard deduction for salaried/pensioners in old regime
        taxable_income = income
        if regime == 'old' and occupation in ['salaried', 'pensioner']:
            standard_deduction = self.standard_deduction.get(occupation, 0)
            taxable_income = max(0, income - standard_deduction)

        # Calculate tax
        tax = 0
        for slab in slabs:
            if taxable_income > slab["min"]:
                taxable_in_slab = min(taxable_income, slab["max"]) - slab["min"]
                tax += taxable_in_slab * slab["rate"] / 100

        # Apply cess (4%)
        cess = tax * 0.04
        total_tax = tax + cess

        return {
            "regime": regime,
            "gross_income": income,
            "taxable_income": taxable_income,
            "tax_before_cess": tax,
            "cess": cess,
            "total_tax": total_tax,
            "effective_tax_rate": (total_tax / income * 100) if income > 0 else 0
        }

    def _get_applicable_deductions(self, income, occupation, regime):
        """Get applicable tax deductions based on profile and regime"""
        applicable_deductions = []

        # In new regime, most deductions are not available
        if regime == 'new':
            # Only specific deductions are available in new regime
            for deduction in self.tax_data["deductions"]:
                if deduction["section"] in ["80CCD(2)", "80CCD(1B)"]:  # NPS contributions
                    applicable_deductions.append(deduction)
            return applicable_deductions

        # For old regime, check all deductions
        for deduction in self.tax_data["deductions"]:
            # Check if deduction is applicable for the occupation
            if "all" in deduction["applicable_for"] or occupation in deduction["applicable_for"]:
                # Check income conditions if any
                if not deduction["conditions"] or self._check_deduction_conditions(deduction["conditions"], income):
                    applicable_deductions.append(deduction)

        return applicable_deductions

    def _check_deduction_conditions(self, conditions, income):
        """Check if conditions for a deduction are met"""
        # Simple condition checking - can be expanded
        if not conditions:
            return True

        # Check for income limit conditions
        income_limit_match = re.search(r'income\s*<?\s*(\d+)', conditions.lower())
        if income_limit_match:
            limit = int(income_limit_match.group(1))
            if "less than" in conditions.lower():
                return income < limit
            else:
                return income <= limit

        return True

    def _get_investment_recommendations(self, income, age, occupation):
        """Get tax-saving investment recommendations based on profile"""
        investments = []

        # Filter investments based on profile
        for investment in self.tax_data["investment_options"]:
            # Age-based filtering
            if age < 30 and investment["type"] == "pension":
                continue  # Skip pension funds for young individuals

            # Occupation-based filtering
            if occupation == "senior_citizen" and investment["risk_level"] == "high":
                continue  # Skip high-risk investments for seniors

            investments.append(investment)

        # Sort by returns (simplified - would need more sophisticated sorting in reality)
        investments.sort(key=lambda x: float(x["returns"].replace("%", "")), reverse=True)

        return investments[:self.config.max_investment_options]

    def _create_tax_calculation_summary(self, name, income, regime, tax_breakdown, deductions):
        """Create summary for tax calculation"""
        summary = f"Hello {name}, "
        summary += f"I've calculated your tax liability under the {regime} tax regime. "
        summary += f"Your gross income is ₹{income:,}, "
        summary += f"and your estimated tax liability is ₹{tax_breakdown['total_tax']:,.2f}. "
        summary += f"The effective tax rate is {tax_breakdown['effective_tax_rate']:.2f}%. "

        if regime == 'old' and deductions:
            summary += f"You can claim deductions under sections like: "
            summary += f"{', '.join([d['section'] for d in deductions[:3]])}. "

        summary += f"This includes a cess of ₹{tax_breakdown['cess']:,.2f}."

        return summary

    def _create_tax_planning_summary(self, name, income, recommended_regime, tax_savings, deductions, investments):
        """Create summary for tax planning"""
        summary = f"Hello {name}, "
        summary += f"I've analyzed your tax planning options for an income of ₹{income:,}. "
        summary += f"The {recommended_regime} tax regime is more beneficial for you, "
        summary += f"potentially saving you ₹{tax_savings:,.2f} in taxes. "

        if recommended_regime == 'old' and deductions:
            summary += f"You can maximize deductions under sections like: "
            summary += f"{', '.join([d['section'] for d in deductions[:3]])}. "

        if investments:
            summary += f"Consider these tax-saving investments: "
            summary += f"{', '.join([i['name'] for i in investments[:3]])}. "

        summary += f"Proper tax planning can help you save significantly while building wealth."

        return summary

    def _generate_tax_calculation_recommendations(self, tax_breakdown, deductions, regime):
        """Generate recommendations for tax calculation"""
        recommendations = []

        # Regime comparison recommendation
        if regime == 'old':
            recommendations.append("Compare with new tax regime to ensure you're using the most beneficial option")
        else:
            recommendations.append("Check if old regime with deductions could be more beneficial")

        # Deduction recommendations
        if regime == 'old' and deductions:
            recommendations.append("Ensure you have all necessary documentation for claimed deductions")
            recommendations.append("Consider timing of expenses to maximize deduction benefits")

        # Advance tax recommendations
        if tax_breakdown["total_tax"] > 10000:
            recommendations.append("Pay advance tax in installments to avoid interest penalties")

        # General recommendations
        recommendations.extend([
            "Keep all tax-related documents organized",
            "Consider consulting a tax professional for complex situations"
        ])

        return recommendations[:5]

    def _generate_tax_planning_recommendations(self, income, regime, deductions, investments):
        """Generate recommendations for tax planning"""
        recommendations = []

        # Regime-specific recommendations
        if regime == 'old':
            recommendations.append("Maximize deductions under Section 80C (₹1.5 lakh limit)")
            recommendations.append("Utilize health insurance deduction under Section 80D")
            if income > 5000000:
                recommendations.append("Consider additional surcharge planning strategies")
        else:
            recommendations.append("Focus on investments with good returns rather than tax-saving")

        # Investment recommendations
        if investments:
            recommendations.append(f"Consider investing in {investments[0]['name']} for tax-saving and returns")

        # General tax planning recommendations
        recommendations.extend([
            "Start tax planning early in the financial year",
            "Diversify tax-saving investments across different instruments",
            "Review tax portfolio annually to align with changing financial goals"
        ])

        return recommendations[:5]

    # ==================== CONDITIONAL EDGES ====================
    def _should_route_to_mode(self, state: TaxState) -> str:
        """Route to appropriate analysis mode"""
        if state.get("error"):
            return "end"

        analysis_mode = state.get("analysis_mode", "tax_planning")
        return analysis_mode

    # ==================== GRAPH CONSTRUCTION ====================
    def _build_graph(self):
        """Build the LangGraph state graph"""
        graph = StateGraph(TaxState)

        # Add nodes
        graph.add_node("initialize_analysis", self._node_initialize_analysis)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("tax_calculation", self._node_tax_calculation)
        graph.add_node("tax_planning", self._node_tax_planning)

        # Add edges
        graph.add_edge(START, "initialize_analysis")
        graph.add_edge("initialize_analysis", "determine_mode")

        # Conditional routing based on analysis mode
        graph.add_conditional_edges(
            "determine_mode",
            self._should_route_to_mode,
            {
                "tax_calculation": "tax_calculation",
                "tax_planning": "tax_planning",
                "end": END
            }
        )

        # Both analysis modes go to END
        graph.add_edge("tax_calculation", END)
        graph.add_edge("tax_planning", END)

        return graph.compile()

    # ==================== PUBLIC INTERFACE ====================
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the tax planning LangGraph

        Args:
            state: Input state with query, profile, transactions

        Returns:
            Analysis results with summary and recommendations
        """
        try:
            # Convert input to TaxState
            tax_state: TaxState = {
                "query": state.get("query", ""),
                "profile": state.get("profile", {}),
                "transactions": state.get("transactions", [])
            }

            # Run the graph
            final_state = self.graph.invoke(tax_state)

            # Format output for compatibility with existing system
            result = {
                "summary": final_state.get("summary", "Analysis completed"),
                "next_best_actions": final_state.get("next_best_actions", []),
                "mode": final_state.get("analysis_mode", "unknown"),
            }

            # Add mode-specific data
            if final_state.get("analysis_mode") == "tax_calculation":
                result.update({
                    "tax_breakdown": final_state.get("tax_breakdown", {}),
                    "deductions": final_state.get("deductions", []),
                    "tax_regime": final_state.get("tax_regime", "old")
                })
            else:
                result.update({
                    "tax_analysis": final_state.get("tax_analysis", {}),
                    "investments": final_state.get("investments", []),
                    "recommended_regime": final_state.get("tax_regime", "old")
                })

            return result

        except Exception as e:
            return {
                "summary": f"Tax planning error: {str(e)}",
                "next_best_actions": ["Please try again with a different query"],
                "mode": "error"
            }

# Create the main class for compatibility
class TaxPlanner:
    """LangGraph-based Tax Planner with intent-aware routing"""

    def __init__(self):
        self.planner_graph = TaxPlannerGraph()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run tax planning with intent-aware routing

        Args:
            state: Input state with query, profile, transactions, and optionally 'intent'

        Returns:
            Analysis results with summary and recommendations
        """
        from backend.agents.agent_helpers import (
            ProfileValidator,
            is_general_query,
            format_general_info_response
        )

        query = state.get("query", "")
        intent = state.get("intent", "personalized")  # Default to personalized if not specified
        profile = state.get("profile", {})

        # Handle general queries - provide tax slab information without personalization
        if is_general_query(query, intent):
            return self._provide_general_tax_info(query)

        # For personalized queries, check profile completeness
        required_fields = ["income"]  # Minimum requirement for tax calculation
        missing_fields = ProfileValidator.get_missing_fields(profile, required_fields)

        if missing_fields:
            # Generate request for missing information
            message = ProfileValidator.generate_missing_fields_message(
                missing_fields,
                agent_name="Tax Planner",
                query_context="calculate your tax liability accurately"
            )
            return {
                "summary": message,
                "next_best_actions": [f"Provide your {field}" for field in missing_fields],
                "mode": "awaiting_profile",
                "requires_input": True,
                "missing_fields": missing_fields
            }

        # Profile is complete, proceed with personalized calculation
        return self.planner_graph.run(state)

    def _provide_general_tax_info(self, query: str) -> Dict[str, Any]:
        """
        Provide general tax information without personalization

        Args:
            query: User query

        Returns:
            General tax information response
        """
        from backend.agents.agent_helpers import format_general_info_response

        # Generate comprehensive tax slab information for FY 2024-25
        info = """**Income Tax Slabs for FY 2024-25 (AY 2025-26)**

**New Tax Regime (Default):**
• ₹0 - ₹2.5 lakhs: 0%
• ₹2.5 - ₹5 lakhs: 5%
• ₹5 - ₹7.5 lakhs: 10%
• ₹7.5 - ₹10 lakhs: 15%
• ₹10 - ₹12.5 lakhs: 20%
• ₹12.5 - ₹15 lakhs: 25%
• Above ₹15 lakhs: 30%

*Standard deduction of ₹50,000 available*
*Rebate under Section 87A: Full tax rebate for income up to ₹7 lakhs*

**Old Tax Regime (Optional):**
*General (Below 60 years):*
• ₹0 - ₹2.5 lakhs: 0%
• ₹2.5 - ₹5 lakhs: 5%
• ₹5 - ₹10 lakhs: 20%
• Above ₹10 lakhs: 30%

*Senior Citizens (60-80 years):*
• ₹0 - ₹3 lakhs: 0%
• ₹3 - ₹5 lakhs: 5%
• ₹5 - ₹10 lakhs: 20%
• Above ₹10 lakhs: 30%

*Super Senior Citizens (Above 80 years):*
• ₹0 - ₹5 lakhs: 0%
• ₹5 - ₹10 lakhs: 20%
• Above ₹10 lakhs: 30%

**Key Differences:**
• New regime: No deductions (80C, 80D, HRA, etc.) but lower tax rates
• Old regime: Allows all deductions and exemptions but higher rates
• Standard deduction: ₹50,000 for salaried/pensioners in both regimes

**Health & Education Cess:** 4% on total tax (both regimes)"""

        formatted = format_general_info_response(info, "Tax Planner")

        return {
            "summary": formatted,
            "next_best_actions": [
                "Compare which regime is better for you",
                "Calculate tax savings with investments",
                "Plan tax-saving investments"
            ],
            "mode": "general_information"
        }
