# ==============================================
# File: src/agents/insurance_analyzer_langgraph.py
# Description: LangGraph State-Based Insurance Analyzer
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

# Define Insurance-specific State
class InsuranceState(TypedDict, total=False):
    # Input data
    query: str
    profile: Dict[str, Any]
    transactions: List[Dict[str, Any]]
    
    # User information
    user_name: str
    age: int
    income: int
    dependents: int
    
    # Analysis results
    insurance_spending: Dict[str, Any]
    criteria: Dict[str, Any]
    query_info: Dict[str, Any]
    insurance_needs: List[str]
    risk_profile: str
    
    # Policy search results
    policies: List[Dict[str, Any]]
    
    # Mode selection
    analysis_mode: str  # "policy_search" or "profile_analysis"
    
    # Final output
    summary: str
    next_best_actions: List[str]
    insurance_analysis: Dict[str, Any]
    
    # Error handling
    error: Optional[str]

@dataclass
class InsuranceConfig:
    max_policies: int = 5
    default_age: int = 35
    default_income: int = 500000

class InsuranceAnalyzerGraph:
    """LangGraph-based Insurance Analyzer with State Management"""
    
    def __init__(self, config: Optional[InsuranceConfig] = None):
        self.config = config or InsuranceConfig()
        
        # Initialize insurance data and analysis parameters
        self.insurance_data = {"vehicle": [], "health": [], "term": []}
        self._load_insurance_data()
        
        # Profile-based analysis parameters
        self.age_based_recommendations = {
            "young_adult": (18, 30, ["term_life", "health", "vehicle"]),
            "early_career": (30, 40, ["term_life", "health", "vehicle", "disability"]),
            "mid_career": (40, 50, ["term_life", "health", "vehicle", "disability", "umbrella"]),
            "pre_retirement": (50, 60, ["health", "long_term_care", "umbrella"]),
            "retirement": (60, 100, ["health", "long_term_care"])
        }
        
        self.income_thresholds = {
            "low": 300000,
            "medium": 1000000,
            "high": 2500000,
            "very_high": 2500000
        }
        
        # Build the graph
        self.graph = self._build_graph()

    def _load_insurance_data(self):
        """Load insurance data from XML files"""
        possible_paths = ["data/"]
        files = {
            "vehicle": "vehicleInsurance.xml",
            "health": "healthInsurance.xml",
            "term": "termInsurance.xml"
        }
        
        for insurance_type, filename in files.items():
            loaded = False
            for base_path in possible_paths:
                file_path = os.path.join(base_path, filename)
                try:
                    if os.path.exists(file_path):
                        tree = ET.parse(file_path)
                        if insurance_type == "vehicle":
                            self.insurance_data[insurance_type] = self._parse_vehicle_xml(tree.getroot())
                        elif insurance_type == "health":
                            self.insurance_data[insurance_type] = self._parse_health_xml(tree.getroot())
                        elif insurance_type == "term":
                            self.insurance_data[insurance_type] = self._parse_term_xml(tree.getroot())
                        logger.info("Loaded %s %s policies", len(self.insurance_data[insurance_type]), insurance_type)
                        loaded = True
                        break
                except Exception:
                    continue
            if not loaded:
                logger.warning("No %s insurance data found (expected %s)", insurance_type, filename)

    def _parse_vehicle_xml(self, root):
        """Parse vehicle insurance XML data"""
        policies = []
        for insurance in root.findall("Insurance"):
            policy = {
                "name": insurance.get("name", ""),
                "type": self._get_element_text(insurance, "Type"),
                "premium": int(self._get_element_text(insurance, "Premium", "0")),
                "no_claim_bonus": self._get_element_text(insurance, "NoClaimBonus", "No").lower() == "yes",
                "insured_declared_value": int(self._get_element_text(insurance, "InsuredDeclaredValue", "0")),
                "zero_depth": self._get_element_text(insurance, "ZeroDepth", "0%"),
                "natural_calamities_protection": self._get_element_text(insurance, "NaturalCalamitiesProtection", "No").lower() == "yes",
                "duration": self._get_element_text(insurance, "Duration", ""),
                "pillion_coverage": self._get_element_text(insurance, "PillionCoverage", "No").lower() == "yes"
            }
            policies.append(policy)
        return policies

    def _parse_health_xml(self, root):
        """Parse health insurance XML data"""
        policies = []
        for insurance in root.findall("Insurance"):
            covered_diseases = []
            diseases_elem = insurance.find("CoveredDiseases")
            if diseases_elem is not None:
                for disease in diseases_elem.findall("Disease"):
                    if disease.text:
                        covered_diseases.append(disease.text)
            
            policy = {
                "name": insurance.get("name", ""),
                "coverage_amount": int(self._get_element_text(insurance, "CoverageAmount", "0")),
                "coverage_till_age": int(self._get_element_text(insurance, "CoverageTillAge", "0")),
                "smart_exit": self._get_element_text(insurance, "SmartExit", "No").lower() == "yes",
                "payment_type": self._get_element_text(insurance, "PaymentType", ""),
                "tax_benefit_upto": int(self._get_element_text(insurance, "TaxBenefitUpto", "0")),
                "settled_claim_percent": self._get_element_text(insurance, "SettledClaimPercent", "0%"),
                "premium_amount": int(self._get_element_text(insurance, "PremiumAmount", "0")),
                "covered_diseases": covered_diseases,
                "e_consultation": self._get_element_text(insurance, "EConsultation", "No").lower() == "yes",
                "oversea_treatment": self._get_element_text(insurance, "OverseaTreatment", "No").lower() == "yes",
                "free_health_checkups": self._get_element_text(insurance, "FreeHealthCheckups", "No").lower() == "yes",
                "cashless": self._get_element_text(insurance, "Cashless", "No").lower() == "yes",
                "mid_year_member_addition": self._get_element_text(insurance, "MidYearMemberAddition", "No").lower() == "yes"
            }
            policies.append(policy)
        return policies

    def _parse_term_xml(self, root):
        """Parse term insurance XML data"""
        policies = []
        for insurance in root.findall("Insurance"):
            covered_diseases = []
            diseases_elem = insurance.find("CoveredLifeThreateningDiseases")
            if diseases_elem is not None:
                for disease in diseases_elem.findall("Disease"):
                    if disease.text:
                        covered_diseases.append(disease.text)
            
            covered_activities = []
            activities_elem = insurance.find("CoveredActivities")
            if activities_elem is not None:
                for activity in activities_elem.findall("Activity"):
                    if activity.text:
                        covered_activities.append(activity.text)
            
            policy = {
                "name": insurance.get("name", ""),
                "life_coverage_amount": int(self._get_element_text(insurance, "LifeCoverageAmount", "0")),
                "coverage_till_age": int(self._get_element_text(insurance, "CoverageTillAge", "0")),
                "smart_exit": self._get_element_text(insurance, "SmartExit", "No").lower() == "yes",
                "payment_type": self._get_element_text(insurance, "PaymentType", ""),
                "tax_benefit_upto": int(self._get_element_text(insurance, "TaxBenefitUpto", "0")),
                "settled_claim_percent": self._get_element_text(insurance, "SettledClaimPercent", "0%"),
                "premium_amount": int(self._get_element_text(insurance, "PremiumAmount", "0")),
                "covered_life_threatening_diseases": covered_diseases,
                "covered_activities": covered_activities
            }
            policies.append(policy)
        return policies

    def _get_element_text(self, parent, tag_name, default=""):
        """Safely get text from XML element"""
        element = parent.find(tag_name)
        return element.text if element is not None and element.text is not None else default

    # ==================== LANGGRAPH NODES ====================

    def _node_initialize_analysis(self, state: InsuranceState) -> InsuranceState:
        """Initialize insurance analysis - extract user data and query info"""
        try:
            # Extract user information
            profile = state.get("profile", {}) or {}
            normalized = self._normalize_profile(profile)
            user_name = normalized["name"]
            age = normalized["age"]
            income = normalized["income"]
            dependents = normalized["dependents"]
            
            # Analyze current insurance spending from transactions
            transactions = state.get("transactions", [])
            insurance_spending = self._analyze_insurance_transactions(transactions)
            
            # Extract criteria from query
            query = state.get("query", "")
            criteria = self._extract_criteria_from_query(query)
            
            # Parse query for insurance types
            query_info = self._parse_insurance_query(query.lower())
            
            # Update state
            new_state = dict(state)
            new_state.update({
                "user_name": user_name,
                "age": age,
                "income": income,
                "dependents": dependents,
                "insurance_spending": insurance_spending,
                "criteria": criteria,
                "query_info": query_info,
                "error": None
            })
            
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Initialization error: {str(e)}"
            return new_state

    def _node_determine_mode(self, state: InsuranceState) -> InsuranceState:
        """Determine analysis mode: policy search vs profile analysis"""
        try:
            criteria = state.get("criteria", {})
            
            # Policy Search Mode: Specific queries with premium/coverage requirements
            if criteria.get('insurance_type') and ('max_premium' in criteria or 'min_coverage' in criteria):
                analysis_mode = "policy_search"
            else:
                # Profile Analysis Mode: General insurance advice
                analysis_mode = "profile_analysis"
            
            new_state = dict(state)
            new_state["analysis_mode"] = analysis_mode
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Mode determination error: {str(e)}"
            return new_state

    def _node_policy_search(self, state: InsuranceState) -> InsuranceState:
        """Search for specific insurance policies based on criteria"""
        try:
            criteria = state.get("criteria", {})
            policies = self._search_policies(criteria)
            
            if not policies:
                summary = f"No {criteria.get('insurance_type', 'insurance')} policies found matching your criteria: {criteria}"
                next_best_actions = [
                    "Try adjusting your budget requirements",
                    "Consider different insurance types",
                    "Contact an insurance advisor"
                ]
            else:
                # Format policy search results
                insurance_type = criteria.get('insurance_type', 'insurance')
                summary_parts = [f"Found {len(policies)} {insurance_type} insurance policies matching your criteria:\n"]
                
                for i, policy in enumerate(policies, 1):
                    summary_parts.append(f"\n{i}. **{policy['name']}**")
                    
                    if insurance_type == 'vehicle':
                        summary_parts.append(f"   - Premium: ₹{policy.get('premium', 0):,}/year")
                        summary_parts.append(f"   - Coverage: ₹{policy.get('insured_declared_value', 0):,}")
                        summary_parts.append(f"   - Type: {policy.get('type', 'N/A')}")
                        summary_parts.append(f"   - Duration: {policy.get('duration', 'N/A')}")
                        if policy.get('no_claim_bonus'):
                            summary_parts.append(f"   - No Claim Bonus: Available")
                    elif insurance_type == 'health':
                        summary_parts.append(f"   - Premium: ₹{policy.get('premium_amount', 0):,}/year")
                        summary_parts.append(f"   - Coverage: ₹{policy.get('coverage_amount', 0):,}")
                        summary_parts.append(f"   - Claim Settlement: {policy.get('settled_claim_percent', 'N/A')}")
                        summary_parts.append(f"   - Payment: {policy.get('payment_type', 'N/A')}")
                    elif insurance_type == 'term':
                        summary_parts.append(f"   - Premium: ₹{policy.get('premium_amount', 0):,}/year")
                        summary_parts.append(f"   - Life Coverage: ₹{policy.get('life_coverage_amount', 0):,}")
                        summary_parts.append(f"   - Claim Settlement: {policy.get('settled_claim_percent', 'N/A')}")
                
                summary_parts.append(f"\nSearch criteria: {criteria}")
                summary = "".join(summary_parts)
                
                next_best_actions = [
                    "Compare policy features in detail",
                    "Check claim settlement ratios",
                    "Read terms and conditions carefully",
                    "Get personalized quotes from insurers",
                    "Consult with insurance agent for advice"
                ]
            
            new_state = dict(state)
            new_state.update({
                "policies": policies,
                "summary": summary,
                "next_best_actions": next_best_actions
            })
            
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Policy search error: {str(e)}"
            return new_state

    def _node_profile_analysis(self, state: InsuranceState) -> InsuranceState:
        """Analyze insurance needs based on user profile"""
        try:
            # Extract state variables
            age = state.get("age", self.config.default_age)
            income = state.get("income", self.config.default_income)
            dependents = state.get("dependents", 0)
            user_name = state.get("user_name", "user")
            insurance_spending = state.get("insurance_spending", {})
            query_info = state.get("query_info", {})
            
            # Assess insurance needs based on profile
            insurance_needs = self._assess_insurance_needs(age, income, dependents)
            
            # Get risk profile
            risk_profile = self._get_risk_profile(age, income, dependents)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                insurance_needs, insurance_spending, query_info, income
            )
            
            # Create summary using original algorithm
            summary = self._create_summary(
                user_name, age, income, insurance_needs, insurance_spending, query_info
            )
            
            # Create insurance analysis object
            insurance_analysis = {
                "current_spending": insurance_spending,
                "recommended_types": insurance_needs,
                "risk_profile": risk_profile
            }
            
            new_state = dict(state)
            new_state.update({
                "insurance_needs": insurance_needs,
                "risk_profile": risk_profile,
                "summary": summary,
                "next_best_actions": recommendations,
                "insurance_analysis": insurance_analysis
            })
            
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Profile analysis error: {str(e)}"
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
        try:
            dependents = int(profile.get("dependents", 0))
        except Exception:
            dependents = 0

        age = max(0, age)
        income = max(0, income)
        dependents = max(0, dependents)

        return {"name": name, "age": age, "income": income, "dependents": dependents}

    def _analyze_insurance_transactions(self, transactions):
        """Analyze transactions to find current insurance spending"""
        insurance_spend = {"health": 0, "vehicle": 0, "life": 0, "other": 0, "total": 0}
        insurance_keywords = {
            "health": ["health insurance", "medical insurance", "mediclaim"],
            "vehicle": ["car insurance", "vehicle insurance", "motor insurance"],
            "life": ["life insurance", "term insurance", "lic"],
            "other": ["insurance"]
        }

        for txn in transactions:
            description = txn.get("description", "").lower()
            category = txn.get("category", "").lower() 
            amount = txn.get("amount", 0)
            
            if "insurance" in description or "insurance" in category:
                categorized = False
                for ins_type, keywords in insurance_keywords.items():
                    if ins_type != "other" and any(keyword in description for keyword in keywords):
                        insurance_spend[ins_type] += amount
                        categorized = True
                        break
                if not categorized:
                    insurance_spend["other"] += amount
                insurance_spend["total"] += amount

        return insurance_spend

    def _extract_criteria_from_query(self, query):
        """Extract insurance criteria from natural language query"""
        criteria = {}
        query_lower = query.lower()
        
        # Extract premium budget
        premium_patterns = [
            r'under\s+(?:rs\.?\s*)?(\d+(?:,\d+)*)',
            r'below\s+(?:rs\.?\s*)?(\d+(?:,\d+)*)',
            r'less\s+than\s+(?:rs\.?\s*)?(\d+(?:,\d+)*)',
            r'maximum\s+(?:rs\.?\s*)?(\d+(?:,\d+)*)',
            r'budget\s+(?:of\s+)?(?:rs\.?\s*)?(\d+(?:,\d+)*)'
        ]
        
        for pattern in premium_patterns:
            match = re.search(pattern, query_lower)
            if match:
                premium_str = match.group(1).replace(',', '')
                criteria['max_premium'] = int(premium_str)
                break
        
        # Extract coverage amount  
        coverage_patterns = [
            r'coverage\s+(?:of\s+)?(?:rs\.?\s*)?(\d+(?:,\d+)*)\s*(?:lakh|lakhs?|crore|crores?)?',
            r'cover\s+(?:of\s+)?(?:rs\.?\s*)?(\d+(?:,\d+)*)\s*(?:lakh|lakhs?|crore|crores?)?'
        ]
        
        for pattern in coverage_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = int(amount_str)
                if 'lakh' in query_lower:
                    amount *= 100000
                elif 'crore' in query_lower:
                    amount *= 10000000
                criteria['min_coverage'] = amount
                break
        
        # Detect insurance and vehicle type
        if any(term in query_lower for term in ['2 wheeler', 'two wheeler', 'bike', 'motorcycle', 'scooter']):
            criteria['insurance_type'] = 'vehicle'
            criteria['vehicle_type'] = '2 Wheeler'
        elif any(term in query_lower for term in ['4 wheeler', 'four wheeler', 'car', 'auto']):
            criteria['insurance_type'] = 'vehicle'
            criteria['vehicle_type'] = '4 Wheeler'
        elif 'health' in query_lower:
            criteria['insurance_type'] = 'health'
        elif any(term in query_lower for term in ['term', 'life']):
            criteria['insurance_type'] = 'term'
        elif 'vehicle' in query_lower:
            criteria['insurance_type'] = 'vehicle'
        
        return criteria

    def _parse_insurance_query(self, query):
        """Parse user query for specific insurance requests"""
        query_info = {
            "type": None,
            "budget": None,
            "coverage_amount": None,
            "specific_request": False
        }

        # Check for insurance types
        if any(term in query for term in ['health', 'medical', 'mediclaim']):
            query_info["type"] = "health"
            query_info["specific_request"] = True
        elif any(term in query for term in ['vehicle', 'car', 'motor', 'bike']):
            query_info["type"] = "vehicle"
            query_info["specific_request"] = True
        elif any(term in query for term in ['life', 'term']):
            query_info["type"] = "life"
            query_info["specific_request"] = True

        return query_info

    def _search_policies(self, criteria):
        """Search policies based on criteria"""
        insurance_type = criteria.get('insurance_type', 'vehicle')
        all_policies = self.insurance_data.get(insurance_type, [])
        
        if not all_policies:
            return []
        
        filtered = []
        for policy in all_policies:
            # Check vehicle type for vehicle insurance
            if insurance_type == 'vehicle' and 'vehicle_type' in criteria:
                if policy.get('type') != criteria['vehicle_type']:
                    continue
            
            # Check premium budget
            if 'max_premium' in criteria:
                if insurance_type == 'vehicle':
                    if policy.get('premium', 0) > criteria['max_premium']:
                        continue
                else:  # health or term
                    if policy.get('premium_amount', 0) > criteria['max_premium']:
                        continue
            
            # Check minimum coverage
            if 'min_coverage' in criteria:
                if insurance_type == 'vehicle':
                    if policy.get('insured_declared_value', 0) < criteria['min_coverage']:
                        continue
                elif insurance_type == 'health':
                    if policy.get('coverage_amount', 0) < criteria['min_coverage']:
                        continue
                elif insurance_type == 'term':
                    if policy.get('life_coverage_amount', 0) < criteria['min_coverage']:
                        continue
            
            filtered.append(policy)
        
        # Sort by premium (ascending)
        if insurance_type == 'vehicle':
            filtered.sort(key=lambda x: x.get('premium', float('inf')))
        else:
            filtered.sort(key=lambda x: x.get('premium_amount', float('inf')))
            
        return filtered[:self.config.max_policies]

    def _assess_insurance_needs(self, age, income, dependents):
        """Assess insurance needs based on user profile"""
        needs = []
        
        # Age-based recommendations
        for category, (min_age, max_age, recommendations) in self.age_based_recommendations.items():
            if min_age <= age <= max_age:
                needs.extend(recommendations)
                break
        
        # Adjust based on dependents
        if dependents > 0:
            if "term_life" not in needs:
                needs.append("term_life")
            needs.append("child_education")
        
        # Income-based adjustments
        if income > self.income_thresholds["high"]:
            if "umbrella" not in needs:
                needs.append("umbrella")
                
        return list(set(needs))

    def _get_risk_profile(self, age, income, dependents):
        """Determine risk profile for insurance planning"""
        risk_score = 0
        
        # Age factor
        if age < 30:
            risk_score += 1
        elif age > 50:
            risk_score += 3
        else:
            risk_score += 2
            
        # Income factor
        if income < self.income_thresholds["low"]:
            risk_score += 3
        elif income > self.income_thresholds["high"]:
            risk_score += 1
        else:
            risk_score += 2
            
        # Dependents factor
        risk_score += min(dependents, 3)
        
        if risk_score <= 3:
            return "Low Risk"
        elif risk_score <= 6:
            return "Moderate Risk"
        else:
            return "High Risk"

    def _create_summary(self, name, age, income, needs, spending, query_info):
        """Create comprehensive summary of insurance analysis (ORIGINAL CODE!)"""
        total_spending = spending.get("total", 0)
        summary = f"Hello {name}, "
        
        if query_info.get("specific_request"):
            insurance_type = query_info["type"]
            summary += f"I've analyzed your {insurance_type} insurance requirements. "
        else:
            summary += f"I've analyzed your insurance needs based on your profile (age: {age}, income: ₹{income:,}). "
        
        if total_spending > 0:
            summary += f"You're currently spending ₹{total_spending:,} annually on insurance. "
        else:
            summary += "You don't appear to have significant insurance coverage currently. "
        
        summary += f"Based on your profile, I recommend focusing on: {', '.join(needs[:3])}. "
        
        recommended_budget = int(income * 0.12)
        if total_spending < recommended_budget * 0.5:
            summary += f"Consider allocating ₹{recommended_budget:,} annually (12% of income) for comprehensive insurance coverage."
        else:
            summary += "Your insurance spending appears to be on track."
            
        return summary

    def _generate_recommendations(self, needs, current_spending, query_info, income):
        """Generate actionable insurance recommendations"""
        recommendations = []
        
        # Budget recommendation (12% of income for insurance)
        recommended_budget = int(income * 0.12)
        current_total = current_spending.get("total", 0)
        
        if current_total < recommended_budget * 0.5:
            recommendations.append(f"Increase insurance budget to ₹{recommended_budget:,} annually (12% of income)")

        # Specific type recommendations
        if query_info.get("specific_request"):
            insurance_type = query_info["type"]
            if insurance_type == "health":
                recommendations.extend([
                    "Compare health insurance policies with minimum ₹5 lakh coverage",
                    "Look for policies with cashless facility and network hospitals",
                    "Consider family floater if you have dependents"
                ])
            elif insurance_type == "vehicle":
                recommendations.extend([
                    "Get comprehensive vehicle insurance with zero depreciation",
                    "Include personal accident cover for driver",
                    "Consider roadside assistance add-on"
                ])
            elif insurance_type == "life":
                coverage_needed = income * 10  # 10x annual income
                recommendations.extend([
                    f"Get term life insurance with ₹{coverage_needed:,} coverage",
                    "Choose 20-30 year policy term based on retirement age",
                    "Avoid mixing insurance with investment"
                ])
        else:
            # General recommendations based on needs assessment
            if "health" in needs and current_spending.get("health", 0) < 25000:
                recommendations.append("Get health insurance with minimum ₹5 lakh coverage")
            if "term_life" in needs and current_spending.get("life", 0) < 20000:
                recommendations.append("Purchase term life insurance (10x your annual income)")
            if "vehicle" in needs and current_spending.get("vehicle", 0) < 15000:
                recommendations.append("Ensure comprehensive vehicle insurance is current")

        # General best practices
        recommendations.extend([
            "Review and update insurance policies annually",
            "Keep all insurance documents digitally backed up",
            "Inform family members about all active policies"
        ])

        return recommendations[:5]  # Limit to top 5 recommendations

    # ==================== CONDITIONAL EDGES ====================

    def _should_route_to_mode(self, state: InsuranceState) -> str:
        """Route to appropriate analysis mode"""
        if state.get("error"):
            return "end"
        
        analysis_mode = state.get("analysis_mode", "profile_analysis")
        return analysis_mode

    # ==================== GRAPH CONSTRUCTION ====================

    def _build_graph(self):
        """Build the LangGraph state graph"""
        graph = StateGraph(InsuranceState)
        
        # Add nodes
        graph.add_node("initialize_analysis", self._node_initialize_analysis)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("policy_search", self._node_policy_search)
        graph.add_node("profile_analysis", self._node_profile_analysis)
        
        # Add edges
        graph.add_edge(START, "initialize_analysis")
        graph.add_edge("initialize_analysis", "determine_mode")
        
        # Conditional routing based on analysis mode
        graph.add_conditional_edges(
            "determine_mode",
            self._should_route_to_mode,
            {
                "policy_search": "policy_search",
                "profile_analysis": "profile_analysis",
                "end": END
            }
        )
        
        # Both analysis modes go to END
        graph.add_edge("policy_search", END)
        graph.add_edge("profile_analysis", END)
        
        return graph.compile()

    # ==================== PUBLIC INTERFACE ====================

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the insurance analysis LangGraph
        
        Args:
            state: Input state with query, profile, transactions
            
        Returns:
            Analysis results with summary and recommendations
        """
        try:
            # Convert input to InsuranceState
            insurance_state: InsuranceState = {
                "query": state.get("query", ""),
                "profile": state.get("profile", {}),
                "transactions": state.get("transactions", [])
            }
            
            # Run the graph
            final_state = self.graph.invoke(insurance_state)
            
            # Format output for compatibility with existing system
            result = {
                "summary": final_state.get("summary", "Analysis completed"),
                "next_best_actions": final_state.get("next_best_actions", []),
                "mode": final_state.get("analysis_mode", "unknown"),
            }
            
            # Add mode-specific data
            if final_state.get("analysis_mode") == "policy_search":
                result.update({
                    "policies": final_state.get("policies", []),
                    "criteria": final_state.get("criteria", {})
                })
            else:
                result["insurance_analysis"] = final_state.get("insurance_analysis", {})
            
            return result
            
        except Exception as e:
            return {
                "summary": f"Insurance analysis error: {str(e)}",
                "next_best_actions": ["Please try again with a different query"],
                "mode": "error"
            }

# Create the main class for compatibility
class InsuranceAnalyzer:
    """LangGraph-based Insurance Analyzer - Compatible with existing orchestrator"""

    def __init__(self):
        self.analyzer_graph = InsuranceAnalyzerGraph()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run insurance analysis with intent-aware routing

        Flow:
        1. Check query intent (general vs personalized)
        2. For general queries: Provide educational insurance information
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

        # Handle general queries - provide insurance education without personalization
        if is_general_query(query, intent):
            return self._provide_general_insurance_info(query)

        # For personalized queries, check profile completeness
        required_fields = ["age", "income", "dependents"]
        missing_fields = ProfileValidator.get_missing_fields(profile, required_fields)

        if missing_fields:
            # Generate request for missing information
            message = ProfileValidator.generate_missing_fields_message(
                missing_fields,
                agent_name="Insurance Analyzer",
                query_context="recommend suitable insurance coverage based on your needs"
            )
            return {
                "summary": message,
                "next_best_actions": [f"Provide your {field.replace('_', ' ')}" for field in missing_fields],
                "mode": "awaiting_profile",
                "requires_input": True,
                "missing_fields": missing_fields
            }

        # Profile is complete, proceed with personalized analysis using LangGraph
        return self.analyzer_graph.run(state)

    def _provide_general_insurance_info(self, query: str) -> Dict[str, Any]:
        """Provide general insurance information without personalization"""
        from backend.agents.agent_helpers import format_general_info_response

        query_lower = query.lower()

        # Determine what type of information to provide
        if any(word in query_lower for word in ["term insurance", "term plan", "life insurance"]):
            info = """**Term Insurance: Pure Life Protection**

**What is Term Insurance?**
Term insurance is a pure life insurance product that provides financial protection to your family in case of your untimely death. It pays a lump sum to nominees if the insured dies during the policy term.

**Key Features:**
• **Coverage**: ₹25 lakhs to ₹2 crores (or higher)
• **Premium**: ₹500-3,000/month for ₹1 crore cover (30-year-old)
• **Policy Term**: 10-40 years
• **No Maturity Benefit**: If you survive, no payout (pure protection)
• **Tax Benefits**: Premium under Section 80C, claim under 10(10D)

**Types of Term Plans:**
1. **Level Term**: Fixed sum assured throughout
2. **Increasing Term**: Coverage increases annually
3. **Decreasing Term**: Coverage decreases (for loan protection)
4. **Return of Premium (ROP)**: Premiums refunded if you survive
5. **Term with Critical Illness**: Additional coverage for major illnesses

**Who Needs Term Insurance?**
• Primary earning member of family
• Parents with dependent children
• Individuals with loans (home, car)
• Self-employed professionals
• Anyone whose family depends on their income

**Coverage Calculation:**
**Recommended coverage = 10-15x annual income**

Example:
• Annual income: ₹10 lakhs
• Recommended coverage: ₹1-1.5 crore
• Premium at age 30: ₹15,000-20,000/year

**Key Benefits:**
• Affordable premiums
• High coverage amount
• Financial security for family
• Tax savings on premium
• Add-ons available (critical illness, accidental death)

**Important Riders (Add-ons):**
• **Critical Illness**: Payout on diagnosis of specified diseases
• **Accidental Death**: Extra payout if death due to accident
• **Waiver of Premium**: No premiums if disabled
• **Income Benefit**: Monthly payout instead of lump sum

**Things to Know:**
• Buy early (premiums increase with age)
• Disclose all health conditions honestly
• Compare online vs offline (online usually 30-40% cheaper)
• Choose term based on dependents' age
• Review coverage every 3-5 years
• Don't mix investment with insurance (avoid ULIPs/endowment)

**Claim Settlement Ratio:**
Check insurer's CSR (should be >95%)
- LIC: 98%+
- HDFC Life: 99%+
- ICICI Prudential: 98%+"""

        elif any(word in query_lower for word in ["health insurance", "mediclaim", "medical insurance"]):
            info = """**Health Insurance: Medical Cost Protection**

**What is Health Insurance?**
Health insurance covers hospitalization expenses, medical treatments, and related costs. It protects you from high medical bills and provides cashless treatment at network hospitals.

**Key Features:**
• **Sum Insured**: ₹3 lakhs to ₹1 crore
• **Premium**: ₹5,000-25,000/year for family (varies by age, coverage)
• **Cashless Treatment**: At network hospitals
• **Reimbursement**: For non-network hospitals
• **Tax Benefits**: Premium deduction under Section 80D

**Types of Health Insurance:**
1. **Individual**: Covers single person
2. **Family Floater**: Covers entire family (shared sum insured)
3. **Senior Citizen**: For 60+ years
4. **Critical Illness**: Lump sum on diagnosis of major illnesses
5. **Top-up/Super Top-up**: Additional coverage after deductible

**What's Covered?**
[OK] Hospitalization expenses (room, doctor, tests)
[OK] Pre and post hospitalization (30-60 days)
[OK] Daycare procedures (cataract, dialysis)
[OK] Ambulance charges
[OK] AYUSH treatment (some policies)
[OK] COVID-19 treatment

**What's NOT Covered?**
[ERROR] Pre-existing conditions (for first 2-4 years)
[ERROR] Cosmetic surgery
[ERROR] Dental treatment (unless accident)
[ERROR] Maternity (unless specific cover)
[ERROR] Self-inflicted injuries
[ERROR] War/nuclear risks

**Important Terms:**

**Waiting Period:**
• Initial: 30 days for general ailments
• Pre-existing: 2-4 years
• Specific diseases: 1-2 years (hernia, cataract, etc.)

**Co-payment:**
You pay a percentage (10-20%) of claim amount

**Sub-limits:**
Caps on room rent, specific procedures

**No Claim Bonus (NCB):**
Sum insured increases 5-10% every claim-free year

**Coverage Recommendations:**

**Individual (25-35 years):** ₹5-10 lakhs
**Family (with kids):** ₹10-15 lakhs floater
**Senior Citizens:** ₹5-10 lakhs individual
**Additional Top-up:** ₹25-50 lakhs (for major illnesses)

**Premiums (Approximate):**
• Age 25-35: ₹5-8k/year for ₹5L
• Age 35-45: ₹8-12k/year for ₹5L
• Age 45-60: ₹15-25k/year for ₹5L
• Family floater (4 members): ₹15-30k/year for ₹10L

**Choosing the Right Policy:**

**Check:**
• Claim settlement ratio (>85%)
• Network hospitals in your city
• Cashless facility availability
• Room rent limits (ICU important)
• Disease-specific sub-limits
• Restoration benefit
• No Claim Bonus

**Tax Benefits (Section 80D):**
• Self/spouse/children: Up to ₹25,000
• Parents (<60 years): Up to ₹25,000
• Parents (>60 years): Up to ₹50,000
• **Total max**: ₹75,000-1,00,000/year

**Tips:**
• Buy young (low premiums, easy approval)
• Disclose all pre-existing conditions
• Prefer family floater over individual
• Add parents on separate policy
• Consider top-up for major illnesses
• Port policy if better options available
• Keep all medical records"""

        elif any(word in query_lower for word in ["car insurance", "vehicle insurance", "motor insurance"]):
            info = """**Car/Vehicle Insurance: Mandatory Protection**

**What is Car Insurance?**
Insurance that covers damages to your vehicle and third-party liabilities. It's legally mandatory for all vehicles on Indian roads.

**Types of Car Insurance:**

**1. Third-Party Insurance (TP)**
• **Mandatory by law**
• Covers damage/injury to others (not your car)
• Death: Up to ₹15 lakhs
• Injury: Up to ₹7.5 lakhs
• Property damage: Up to ₹1 lakh
• **Premium**: ₹2,000-5,000/year (fixed by IRDAI)

**2. Comprehensive Insurance**
• Covers your car + third-party
• **Own Damage**: Accident, theft, fire, natural calamities
• **Third-Party Liability**: As above
• **Add-ons available**
• **Premium**: ₹10,000-50,000/year (depends on IDV)

**3. Standalone Own Damage**
• Only covers your car (not third-party)
• Useful if you have existing TP for 3 years

**Key Terms:**

**IDV (Insured Declared Value):**
• Current market value of your car
• Decreases every year (depreciation)
• Maximum claim amount you can get
• Higher IDV = Higher premium

**NCB (No Claim Bonus):**
• Discount for claim-free years
• Year 1: 20%
• Year 2: 25%
• Year 3: 35%
• Year 4: 45%
• Year 5+: 50%
• **NCB is personal** (transferable to new car)

**Deductible:**
• Amount you pay from pocket during claim
• Higher deductible = Lower premium
• Example: ₹2,000 deductible means you pay first ₹2k

**Add-on Covers (Riders):**
1. **Zero Depreciation**: No depreciation on parts during claim
2. **Engine Protection**: Covers engine damage (water, oil)
3. **Consumables Cover**: Nuts, bolts, fluids, AC gas
4. **Roadside Assistance**: Towing, flat tire, fuel delivery
5. **Return to Invoice**: Get full purchase price if total loss
6. **NCB Protection**: Retain NCB even after 1 claim
7. **Key Replacement**: Lost/stolen key replacement

**What's Covered (Comprehensive):**
[OK] Accidents (collision, overturn)
[OK] Theft
[OK] Fire explosion
[OK] Natural calamities (flood, earthquake, landslide)
[OK] Riots, strikes, terrorism
[OK] Third-party death/injury/property damage

**What's NOT Covered:**
[ERROR] Wear and tear
[ERROR] Mechanical/electrical breakdown
[ERROR] Drunk driving
[ERROR] Driving without valid license
[ERROR] Consequential damage
[ERROR] Depreciation (unless zero-dep cover)

**Premium Calculation Factors:**
• IDV (car's current value)
• City (metro = higher)
• Car make/model (luxury = higher)
• Cubic capacity (>1500cc = higher)
• Add-ons selected
• NCB discount
• Age of car

**Example Premiums:**
**Maruti Swift (Delhi, ₹5L IDV):**
• TP only: ₹2,094
• Comprehensive: ₹12,000-15,000
• With zero-dep: ₹18,000-20,000

**Tips:**
• Always compare online (10-15% cheaper)
• Don't under-insure (declare correct IDV)
• Add zero-depreciation for new cars (<3 years)
• Check claim settlement ratio
• Keep policy and documents in car
• Inform insurer within 48 hours of accident
• Renew before expiry (no NCB loss)"""

        elif any(word in query_lower for word in ["travel insurance", "trip insurance"]):
            info = """**Travel Insurance: Protection on the Go**

**What is Travel Insurance?**
Insurance that covers medical emergencies, trip cancellations, lost baggage, and other travel-related risks during domestic or international trips.

**Coverage Areas:**

**1. Medical Expenses Abroad**
• Emergency hospitalization
• Doctor consultations
• Ambulance charges
• COVID-19 treatment
• Medical evacuation
• Repatriation (body transportation)

**2. Trip Cancellation/Interruption**
• Refund if trip cancelled due to:
  - Medical emergency
  - Family member's death
  - Natural disaster
  - Visa rejection

**3. Baggage Loss/Delay**
• Lost checked-in baggage
• Delayed baggage (>12 hours)
• Stolen belongings
• Passport loss assistance

**4. Flight Delay/Cancellation**
• Compensation for delays >6 hours
• Hotel accommodation
• Meal expenses

**5. Personal Accident**
• Accidental death
• Permanent disability
• Fractures/burns

**6. Other Benefits**
• Personal liability (damage to property/injury to others)
• Missed connection
• Hijack distress
• Loss of documents

**Types:**

**Domestic Travel:**
• Coverage: ₹1-5 lakhs
• Premium: ₹200-500 for 7 days
• Medical emergencies within India

**International Travel:**
• Coverage: $25,000-1,00,000
• Premium: ₹500-3,000 per week (varies by destination)
• Schengen countries require minimum €30,000

**Premium Factors:**
• Destination (USA/Canada = expensive)
• Trip duration
• Age of traveler
• Coverage amount
• Pre-existing conditions
• Adventure activities

**Common Exclusions:**
[ERROR] Pre-existing medical conditions (unless declared)
[ERROR] Adventure sports (unless rider added)
[ERROR] Intoxication-related incidents
[ERROR] War zones
[ERROR] Self-inflicted injuries
[ERROR] Routine check-ups

**Schengen Travel Requirements:**
• Minimum €30,000 medical coverage
• Valid for all Schengen countries
• Covers entire trip duration
• Includes repatriation

**Tips:**
• Buy before trip booking (covers cancellation)
• Declare pre-existing conditions
• Add adventure sports rider if needed
• Keep claim documents (bills, reports, FIR)
• Register with insurer before hospitalization
• Check cashless network hospitals abroad"""

        else:
            # General insurance overview
            info = """**Insurance in India: Complete Overview**

**Why Insurance Matters:**
• Protects against financial shocks
• Provides peace of mind
• Tax benefits available
• Mandatory for certain assets (vehicles, loans)

**Essential Insurance Types:**

**1. Term Life Insurance** (Priority 1)
• **Who needs**: Primary earning member
• **Coverage**: 10-15x annual income
• **Cost**: ₹500-3,000/month for ₹1 crore
• **Purpose**: Financial security for family
• **Tax**: Deduction under 80C

**2. Health Insurance** (Priority 2)
• **Who needs**: Everyone
• **Coverage**: ₹5-15 lakhs family floater
• **Cost**: ₹10,000-30,000/year
• **Purpose**: Medical expense protection
• **Tax**: Deduction under 80D (up to ₹75k-1L)

**3. Accidental Insurance** (Priority 3)
• **Who needs**: Working professionals
• **Coverage**: ₹50 lakhs-1 crore
• **Cost**: ₹500-2,000/year
• **Purpose**: Disability/death due to accident

**4. Motor Insurance** (Mandatory)
• **Two-wheeler**: ₹500-5,000/year
• **Four-wheeler**: ₹2,000-25,000/year
• **Third-party compulsory by law**
• **Comprehensive recommended**

**Optional but Useful:**

**5. Travel Insurance**
• For frequent travelers
• ₹500-3,000 per trip
• Medical + trip protection

**6. Home Insurance**
• Fire, theft, natural disasters
• ₹2,000-10,000/year
• Structure + contents coverage

**7. Critical Illness**
• Lump sum on diagnosis
• ₹25-50 lakhs coverage
• ₹8,000-20,000/year

**Insurance by Life Stage:**

**20s-30s (Single):**
• Term insurance (₹50L-1Cr)
• Health insurance (₹5-10L)
• Accidental insurance

**30s-40s (Married with kids):**
• Term insurance (₹1-2Cr)
• Health insurance (₹15-25L floater + top-up)
• Children's education insurance
• Retirement planning (NPS/pension)

**40s-50s (Mid-career):**
• Increase term coverage (₹1.5-2.5Cr)
• Super top-up health (₹50L-1Cr)
• Critical illness rider
• Parents' health insurance

**50s-60s (Pre-retirement):**
• Maintain term till 60-65
• Increase health coverage (₹25-50L)
• Senior citizen health for parents
• Pension annuity plans

**Common Mistakes to Avoid:**

[ERROR] **Mixing insurance with investment**
   (Avoid ULIPs, endowment plans - low returns)

[ERROR] **Under-insuring**
   (₹10L term for sole earner with ₹50L loan)

[ERROR] **Not disclosing pre-existing conditions**
   (Claims will be rejected)

[ERROR] **Buying only employer health insurance**
   (Coverage stops when you leave job)

[ERROR] **Ignoring inflation**
   (₹5L health cover won't be enough in 10 years)

[ERROR] **Too many policies**
   (Complexity in renewals, claims)

**Tax Benefits:**

**Section 80C** (Up to ₹1.5 lakhs):
• Term insurance premium
• ULIP/endowment premium
• Pension plan contributions

**Section 80D** (Up to ₹75k-1L):
• Health insurance premium
  - Self/family: ₹25,000
  - Parents: ₹25,000
  - Senior citizen parents: ₹50,000
• Preventive health check-up: ₹5,000

**Section 10(10D):**
• Life insurance claim/maturity amount (tax-free)

**Choosing the Right Insurer:**

**Check:**
• Claim Settlement Ratio (>95% for life, >85% for health)
• Solvency Ratio (>1.5)
• Customer reviews
• Network hospitals (health)
• Claim process simplicity

**Top Insurers (CSR):**
• **Life**: HDFC Life (99%), ICICI Prudential (98%), Max Life (99%)
• **Health**: Star Health (90%), HDFC Ergo (95%), Care Health (93%)
• **Motor**: HDFC Ergo, ICICI Lombard, Bajaj Allianz

**Annual Insurance Budget:**

For ₹10 lakh annual income family:
• Term insurance: ₹15,000 (₹1.5Cr cover)
• Health insurance: ₹20,000 (₹15L floater)
• Accidental: ₹1,500 (₹1Cr cover)
• Motor: ₹15,000 (car comprehensive)
• **Total**: ~₹50,000/year (5% of income)

**Key Principles:**
1. Buy term insurance, not endowment
2. Separate insurance from investment
3. Increase coverage with life changes
4. Review policies annually
5. Disclose everything honestly
6. Keep all policies active
7. Nominate beneficiaries correctly"""

        formatted = format_general_info_response(info, "Insurance Analyzer")

        return {
            "summary": formatted,
            "next_best_actions": [
                "Provide your age, income, and number of dependents for personalized insurance recommendations",
                "Ask specific questions about insurance types or coverage",
                "Compare insurance plans from different providers"
            ],
            "mode": "general_information"
        }
