# ==============================================
# File: src/agents/investment_coach.py
# Description: LangGraph State-Based Insurance Analyzer
# ==============================================

from __future__ import annotations
from typing import Dict, Any, List, TypedDict, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import os
import re
import asyncio
import requests
import json
import logging
from dotenv import load_dotenv
from pathlib import Path
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

# Define Investment-specific State
class InvestmentState(TypedDict, total=False):
    # Input data
    query: str
    profile: Dict[str, Any]
    transactions: List[Dict[str, Any]]
    
    # User information
    user_name: str
    age: int
    income: int
    risk_tolerance: str
    investment_horizon: str
    
    # Analysis results
    investment_spending: Dict[str, Any]
    criteria: Dict[str, Any]
    query_info: Dict[str, Any]
    investment_needs: List[str]
    risk_profile: str
    
    # Investment search results
    investments: List[Dict[str, Any]]
    
    # Mode selection
    analysis_mode: str  # "investment_search" or "portfolio_analysis"
    
    # Final output
    summary: str
    next_best_actions: List[str]
    investment_analysis: Dict[str, Any]
    
    # Error handling
    error: Optional[str]

@dataclass
class InvestmentConfig:
    max_investments: int = 5
    default_age: int = 35
    default_income: int = 500000
    xml_path: str = os.path.join("data", "financial_offers.xml")

class InvestmentCoachGraph:
    """LangGraph-based Investment Coach with State Management"""
    
    SUPPORTED_TYPES = ("stock", "mutual fund", "fixed deposit", "gold")
    
    def __init__(self, config: Optional[InvestmentConfig] = None):
        self.config = config or InvestmentConfig()
        
        # --- Load environment variables from project root ---
        ROOT_DIR = Path(__file__).resolve().parents[2]   # go up 2 levels: src/agents -> src -> project root
        ENV_PATH = ROOT_DIR / ".env"
        load_dotenv(dotenv_path=ENV_PATH)
        
        # Initialize investment data
        self.investment_data: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.SUPPORTED_TYPES}
        self._load_investment_data()
        
        # Risk and age-based recommendations
        self.age_based_allocation = {
            "young_adult": (18, 30, {"stock": 70, "mutual fund": 20, "fixed deposit": 10, "gold": 0}),
            "early_career": (30, 40, {"stock": 60, "mutual fund": 25, "fixed deposit": 10, "gold": 5}),
            "mid_career": (40, 50, {"stock": 50, "mutual fund": 30, "fixed deposit": 15, "gold": 5}),
            "pre_retirement": (50, 60, {"stock": 30, "mutual fund": 40, "fixed deposit": 25, "gold": 5}),
            "retirement": (60, 100, {"stock": 20, "mutual fund": 30, "fixed deposit": 40, "gold": 10})
        }
        
        self.risk_tolerance_mapping = {
            "conservative": {"stock": 20, "mutual fund": 30, "fixed deposit": 40, "gold": 10},
            "moderate": {"stock": 50, "mutual fund": 30, "fixed deposit": 15, "gold": 5},
            "aggressive": {"stock": 70, "mutual fund": 20, "fixed deposit": 5, "gold": 5}
        }
        
        # Build the graph
        self.graph = self._build_graph()

    def _load_investment_data(self):
        """Load investment data from XML file"""
        xml_path = self.config.xml_path
        if not os.path.exists(xml_path):
            logger.warning("XML not found at %s — investment search will be empty.", xml_path)
            return

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Reset data
            self.investment_data = {t: [] for t in self.SUPPORTED_TYPES}
            
            # Parse different offer types
            for stock_offer in root.findall("StockOffer"):
                parsed = self._parse_stock_xml(stock_offer)
                if parsed:
                    self.investment_data["stock"].append(parsed)
                    
            for mf_offer in root.findall("MutualFundOffer"):
                parsed = self._parse_mutual_fund_xml(mf_offer)
                if parsed:
                    self.investment_data["mutual fund"].append(parsed)
                    
            for fd_offer in root.findall("FDOffer"):
                parsed = self._parse_fd_xml(fd_offer)
                if parsed:
                    self.investment_data["fixed deposit"].append(parsed)
                    
            for gold_offer in root.findall("GoldOffer"):
                parsed = self._parse_gold_xml(gold_offer)
                if parsed:
                    self.investment_data["gold"].append(parsed)
                    
            logger.info("Loaded investment data: %s total offers", sum(len(v) for v in self.investment_data.values()))
            for inv_type, offers in self.investment_data.items():
                if offers:
                    logger.info("  - %s: %s offers", inv_type, len(offers))
                    
        except Exception as e:
            logger.exception("Error loading XML: %s", e)

    def _parse_stock_xml(self, stock_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse stock offer XML element"""
        try:
            name = stock_elem.get("name", "Unknown Stock")
            
            def get_float(tag: str, default: float = 0.0) -> float:
                text = self._parse_xml_safely(stock_elem, tag)
                return self._parse_num(text, float) if text else default
                
            def get_percentage(tag: str, default: float = 0.0) -> float:
                text = self._parse_xml_safely(stock_elem, tag)
                if text and '%' in text:
                    return self._parse_num(text.replace('%', ''), float)
                return default
            
            stock = {
                "name": name,
                "type": "stock",
                "price": get_float("Price"),
                "market_cap": self._get_element_text(stock_elem, "MarketCap"),
                "pe_ratio": get_float("PE_Ratio"),
                "dividend_yield": get_percentage("DividendYield"),
                "return_1_year": get_percentage("Return1Year"),
                "week_52_high": get_float("52WeekHigh"),
                "week_52_low": get_float("52WeekLow"),
                "sector": self._get_element_text(stock_elem, "Sector"),
            }
            return stock
        except Exception as e:
            print(f"[investment_coach] Error parsing stock: {e}")
            return None

    def _parse_mutual_fund_xml(self, mf_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse mutual fund offer XML element"""
        try:
            name = mf_elem.get("name", "Unknown Fund")
            
            def get_float(tag: str, default: float = 0.0) -> float:
                text = self._get_element_text(mf_elem, tag)
                return self._parse_num(text, float) if text else default
                
            def get_percentage(tag: str, default: float = 0.0) -> float:
                text = self._get_element_text(mf_elem, tag)
                if text and '%' in text:
                    return self._parse_num(text.replace('%', ''), float)
                return default
            
            fund = {
                "name": name,
                "type": "mutual fund",
                "nav": get_float("NAV"),
                "expense_ratio": get_percentage("ExpenseRatio"),
                "aum": self._get_element_text(mf_elem, "AUM"),
                "category": self._get_element_text(mf_elem, "Category"),
                "return_1_year": get_percentage("Return1Year"),
                "return_3_year": get_percentage("Return3Year"),
                "return_5_year": get_percentage("Return5Year"),
                "risk_level": self._get_element_text(mf_elem, "RiskLevel"),
            }
            return fund
        except Exception as e:
            print(f"[investment_coach] Error parsing mutual fund: {e}")
            return None

    def _parse_fd_xml(self, fd_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse fixed deposit offer XML element"""
        try:
            name = fd_elem.get("name", "Unknown FD")
            
            def get_float(tag: str, default: float = 0.0) -> float:
                text = self._get_element_text(fd_elem, tag)
                return self._parse_num(text, float) if text else default
                
            def get_percentage(tag: str, default: float = 0.0) -> float:
                text = self._get_element_text(fd_elem, tag)
                if text and '%' in text:
                    return self._parse_num(text.replace('%', ''), float)
                return default
            
            fd = {
                "name": name,
                "type": "fixed deposit",
                "interest_rate": get_percentage("InterestRate"),
                "min_amount": get_float("MinAmount"),
                "max_amount": get_float("MaxAmount"),
                "tenure": self._get_element_text(fd_elem, "Tenure"),
                "premature_withdrawal": self._get_element_text(fd_elem, "PrematureWithdrawal"),
                "compounding_frequency": self._get_element_text(fd_elem, "CompoundingFrequency"),
                "senior_citizen_benefit": self._get_element_text(fd_elem, "SeniorCitizenBenefit"),
            }
            return fd
        except Exception as e:
            print(f"[investment_coach] Error parsing FD: {e}")
            return None

    def _parse_gold_xml(self, gold_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse gold offer XML element"""
        try:
            name = gold_elem.get("name", "Unknown Gold")
            
            def get_float(tag: str, default: float = 0.0) -> float:
                text = self._get_element_text(gold_elem, tag)
                return self._parse_num(text, float) if text else default
                
            def get_percentage(tag: str, default: float = 0.0) -> float:
                text = self._get_element_text(gold_elem, tag)
                if text and '%' in text:
                    return self._parse_num(text.replace('%', ''), float)
                return default
            
            gold = {
                "name": name,
                "type": "gold",
                "rate_per_gram": get_float("RatePerGram"),
                "purity": self._get_element_text(gold_elem, "Purity"),
                "loan_to_value": get_percentage("LoanToValue"),
                "tenure": self._get_element_text(gold_elem, "Tenure"),
                "collateral_required": self._get_element_text(gold_elem, "CollateralRequired"),
                "processing_fee": get_float("ProcessingFee"),
                "eligibility": self._get_element_text(gold_elem, "Eligibility"),
            }
            return gold
        except Exception as e:
            print(f"[investment_coach] Error parsing gold: {e}")
            return None

    def _get_element_text(self, parent: ET.Element, tag_name: str, default: str = "") -> str:
        """Safely get text from XML element"""
        element = parent.find(tag_name)
        return element.text if element is not None and element.text is not None else default

    def _parse_xml_safely(self, parent: ET.Element, tag_name: str, default: str = "") -> str:
        """Helper to find an element even if the tag name is not a valid XML identifier."""
        for child in parent:
            if child.tag == tag_name:
                return child.text if child.text is not None else default
        return default

    def _parse_num(self, text: Optional[str], f=float) -> float:
        """Parse numeric value from string"""
        if text is None:
            return 0.0
        try:
            s = re.sub(r"[^0-9.]+", "", str(text))
            if s == "":
                return 0.0
            return f(s)
        except Exception:
            return 0.0

    # ==================== LANGGRAPH NODES ====================

    async def _node_initialize_analysis(self, state: InvestmentState) -> InvestmentState:
        """Initialize investment analysis - extract user data and query info"""
        try:
            # Extract user information
            profile = state.get("profile", {}) or {}
            normalized = self._normalize_profile(profile)
            user_name = normalized["name"]
            age = normalized["age"]
            income = normalized["income"]
            risk_tolerance = normalized["risk_tolerance"]
            investment_horizon = normalized["investment_horizon"]
            
            # Analyze current investment spending from transactions
            transactions = state.get("transactions", [])
            investment_spending = self._analyze_investment_transactions(transactions)
            
            # Extract criteria from query using NLP
            query = state.get("query", "")
            criteria = await self._extract_criteria_from_query_nlp(query)
            
            # Parse query for investment types
            query_info = self._parse_investment_query(query.lower())
            
            # Update state
            new_state = dict(state)
            new_state.update({
                "user_name": user_name,
                "age": age,
                "income": income,
                "risk_tolerance": risk_tolerance,
                "investment_horizon": investment_horizon,
                "investment_spending": investment_spending,
                "criteria": criteria,
                "query_info": query_info,
                "error": None
            })
            
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Initialization error: {str(e)}"
            return new_state

    async def _node_determine_mode(self, state: InvestmentState) -> InvestmentState:
        """Determine analysis mode: investment search vs portfolio analysis with improved detection"""
        try:
            criteria = state.get("criteria", {})
            query = state.get("query", "").lower()
            
            # Investment Search Mode indicators:
            search_indicators = [
                # Has specific investment type AND specific constraints
                bool(criteria.get('investment_type') and (
                    'min_amount' in criteria or 'max_amount' in criteria or 
                    'price' in criteria or 'min_return' in criteria or 'max_risk' in criteria
                )),
                
                # Query contains specific search terms
                any(term in query for term in [
                    'suggest', 'recommend', 'find', 'search', 'list', 'show me',
                    'under', 'below', 'above', 'over', 'less than', 'more than',
                    'priced', 'with price', 'budget', 'cheap', 'expensive',
                    'best performing', 'high return', 'low risk'
                ]),
                
                # Query has specific numerical constraints
                bool(re.search(r'\d+(?:,\d+)*(?:\.\d+)?', query)),
                
                # Query asks for specific products
                any(phrase in query for phrase in [
                    'stocks under', 'funds above', 'fd with', 'gold under',
                    'stocks below', 'funds over', 'stocks priced', 'cheap stocks',
                    'high return', 'best stocks', 'good mutual funds'
                ])
            ]
            
            # Portfolio Analysis Mode indicators:
            portfolio_indicators = [
                any(term in query for term in [
                    'portfolio', 'allocation', 'diversify', 'balance', 'overall',
                    'financial planning', 'investment strategy', 'how much to invest',
                    'what should i invest', 'investment advice', 'where to invest'
                ]),
                
                # General questions without specific constraints
                not bool(criteria.get('investment_type')) and not any(search_indicators),
                
                # Query asks for general advice
                any(phrase in query for phrase in [
                    'should i invest', 'how to invest', 'investment plan',
                    'financial advice', 'money management', 'wealth building'
                ])
            ]
            
            # Determine mode based on indicators
            if any(search_indicators) and criteria.get('investment_type'):
                analysis_mode = "investment_search"
                print(f"Selected investment_search mode - criteria: {criteria}")
            elif any(portfolio_indicators):
                analysis_mode = "portfolio_analysis"
                print(f"Selected portfolio_analysis mode")
            else:
                # Default fallback - if we have specific investment type, prefer search
                if criteria.get('investment_type'):
                    analysis_mode = "investment_search"
                    print(f"Fallback to investment_search mode - has investment type: {criteria.get('investment_type')}")
                else:
                    analysis_mode = "portfolio_analysis"
                    print(f"Fallback to portfolio_analysis mode")
            
            new_state = dict(state)
            new_state["analysis_mode"] = analysis_mode
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Mode determination error: {str(e)}"
            new_state["analysis_mode"] = "portfolio_analysis"  # Safe fallback
            return new_state

    async def _node_investment_search(self, state: InvestmentState) -> InvestmentState:
        """Search for specific investment products based on criteria"""
        try:
            criteria = state.get("criteria", {})
            investments = self._search_investments(criteria)
            
            if not investments:
                summary = f"No {criteria.get('investment_type', 'investment')} products found matching your criteria: {criteria}"
                next_best_actions = [
                    "Try adjusting your return expectations",
                    "Consider different investment types",
                    "Diversify across asset classes",
                    "Consult with a financial advisor"
                ]
            else:
                # Format investment search results
                investment_type = criteria.get('investment_type', 'investment')
                summary_parts = [f"Found {len(investments)} {investment_type} investment options matching your criteria:\n"]
                
                for i, inv in enumerate(investments, 1):
                    summary_parts.append(f"\n{i}. **{inv['name']}**")
                    
                    if investment_type == 'stock':
                        summary_parts.append(f"   - Price: ₹{inv.get('price', 0):,.2f}")
                        summary_parts.append(f"   - Market Cap: {inv.get('market_cap', 'N/A')}")
                        summary_parts.append(f"   - PE Ratio: {inv.get('pe_ratio', 0):g}")
                        summary_parts.append(f"   - Dividend Yield: {inv.get('dividend_yield', 0):g}%")
                        summary_parts.append(f"   - 1Y Return: {inv.get('return_1_year', 0):g}%")
                        summary_parts.append(f"   - Sector: {inv.get('sector', 'N/A')}")
                    elif investment_type == 'mutual fund':
                        summary_parts.append(f"   - NAV: ₹{inv.get('nav', 0):,.2f}")
                        summary_parts.append(f"   - Expense Ratio: {inv.get('expense_ratio', 0):g}%")
                        summary_parts.append(f"   - Category: {inv.get('category', 'N/A')}")
                        summary_parts.append(f"   - 1Y Return: {inv.get('return_1_year', 0):g}%")
                        summary_parts.append(f"   - 3Y Return: {inv.get('return_3_year', 0):g}%")
                        summary_parts.append(f"   - Risk Level: {inv.get('risk_level', 'N/A')}")
                    elif investment_type == 'fixed deposit':
                        summary_parts.append(f"   - Interest Rate: {inv.get('interest_rate', 0):g}%")
                        summary_parts.append(f"   - Min Amount: ₹{int(inv.get('min_amount', 0)):,}")
                        summary_parts.append(f"   - Max Amount: ₹{int(inv.get('max_amount', 0)):,}")
                        summary_parts.append(f"   - Tenure: {inv.get('tenure', 'N/A')}")
                        summary_parts.append(f"   - Premature Withdrawal: {inv.get('premature_withdrawal', 'N/A')}")
                    elif investment_type == 'gold':
                        summary_parts.append(f"   - Rate per gram: ₹{inv.get('rate_per_gram', 0):,.2f}")
                        summary_parts.append(f"   - Purity: {inv.get('purity', 'N/A')}")
                        summary_parts.append(f"   - Loan to Value: {inv.get('loan_to_value', 0):g}%")
                        summary_parts.append(f"   - Processing Fee: ₹{int(inv.get('processing_fee', 0)):,}")
                
                summary_parts.append(f"\nSearch criteria: {criteria}")
                summary = "".join(summary_parts)
                
                next_best_actions = [
                    "Compare expense ratios and fees across options",
                    "Check historical performance over multiple years",
                    "Review fund manager track record and strategy",
                    "Ensure investments align with your risk profile",
                    "Consider tax implications before investing"
                ]
            
            new_state = dict(state)
            new_state.update({
                "investments": investments,
                "summary": summary,
                "next_best_actions": next_best_actions
            })
            
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Investment search error: {str(e)}"
            return new_state

    async def _node_portfolio_analysis(self, state: InvestmentState) -> InvestmentState:
        """Analyze investment needs based on user profile"""
        try:
            # Extract state variables
            age = state.get("age", self.config.default_age)
            income = state.get("income", self.config.default_income)
            risk_tolerance = state.get("risk_tolerance", "moderate")
            user_name = state.get("user_name", "user")
            investment_spending = state.get("investment_spending", {})
            query_info = state.get("query_info", {})
            
            # Assess investment needs based on profile
            investment_needs = self._assess_investment_needs(age, income, risk_tolerance)
            
            # Get risk profile
            risk_profile = self._get_risk_profile(age, income, risk_tolerance)
            
            # Generate portfolio allocation recommendations
            allocation = self._get_portfolio_allocation(age, risk_tolerance)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                investment_needs, investment_spending, query_info, income, risk_tolerance, allocation
            )
            
            # Create summary
            summary = self._create_summary(
                user_name, age, income, investment_needs, investment_spending, query_info, allocation
            )
            
            # Create investment analysis object
            investment_analysis = {
                "current_spending": investment_spending,
                "recommended_types": investment_needs,
                "risk_profile": risk_profile,
                "suggested_allocation": allocation
            }
            
            new_state = dict(state)
            new_state.update({
                "investment_needs": investment_needs,
                "risk_profile": risk_profile,
                "summary": summary,
                "next_best_actions": recommendations,
                "investment_analysis": investment_analysis
            })
            
            return new_state
            
        except Exception as e:
            new_state = dict(state)
            new_state["error"] = f"Portfolio analysis error: {str(e)}"
            return new_state

    # ==================== HELPER METHODS ====================

    def _analyze_investment_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transactions to find current investment spending"""
        investment_spend = {"stocks": 0, "mutual_funds": 0, "fixed_deposits": 0, "gold": 0, "other": 0, "total": 0}
        investment_keywords = {
            "stocks": ["stock", "equity", "share", "dividend"],
            "mutual_funds": ["mutual fund", "mf", "sip", "nav"],
            "fixed_deposits": ["fd", "fixed deposit", "term deposit"],
            "gold": ["gold", "precious metal"],
            "other": ["investment", "invest"]
        }

        for txn in transactions:
            description = txn.get("description", "").lower()
            category = txn.get("category", "").lower() 
            amount = txn.get("amount", 0)
            
            if any(keyword in description or keyword in category for keywords in investment_keywords.values() for keyword in keywords):
                categorized = False
                for inv_type, keywords in investment_keywords.items():
                    if inv_type != "other" and any(keyword in description for keyword in keywords):
                        investment_spend[inv_type] += amount
                        categorized = True
                        break
                if not categorized:
                    investment_spend["other"] += amount
                investment_spend["total"] += amount

        return investment_spend

    async def _extract_criteria_from_query_nlp(self, query: str) -> Dict[str, Any]:
        """Extract investment criteria from a natural language query with fallback"""
        try:
            # Load API key from environment variable
            apiKey = os.getenv('LLM_API_KEY')
            
            if not apiKey:
                print("API key not found, using fallback extraction")
                return self._extract_criteria_fallback(query)

            # Use OpenAI API endpoint
            apiUrl = os.getenv("BASE_URL", "https://api.openai.com/v1/").rstrip("/") + "/chat/completions"
            
            system_prompt = (
                "You are an expert financial assistant. Your task is to analyze a user's query and extract specific investment criteria. "
                "You must respond with a JSON object. "
                "The object should contain 'investment_type' (one of: 'stock', 'mutual fund', 'fixed deposit', 'gold', or null), "
                "'min_amount' (integer or null), 'max_amount' (integer or null), 'min_return' (float or null), "
                "'max_risk' (one of: 'low', 'moderate', 'high', or null), and 'error' (string or null). "
                "If a specific value for a criterion cannot be found, use 'null'. Do not include any other text."
            )
            
            payload = {
                "model": os.getenv("CUSTOM_MODEL_NAME", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                "response_format": {"type": "json_object"}
            }

            retries = 2  # Reduced retries to fail faster
            for i in range(retries):
                try:
                    # Use asyncio.to_thread for the blocking requests call
                    response = await asyncio.to_thread(
                        requests.post,
                        apiUrl,
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {apiKey}'
                        },
                        data=json.dumps(payload),
                        timeout=10  # Shorter timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                        json_str = result['choices'][0]['message']['content']
                        parsed_json = json.loads(json_str)
                        return parsed_json
                    else:
                        error_message = result.get('error', {}).get('message', 'Unknown API error')
                        print(f"Gemini API error: {error_message}")
                        break
                        
                except requests.exceptions.RequestException as e:
                    print(f"Gemini API call failed (attempt {i+1}): {e}")
                    if i < retries - 1:
                        await asyncio.sleep(1)  # Shorter delay
                    else:
                        break

        except Exception as e:
            print(f"NLP extraction error: {str(e)}")

        # Fallback to rule-based extraction
        print("Using fallback criteria extraction")
        return self._extract_criteria_fallback(query)

    def _extract_criteria_fallback(self, query: str) -> Dict[str, Any]:
        """Fallback criteria extraction when Gemini API is unavailable"""
        criteria = {}
        
        # Extract investment type
        query_lower = query.lower()
        if any(term in query_lower for term in ['stock', 'equity', 'share']):
            criteria['investment_type'] = 'stock'
        elif any(term in query_lower for term in ['mutual fund', 'mf', 'sip']):
            criteria['investment_type'] = 'mutual fund'
        elif any(term in query_lower for term in ['fd', 'fixed deposit']):
            criteria['investment_type'] = 'fixed deposit'
        elif 'gold' in query_lower:
            criteria['investment_type'] = 'gold'
        
        # Extract price/amount constraints with better patterns
        price_patterns = [
            r"(?:price|priced)\s+(?:under|below|less than|cheaper than)\s*(?:₹|rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?:under|below|less than|cheaper than)\s*(?:₹|rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)",
            r"(?:budget|amount)\s*(?:of|is|around)?\s*(?:₹|rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    amount = float(price_str)
                    criteria['max_amount'] = amount
                    if criteria.get('investment_type') == 'stock':
                        criteria['price'] = amount
                    break
                except ValueError:
                    continue
        
        # Extract minimum amount patterns
        min_patterns = [
            r"(?:above|over|more than|higher than|minimum)\s*(?:₹|rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)",
        ]
        
        for pattern in min_patterns:
            match = re.search(pattern, query_lower)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    criteria['min_amount'] = float(price_str)
                    break
                except ValueError:
                    continue
        
        # Extract return expectations
        return_patterns = [
            r"(?:return|returns?|yield)\s*(?:above|over|more than|higher than|at least)\s*(\d+(?:\.\d+)?)%?",
            r"(\d+(?:\.\d+)?)%\s*(?:return|returns?|yield)",
        ]
        
        for pattern in return_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    criteria['min_return'] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract risk tolerance
        if any(term in query_lower for term in ['safe', 'conservative', 'low risk', 'secure']):
            criteria['max_risk'] = 'low'
        elif any(term in query_lower for term in ['aggressive', 'high risk', 'risky', 'growth']):
            criteria['max_risk'] = 'high'
        else:
            criteria['max_risk'] = 'moderate'
        
        logger.debug("Extracted criteria: %s", criteria)
        return criteria

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

        risk_tolerance = (profile.get("risk_tolerance") or "moderate").strip().lower()
        if risk_tolerance not in self.risk_tolerance_mapping:
            risk_tolerance = "moderate"

        investment_horizon = (profile.get("investment_horizon") or "medium").strip().lower()

        age = max(0, age)
        income = max(0, income)

        return {
            "name": name,
            "age": age,
            "income": income,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon,
        }
    def _parse_investment_query(self, query: str) -> Dict[str, Any]:
        """Parse user query for specific investment requests"""
        query_info = {
            "type": None,
            "amount": None,
            "return_expectation": None,
            "specific_request": False
        }

        # Check for investment types
        if any(term in query for term in ['stock', 'equity', 'share']):
            query_info["type"] = "stock"
            query_info["specific_request"] = True
        elif any(term in query for term in ['mutual fund', 'mf', 'sip']):
            query_info["type"] = "mutual fund"
            query_info["specific_request"] = True
        elif any(term in query for term in ['fd', 'fixed deposit', 'term deposit']):
            query_info["type"] = "fixed deposit"
            query_info["specific_request"] = True
        elif 'gold' in query:
            query_info["type"] = "gold"
            query_info["specific_request"] = True

        return query_info

    def _search_investments(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search investments based on criteria"""
        investment_type = criteria.get('investment_type', 'mutual fund')
        all_investments = self.investment_data.get(investment_type, [])
        
        if not all_investments:
            return []
        
        filtered = []
        for investment in all_investments:
            # Check minimum return requirement
            if 'min_return' in criteria:
                if investment_type == 'stock' and investment.get('return_1_year', 0) < criteria['min_return']:
                    continue
                elif investment_type == 'mutual fund' and investment.get('return_1_year', 0) < criteria['min_return']:
                    continue
                elif investment_type == 'fixed deposit' and investment.get('interest_rate', 0) < criteria['min_return']:
                    continue
                elif investment_type == 'gold' and investment.get('loan_to_value', 0) < criteria['min_return']:
                    continue
            
            # Check amount/price requirements
            if 'min_amount' in criteria:
                if investment_type == 'fixed deposit' and investment.get('min_amount', 0) > criteria['min_amount']:
                    continue
            
            if 'max_amount' in criteria:
                if investment_type == 'stock' and investment.get('price', 0) > criteria['max_amount']:
                    continue
                if investment_type == 'fixed deposit' and investment.get('min_amount', 0) > criteria['max_amount']:
                    continue
            
            if 'price' in criteria and investment_type == 'stock':
                if investment.get('price', 0) > criteria['price'] + 10 or investment.get('price', 0) < criteria['price'] - 10:
                    continue
            
            # Check risk tolerance
            if 'max_risk' in criteria and investment_type == 'mutual fund':
                inv_risk = investment.get('risk_level', 'moderate').lower()
                max_risk = criteria['max_risk'].lower()
                if max_risk == 'low' and inv_risk not in ['low']:
                    continue
                elif max_risk == 'moderate' and inv_risk not in ['low', 'moderate']:
                    continue
            
            filtered.append(investment)
        
        # Sort based on investment type
        if investment_type == 'stock':
            filtered.sort(key=lambda x: x.get('return_1_year', 0), reverse=True)
        elif investment_type == 'mutual fund':
            filtered.sort(key=lambda x: (x.get('return_3_year', 0), -x.get('expense_ratio', 100)), reverse=True)
        elif investment_type == 'fixed deposit':
            filtered.sort(key=lambda x: x.get('interest_rate', 0), reverse=True)
        elif investment_type == 'gold':
            filtered.sort(key=lambda x: -x.get('processing_fee', 0))
            
        return filtered[:self.config.max_investments]

    def _assess_investment_needs(self, age: int, income: int, risk_tolerance: str) -> List[str]:
        """Assess investment needs based on user profile"""
        needs = []
        
        # Age-based recommendations
        if age < 30:
            needs.extend(["equity mutual funds", "large cap stocks", "ELSS funds"])
        elif age < 40:
            needs.extend(["balanced funds", "mid cap stocks", "PPF"])
        elif age < 50:
            needs.extend(["debt funds", "blue chip stocks", "fixed deposits"])
        else:
            needs.extend(["fixed deposits", "government bonds", "conservative funds"])
        
        # Income-based adjustments
        if income > 1000000:
            needs.append("direct equity")
            needs.append("real estate")
        
        # Risk tolerance adjustments
        if risk_tolerance == "aggressive":
            needs.extend(["small cap stocks", "sector funds"])
        elif risk_tolerance == "conservative":
            needs = [n for n in needs if "cap stocks" not in n or "blue chip" in n]
            needs.extend(["bank FDs", "government securities"])
        
        return list(set(needs))

    def _get_risk_profile(self, age: int, income: int, risk_tolerance: str) -> str:
        """Determine overall risk profile for investment planning"""
        risk_score = 0
        
        # Age factor (younger = more risk capacity)
        if age < 30:
            risk_score += 3
        elif age < 45:
            risk_score += 2
        else:
            risk_score += 1
            
        # Income factor
        if income < 300000:
            risk_score += 1
        elif income > 1000000:
            risk_score += 3
        else:
            risk_score += 2
            
        # Risk tolerance factor
        tolerance_scores = {"conservative": 1, "moderate": 2, "aggressive": 3}
        risk_score += tolerance_scores.get(risk_tolerance.lower(), 2)
        
        if risk_score <= 3:
            return "Conservative"
        elif risk_score <= 6:
            return "Moderate"
        else:
            return "Aggressive"

    def _get_portfolio_allocation(self, age: int, risk_tolerance: str) -> Dict[str, int]:
        """Get recommended portfolio allocation based on age and risk tolerance"""
        # Start with age-based allocation
        allocation = {"stock": 30, "mutual fund": 30, "fixed deposit": 30, "gold": 10}
        
        for category, (min_age, max_age, age_allocation) in self.age_based_allocation.items():
            if min_age <= age <= max_age:
                allocation = age_allocation.copy()
                break
        
        # Adjust based on risk tolerance
        risk_adjustment = self.risk_tolerance_mapping.get(risk_tolerance.lower(), {})
        if risk_adjustment:
            # Blend age-based and risk-based allocations (60% age, 40% risk)
            for asset_type in allocation:
                allocation[asset_type] = int(0.6 * allocation[asset_type] + 0.4 * risk_adjustment.get(asset_type, 25))
        
        return allocation

    def _create_summary(self, name: str, age: int, income: int, needs: List[str], 
                       spending: Dict[str, Any], query_info: Dict[str, Any], 
                       allocation: Dict[str, int]) -> str:
        """Create comprehensive summary of investment analysis"""
        total_spending = spending.get("total", 0)
        summary = f"Hello {name}, "
        
        if query_info.get("specific_request"):
            investment_type = query_info["type"]
            summary += f"I've analyzed your {investment_type} investment requirements. "
        else:
            summary += f"I've analyzed your investment profile (age: {age}, income: ₹{income:,}). "
        
        if total_spending > 0:
            summary += f"You're currently investing ₹{total_spending:,} annually. "
        else:
            summary += "You don't appear to have significant investment exposure currently. "
        
        # Investment recommendations
        recommended_allocation = int(income * 0.20)  # 20% of income for investments
        if total_spending < recommended_allocation * 0.5:
            summary += f"Consider allocating ₹{recommended_allocation:,} annually (20% of income) for investments. "
        
        # Portfolio allocation guidance
        summary += f"Recommended portfolio allocation: "
        allocation_text = ", ".join([f"{asset_type.replace('_', ' ').title()}: {percent}%" 
                                   for asset_type, percent in allocation.items() if percent > 0])
        summary += allocation_text + ". "
        
        if needs:
            summary += f"Priority investment areas: {', '.join(needs[:3])}."
        
        return summary

    def _generate_recommendations(self, needs: List[str], current_spending: Dict[str, Any],
                                 query_info: Dict[str, Any], income: int, risk_tolerance: str,
                                 allocation: Dict[str, int]) -> List[str]:
        """Generate actionable investment recommendations"""
        recommendations = []
        
        # Budget recommendation
        recommended_investment = int(income * 0.20)  # 20% of income
        current_total = current_spending.get("total", 0)
        
        if current_total < recommended_investment * 0.5:
            recommendations.append(f"Start systematic investing with ₹{recommended_investment:,} annually (20% of income)")

        # Specific type recommendations based on query
        if query_info.get("specific_request"):
            investment_type = query_info["type"]
            if investment_type == "stock":
                recommendations.extend([
                    "Focus on blue-chip stocks with consistent dividend history",
                    "Diversify across sectors to reduce concentration risk",
                    "Consider dollar-cost averaging for volatile stocks"
                ])
            elif investment_type == "mutual fund":
                recommendations.extend([
                    "Start with large-cap equity funds for stability",
                    "Set up SIP to benefit from rupee cost averaging",
                    "Review fund performance annually and rebalance"
                ])
            elif investment_type == "fixed deposit":
                recommendations.extend([
                    "Compare interest rates across banks and NBFCs",
                    "Consider laddering FDs for liquidity management",
                    "Evaluate tax implications on FD interest"
                ])
            elif investment_type == "gold":
                recommendations.extend([
                    "Consider gold ETFs over physical gold for convenience",
                    "Limit gold allocation to 5-10% of total portfolio",
                    "Use gold as hedge against market volatility"
                ])
        else:
            # General recommendations based on allocation
            if allocation.get("stock", 0) > 30:
                recommendations.append("Build equity portfolio gradually through SIP route")
            if allocation.get("mutual fund", 0) > 20:
                recommendations.append("Start with diversified equity mutual funds")
            if allocation.get("fixed deposit", 0) > 20:
                recommendations.append("Secure guaranteed returns through bank FDs and bonds")

        # Risk-specific recommendations
        if risk_tolerance.lower() == "conservative":
            recommendations.append("Focus on capital preservation with moderate growth")
        elif risk_tolerance.lower() == "aggressive":
            recommendations.append("Consider small-cap funds and growth stocks for higher returns")

        # General best practices
        recommendations.extend([
            "Diversify across asset classes to manage risk",
            "Review and rebalance portfolio annually",
            "Keep emergency fund before investing in risky assets",
            "Consider tax-saving investments like ELSS and PPF"
        ])

        return recommendations[:8]  # Limit to top 8 recommendations

    # ==================== CONDITIONAL EDGES ====================

    def _should_route_to_mode(self, state: InvestmentState) -> str:
        """Route to appropriate analysis mode"""
        if state.get("error"):
            return "end"
        
        analysis_mode = state.get("analysis_mode", "portfolio_analysis")
        return analysis_mode

    # ==================== GRAPH CONSTRUCTION ====================

    def _build_graph(self):
        """Build the LangGraph state graph"""
        graph = StateGraph(InvestmentState)
        
        # Add nodes
        graph.add_node("initialize_analysis", self._node_initialize_analysis)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("investment_search", self._node_investment_search)
        graph.add_node("portfolio_analysis", self._node_portfolio_analysis)
        
        # Add edges
        graph.add_edge(START, "initialize_analysis")
        graph.add_edge("initialize_analysis", "determine_mode")
        
        # Conditional routing based on analysis mode
        graph.add_conditional_edges(
            "determine_mode",
            self._should_route_to_mode,
            {
                "investment_search": "investment_search",
                "portfolio_analysis": "portfolio_analysis",
                "end": END
            }
        )
        
        # Both analysis modes go to END
        graph.add_edge("investment_search", END)
        graph.add_edge("portfolio_analysis", END)
        
        return graph.compile()

    # ==================== PUBLIC INTERFACE ====================

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the investment analysis LangGraph
        
        Args:
            state: Input state with query, profile, transactions
            
        Returns:
            Analysis results with summary and recommendations
        """
        try:
            # Convert input to InvestmentState
            investment_state: InvestmentState = {
                "query": state.get("query", ""),
                "profile": state.get("profile", {}),
                "transactions": state.get("transactions", [])
            }
            
            # Run the graph
            final_state = await self.graph.ainvoke(investment_state)
            
            # Format output for compatibility with existing system
            result = {
                "summary": final_state.get("summary", "Analysis completed"),
                "next_best_actions": final_state.get("next_best_actions", []),
                "mode": final_state.get("analysis_mode", "unknown"),
            }
            
            # Add mode-specific data
            if final_state.get("analysis_mode") == "investment_search":
                result.update({
                    "investments": final_state.get("investments", []),
                    "criteria": final_state.get("criteria", {})
                })
            else:
                result["investment_analysis"] = final_state.get("investment_analysis", {})
            
            return result
            
        except Exception as e:
            return {
                "summary": f"Investment analysis error: {str(e)}",
                "next_best_actions": ["Please try again with a different query"],
                "mode": "error"
            }

# Create the main class for compatibility
class InvestmentCoach:
    """LangGraph-based Investment Coach - Compatible with existing orchestrator"""

    def __init__(self):
        self.coach_graph = InvestmentCoachGraph()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run investment analysis with intent-aware routing

        Flow:
        1. Check query intent (general vs personalized)
        2. For general queries: Provide educational investment information
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

        # Handle general queries - provide investment education without personalization
        if is_general_query(query, intent):
            return self._provide_general_investment_info(query)

        # For personalized queries, check profile completeness
        required_fields = ["age", "income", "risk_tolerance"]
        missing_fields = ProfileValidator.get_missing_fields(profile, required_fields)

        if missing_fields:
            # Generate request for missing information
            message = ProfileValidator.generate_missing_fields_message(
                missing_fields,
                agent_name="Investment Coach",
                query_context="provide personalized investment recommendations"
            )
            return {
                "summary": message,
                "next_best_actions": [f"Provide your {field.replace('_', ' ')}" for field in missing_fields],
                "mode": "awaiting_profile",
                "requires_input": True,
                "missing_fields": missing_fields
            }

        # Profile is complete, proceed with personalized analysis using LangGraph
        return asyncio.run(self.coach_graph.run(state))

    def _provide_general_investment_info(self, query: str) -> Dict[str, Any]:
        """Provide general investment information without personalization"""
        from backend.agents.agent_helpers import format_general_info_response

        query_lower = query.lower()

        # Determine what type of information to provide
        if any(word in query_lower for word in ["mutual fund", "mf", "sip"]):
            info = """**Mutual Funds: A Comprehensive Guide**

**What are Mutual Funds?**
Mutual funds pool money from multiple investors to invest in a diversified portfolio of stocks, bonds, or other securities. They're managed by professional fund managers.

**Types of Mutual Funds:**
• **Equity Funds**: Invest primarily in stocks (high risk, high potential returns)
• **Debt Funds**: Invest in bonds and fixed-income securities (lower risk, stable returns)
• **Hybrid Funds**: Mix of equity and debt (moderate risk)
• **Index Funds**: Track a market index like Nifty 50 or Sensex
• **ELSS (Tax-saving)**: Equity funds with 3-year lock-in, eligible for 80C deduction

**Key Benefits:**
• Professional management
• Diversification across multiple securities
• Liquidity (can redeem most funds anytime)
• Systematic Investment Plan (SIP) option
• Regulatory oversight by SEBI

**How to Choose:**
1. Define your investment goal (retirement, wealth creation, etc.)
2. Assess your risk tolerance
3. Consider investment horizon (short/medium/long-term)
4. Compare fund performance, expense ratio, and fund manager track record
5. Check exit load and lock-in periods"""

        elif any(word in query_lower for word in ["stock", "equity", "share"]):
            info = """**Stock Market Investing: Essential Information**

**What are Stocks?**
Stocks represent ownership shares in a company. When you buy stocks, you become a partial owner and can benefit from the company's growth.

**How Stock Markets Work:**
• **BSE (Bombay Stock Exchange)**: Oldest stock exchange in Asia
• **NSE (National Stock Exchange)**: Largest stock exchange in India by volume
• Stock prices fluctuate based on supply/demand, company performance, and market sentiment

**Types of Stocks:**
• **Large-cap**: Well-established companies (₹20,000+ crore market cap) - Lower risk
• **Mid-cap**: Growing companies (₹5,000-20,000 crore) - Moderate risk
• **Small-cap**: Emerging companies (<₹5,000 crore) - Higher risk

**Investment Approaches:**
• **Long-term investing**: Buy and hold quality stocks (5+ years)
• **Value investing**: Find undervalued companies
• **Growth investing**: Invest in high-growth companies
• **Dividend investing**: Focus on stocks with regular dividend payments

**Key Metrics to Evaluate:**
• P/E Ratio (Price-to-Earnings)
• EPS (Earnings Per Share)
• Debt-to-Equity ratio
• ROE (Return on Equity)
• Revenue and profit growth

**Risks:**
• Market volatility
• Company-specific risks
• Sector-specific risks
• Requires active monitoring"""

        elif any(word in query_lower for word in ["fd", "fixed deposit", "recurring deposit"]):
            info = """**Fixed Deposits (FDs): Safe Investment Option**

**What is a Fixed Deposit?**
FDs are time deposits offered by banks where you deposit money for a fixed period at a predetermined interest rate.

**Features:**
• **Tenure**: 7 days to 10 years
• **Interest rates**: 3-7% p.a. (varies by bank and tenure)
• **Safety**: Deposits up to ₹5 lakhs insured by DICGC
• **Guaranteed returns**: Interest rate locked at booking

**Types of FDs:**
• **Regular FD**: Standard fixed deposit
• **Tax-saver FD**: 5-year lock-in, eligible for 80C deduction (up to ₹1.5 lakhs)
• **Senior Citizen FD**: Higher interest rates (0.25-0.5% extra)
• **Cumulative FD**: Interest compounded and paid at maturity
• **Non-cumulative FD**: Interest paid monthly/quarterly/annually

**Interest Calculation:**
• Simple interest or compound interest
• TDS deducted if interest > ₹40,000/year (₹50,000 for senior citizens)

**Pros:**
• Low risk, capital protection
• Predictable returns
• Loan facility against FD (up to 90%)

**Cons:**
• Lower returns compared to equities
• Inflation may erode real returns
• Premature withdrawal penalty (0.5-1%)
• Interest fully taxable"""

        elif any(word in query_lower for word in ["gold", "sovereign gold bond", "sgb"]):
            info = """**Gold Investments: Traditional Meets Modern**

**Ways to Invest in Gold:**
1. **Physical Gold** (jewelry, coins, bars)
2. **Sovereign Gold Bonds (SGBs)** - Government-issued, earn 2.5% interest
3. **Gold ETFs** - Trade on stock exchanges like shares
4. **Gold Mutual Funds** - Invest in gold ETFs or mining companies
5. **Digital Gold** - Buy/sell small quantities online

**Sovereign Gold Bonds (SGB):**
• Issued by RBI, backed by Government of India
• Tenure: 8 years (exit option after 5 years)
• Interest: 2.5% p.a. paid semi-annually
• Tax benefit: No capital gains tax if held till maturity
• Traded on exchanges after issuance
• Minimum investment: 1 gram, maximum: 4 kg/year

**Gold ETFs:**
• 1 unit = 1 gram of gold
• Trade during market hours
• Lower costs than physical gold
• High liquidity
• Stored in Demat form

**Why Invest in Gold?**
• Hedge against inflation
• Portfolio diversification
• Traditionally safe during market volatility
• Cultural significance in India

**Considerations:**
• Gold doesn't generate income (except SGBs)
• Price volatility based on global markets
• Long-term returns historically lower than equities
• Optimal allocation: 5-10% of portfolio"""

        else:
            # General investment overview
            info = """**Investment Basics: Building Wealth Wisely**

**Key Investment Principles:**
1. **Start Early**: Benefit from compound interest over time
2. **Diversify**: Don't put all eggs in one basket
3. **Set Goals**: Define short-term and long-term financial objectives
4. **Assess Risk**: Understand your risk tolerance
5. **Stay Invested**: Avoid timing the market

**Common Investment Options in India:**

**Equity Investments** (High risk, high returns)
• Stocks (direct equity)
• Equity Mutual Funds
• Index Funds
• ELSS (tax-saving)

**Debt Investments** (Low-moderate risk, stable returns)
• Fixed Deposits
• Debt Mutual Funds
• Government Bonds
• Corporate Bonds
• PPF (Public Provident Fund)

**Hybrid Options**
• Balanced/Hybrid Mutual Funds
• Monthly Income Plans (MIPs)

**Alternative Investments**
• Gold (SGBs, ETFs, physical)
• Real Estate
• REITs (Real Estate Investment Trusts)

**Risk vs Return Spectrum:**
• **Conservative** (Low risk): FDs, PPF, Debt funds → 6-8% returns
• **Moderate** (Medium risk): Hybrid funds, Balanced portfolios → 8-12% returns
• **Aggressive** (High risk): Equity funds, Stocks → 12-15%+ returns (volatile)

**Investment Strategies by Age:**
• **20s-30s**: Higher equity allocation (70-80%)
• **40s-50s**: Balanced approach (50-60% equity)
• **60s+**: Conservative, focus on income (20-30% equity)

**Tax-Saving Investments (Section 80C):**
• ELSS Mutual Funds (3-year lock-in)
• PPF (15-year maturity)
• Tax-saver FDs (5-year lock-in)
• NPS (National Pension System)
• Life Insurance premiums

**Golden Rules:**
• Emergency fund first (6 months expenses)
• Clear high-interest debt before investing
• Review portfolio annually
• Avoid emotional decisions
• Consult SEBI-registered advisors for complex decisions"""

        formatted = format_general_info_response(info, "Investment Coach")

        return {
            "summary": formatted,
            "next_best_actions": [
                "Provide your age, income, and risk tolerance for personalized recommendations",
                "Ask specific questions about investment products",
                "Explore SIP (Systematic Investment Plan) options"
            ],
            "mode": "general_information"
        }
