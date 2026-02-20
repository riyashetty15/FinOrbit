"""
Shared Configuration Module
Provides centralized definitions used by both Finorbit_LLM and Finorbit_RAG

This module ensures consistency across the entire FinOrbit system.
"""

# ============================================================================
# Business Modules
# ============================================================================
# These are the core financial planning domains supported by the system
FINORBIT_MODULES = [
    "credit",
    "investment",
    "insurance",
    "retirement",
    "taxation"
]

# ============================================================================
# Module to Database Table Mapping
# ============================================================================
# Maps each module to its corresponding database table in the RAG vector store
FINORBIT_MODULE_TABLES = {
    "credit": "data_credit_chunks",
    "investment": "data_investment_chunks",
    "insurance": "data_insurance_chunks",
    "retirement": "data_retirement_chunks",
    "taxation": "data_tax_chunks"
}

# ============================================================================
# Module Keywords for Routing
# ============================================================================
# Keywords used to identify which module a user query belongs to
FINORBIT_MODULE_KEYWORDS = {
    "credit": ["credit", "loan", "emi", "cibil", "score", "borrow", "debt", "cibil"],
    "investment": ["invest", "portfolio", "mutual fund", "stock", "equity", "bond", "market"],
    "insurance": ["insurance", "premium", "coverage", "term plan", "health insurance", "policy"],
    "retirement": ["retire", "pension", "401k", "nps", "retirement planning", "age", "elderly"],
    "taxation": ["tax", "itr", "deduction", "exemption", "section", "income tax", "gst", "refund"]
}

# ============================================================================
# Module Descriptions
# ============================================================================
# Human-readable descriptions for each module
FINORBIT_MODULE_DESCRIPTIONS = {
    "credit": "Credit products, loans, EMI, credit scoring, and debt management",
    "investment": "Investment strategies, portfolio management, stocks, mutual funds, and bonds",
    "insurance": "Insurance products, coverage types, premiums, and policy management",
    "retirement": "Retirement planning, pension schemes, and long-term financial security",
    "taxation": "Income tax planning, deductions, exemptions, compliance, and tax optimization"
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_modules() -> list:
    """Get list of all supported modules"""
    return FINORBIT_MODULES.copy()


def get_module_table(module: str) -> str:
    """
    Get the database table name for a given module.
    
    Args:
        module: Module name (case-insensitive)
        
    Returns:
        Table name
        
    Raises:
        ValueError: If module is invalid
    """
    module = module.strip().lower()
    if module not in FINORBIT_MODULE_TABLES:
        raise ValueError(f"Invalid module: {module}. Valid modules: {', '.join(FINORBIT_MODULES)}")
    return FINORBIT_MODULE_TABLES[module]


def get_module_keywords(module: str) -> list:
    """
    Get keywords associated with a module for routing/classification.
    
    Args:
        module: Module name (case-insensitive)
        
    Returns:
        List of keywords
        
    Raises:
        ValueError: If module is invalid
    """
    module = module.strip().lower()
    if module not in FINORBIT_MODULE_KEYWORDS:
        raise ValueError(f"Invalid module: {module}. Valid modules: {', '.join(FINORBIT_MODULES)}")
    return FINORBIT_MODULE_KEYWORDS[module].copy()


def get_module_description(module: str) -> str:
    """
    Get human-readable description for a module.
    
    Args:
        module: Module name (case-insensitive)
        
    Returns:
        Description string
        
    Raises:
        ValueError: If module is invalid
    """
    module = module.strip().lower()
    if module not in FINORBIT_MODULE_DESCRIPTIONS:
        raise ValueError(f"Invalid module: {module}. Valid modules: {', '.join(FINORBIT_MODULES)}")
    return FINORBIT_MODULE_DESCRIPTIONS[module]


def is_valid_module(module: str) -> bool:
    """
    Check if a module name is valid.
    
    Args:
        module: Module name to validate (case-insensitive)
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not module or not isinstance(module, str):
        return False
    return module.strip().lower() in FINORBIT_MODULES


# ============================================================================
# Service Configuration
# ============================================================================
# Default service endpoints and timeouts
RAG_SERVICE_DEFAULT_URL = "http://localhost:8081"
RAG_QUERY_ENDPOINT = "/query"
RAG_HEALTH_ENDPOINT = "/health"

LLM_SERVICE_DEFAULT_URL = "http://localhost:8000"
LLM_QUERY_ENDPOINT = "/query"
LLM_HEALTH_ENDPOINT = "/health"

# Timeouts (in seconds)
RAG_HEALTH_CHECK_TIMEOUT = 5
RAG_QUERY_TIMEOUT = 20
LLM_QUERY_TIMEOUT = 30

# Default limits
DEFAULT_TOP_K = 5
MAX_TOP_K = 50

# ============================================================================
# Validation Rules
# ============================================================================

def validate_module_list(modules: list) -> bool:
    """
    Validate that all modules in a list are valid.
    
    Args:
        modules: List of module names to validate
        
    Returns:
        bool: True if all modules are valid, False otherwise
    """
    if not modules or not isinstance(modules, list):
        return False
    return all(is_valid_module(m) for m in modules)


def validate_top_k(top_k: int) -> bool:
    """
    Validate top_k parameter.
    
    Args:
        top_k: Number of results to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return isinstance(top_k, int) and 1 <= top_k <= MAX_TOP_K
