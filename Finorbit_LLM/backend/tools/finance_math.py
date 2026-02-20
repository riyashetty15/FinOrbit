import math
from typing import Dict, Any, Union

class FinanceMathTools:
    """
    MCP-style Tool Server for Financial Calculations.
    These methods are pure functions that agents can call.
    """
    
    @staticmethod
    def calculate_emi(principal: float, rate_annual: float, tenure_years: float) -> Dict[str, Any]:
        """
        Calculate Equated Monthly Installment (EMI) for a loan.
        
        Args:
            principal (float): Loan amount
            rate_annual (float): Annual interest rate in percentage (e.g., 8.5)
            tenure_years (float): Loan tenure in years
            
        Returns:
            dict: { "emi": float, "total_interest": float, "total_payment": float, ... }
        """
        try:
            p = float(principal)
            r = float(rate_annual) / (12 * 100) # Monthly interest rate
            n = float(tenure_years) * 12 # Months
            
            if r == 0:
                emi = p / n
            else:
                emi = p * r * ((1 + r)**n) / (((1 + r)**n) - 1)
            
            total_payment = emi * n
            total_interest = total_payment - p
            
            return {
                "emi": round(emi, 2),
                "total_interest": round(total_interest, 2),
                "total_payment": round(total_payment, 2),
                "monthly_rate_approx": round(r * 100, 4),
                "months": int(n),
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def calculate_sip_returns(monthly_investment: float, rate_annual: float, years: float) -> Dict[str, Any]:
        """
        Calculate expected returns from a Systematic Investment Plan (SIP).
        
        Args:
            monthly_investment (float): Amount invested per month
            rate_annual (float): Expected annual return rate in percentage
            years (float): Duration of investment in years
            
        Returns:
            dict: { "invested_amount": float, "wealth_gained": float, "total_value": float }
        """
        try:
            p = float(monthly_investment)
            i = float(rate_annual) / (12 * 100)
            n = float(years) * 12
            
            if i == 0:
                 total_value = p * n
            else:
                 # Future Value of Annuity formula
                 total_value = p * ((((1 + i)**n) - 1) / i) * (1 + i)
            
            invested_amount = p * n
            wealth_gained = total_value - invested_amount
            
            return {
                "invested_amount": round(invested_amount, 2),
                "wealth_gained": round(wealth_gained, 2),
                "total_value": round(total_value, 2),
                "status": "success"
            }
        except Exception as e:
             return {"status": "error", "message": str(e)}

    @staticmethod
    def calculate_inflation_impact(current_amount: float, inflation_rate: float, years: float) -> Dict[str, Any]:
        """
        Calculate future value or purchasing power impact due to inflation.
        """
        try:
            c = float(current_amount)
            r = float(inflation_rate) / 100
            t = float(years)
            
            future_cost = c * ((1 + r) ** t)
            purchasing_power = c / ((1 + r) ** t)
            
            return {
                "future_value_required": round(future_cost, 2),
                "present_purchasing_power": round(purchasing_power, 2),
                "years": t,
                "inflation_rate": inflation_rate,
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    def estimate_tax_new_regime_2024(income: float) -> Dict[str, Any]:
        """
        Estimate income tax based on India's New Tax Regime (FY 2024-25).
        Note: This is a simplified estimation.
        """
        try:
            income = float(income)
            standard_deduction = 75000 # Updated for FY25
            taxable_income = max(0, income - standard_deduction)
            
            # Slabs FY 24-25 (New Regime)
            # 0-3L: Nil
            # 3-7L: 5% (New slabs often adjusted, using common simplified 2024 structure)
            
            tax = 0
            remaining_income = taxable_income
            
            # 0 - 3L: Nil
            slab1 = 300000
            if remaining_income > slab1:
                remaining_income -= slab1
            else:
                remaining_income = 0
                
            # 3 - 7L @ 5%
            slab2_gap = 400000 
            if remaining_income > 0:
                taxable_in_slab = min(remaining_income, slab2_gap)
                tax += taxable_in_slab * 0.05
                remaining_income -= taxable_in_slab
            
            # 7 - 10L @ 10%
            slab3_gap = 300000
            if remaining_income > 0:
                 taxable_in_slab = min(remaining_income, slab3_gap)
                 tax += taxable_in_slab * 0.10
                 remaining_income -= taxable_in_slab

            # 10 - 12L @ 15%
            slab4_gap = 200000
            if remaining_income > 0:
                 taxable_in_slab = min(remaining_income, slab4_gap)
                 tax += taxable_in_slab * 0.15
                 remaining_income -= taxable_in_slab
                 
            # 12 - 15L @ 20%
            slab5_gap = 300000
            if remaining_income > 0:
                 taxable_in_slab = min(remaining_income, slab5_gap)
                 tax += taxable_in_slab * 0.20
                 remaining_income -= taxable_in_slab
            
            # > 15L @ 30%
            if remaining_income > 0:
                tax += remaining_income * 0.30
                
            # Rebate u/s 87A if taxable income <= 7L
            # In New Regime, if income <= 7L, tax is zero.
            if taxable_income <= 700000:
                tax = 0
                
            cess = tax * 0.04
            total_tax = tax + cess
            
            return {
                "gross_income": income,
                "standard_deduction": standard_deduction,
                "taxable_income": taxable_income,
                "base_tax": round(tax, 2),
                "cess": round(cess, 2),
                "total_tax_estimated": round(total_tax, 2),
                "regime": "New Regime FY24-25",
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
