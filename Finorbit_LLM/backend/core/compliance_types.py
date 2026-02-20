# ==============================================
# File: backend/core/compliance_types.py
# Description: Compliance Layer data structures
# ==============================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal


ComplianceStatus = Literal["OK", "BLOCKED", "ERROR"]


@dataclass
class ComplianceResult:
    status: ComplianceStatus
    final_answer: str
    triggered_rule_ids: List[int]
