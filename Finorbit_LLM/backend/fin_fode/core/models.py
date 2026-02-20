from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FODEInput:
    raw_answer: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FODEOutput:
    final_answer: str
    flags: List[str] = field(default_factory=list)
    applied: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
