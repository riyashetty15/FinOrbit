from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class RedZoneFilter:
    """Hard safety filter.

    - If any block pattern matches: replace response with a block message.
    - If any soft pattern matches: flag but do not block.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.block_message = self.cfg.get(
            "block_message",
            "I can’t help with that request. Please rephrase your question in a safe, legal way.",
        )

        self.block_patterns = [re.compile(p, re.I) for p in (self.cfg.get("block_patterns", []) or [])]
        self.soft_patterns = [re.compile(p, re.I) for p in (self.cfg.get("soft_warn_patterns", []) or [])]

    def apply(self, text: str, context: Dict[str, Any]) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
        applied: List[str] = []
        flags: List[str] = []
        meta: Dict[str, Any] = {}

        t = text or ""

        for pat in self.block_patterns:
            m = pat.search(t)
            if m:
                flags.append("FODE_REDZONE_BLOCK")
                applied.append("red_zone_filter:block")
                meta["matched_block"] = m.group(0)
                return self.block_message, applied, flags, meta

        for pat in self.soft_patterns:
            m = pat.search(t)
            if m:
                flags.append("FODE_REDZONE_WARN")
                applied.append("red_zone_filter:warn")
                meta.setdefault("matched_warn", []).append(m.group(0))

        return t, applied, flags, meta
