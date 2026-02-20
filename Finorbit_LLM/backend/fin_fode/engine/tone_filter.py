from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class ToneFilter:
    """Soft tone enforcement.

    - Replaces configured banned phrases (string-level; keep config conservative)
    - Appends a default disclaimer if required
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.disclaimer = self.cfg.get(
            "default_disclaimer",
            "Note: This is general information, not personalized financial advice.",
        )
        self.force_disclaimer = bool(self.cfg.get("force_disclaimer", True))
        self.banned_phrases = [p.lower() for p in (self.cfg.get("banned_phrases", []) or [])]
        self.replacements = self.cfg.get("replacements", {}) or {}

    def apply(self, text: str, context: Dict[str, Any]) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
        applied: List[str] = []
        flags: List[str] = []
        meta: Dict[str, Any] = {}

        t = text or ""
        lower = t.lower()

        module = str((context or {}).get("module") or "").upper()
        allow_absolutes = bool((context or {}).get("allow_absolutes")) or bool((context or {}).get("compliance_status") == "BLOCKED")

        hits = [bp for bp in self.banned_phrases if bp in lower]
        # In fraud/safety guidance, absolute language is often appropriate (e.g., "Never share OTP").
        if module == "FRAUD" or allow_absolutes:
            hits = [bp for bp in hits if bp != "never"]
        if hits:
            flags.append("FODE_TONE_BANNED_PHRASES")
            meta["banned_hits"] = hits
            for bp in hits:
                repl = self.replacements.get(bp, "may vary")
                t = _replace_case_insensitive(t, bp, repl)
            applied.append("tone_filter:sanitize")

        if self.force_disclaimer and self.disclaimer:
            if self.disclaimer.lower() not in t.lower():
                t = t.rstrip() + "\n\n" + self.disclaimer
                applied.append("tone_filter:disclaimer")

        return t, applied, flags, meta


def _replace_case_insensitive(text: str, needle: str, repl: str) -> str:
    return re.sub(re.escape(needle), repl, text, flags=re.I)
