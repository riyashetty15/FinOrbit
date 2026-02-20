from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SectionBuilder:
    """Builds structured final output."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.cta_line = self.cfg.get(
            "cta_line",
            "Want a more accurate answer? Share your goal + timeframe and I’ll tailor this.",
        )
        self.max_bullets = int(self.cfg.get("max_bullets", 5))

    def apply(self, text: str, context: Dict[str, Any], style: str) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
        applied: List[str] = []
        flags: List[str] = []
        meta: Dict[str, Any] = {"style": style}

        t = (text or "").strip()
        if not t:
            return t, applied, flags, meta

        if style == "clarify":
            questions = context.get("clarify_questions") or [
                "What’s your goal amount and timeframe?",
                "What’s your monthly income and fixed expenses?",
                "Any existing loans/EMIs or dependents?",
            ]
            out = "I need a bit more detail to guide you safely.\n\n"
            out += "Please answer:\n"
            for i, q in enumerate(questions[:3], 1):
                out += f"{i}) {q}\n"
            out += "\n" + self.cta_line
            applied.append("section_builder:clarify")
            return out, applied, flags, meta

        bullets = _to_bullets(t, max_bullets=self.max_bullets)        # If the text is rich (has paragraphs, lists, headings), preserve it as-is
        # instead of converting to bullets which destroys formatting
        has_structure = any(marker in t for marker in ['\n-', '\n•', '\n1.', '\n1)', '##', '**'])
        if has_structure or len(t) > 300:
            out = t + "\n\nNote: This is general information, not personalized financial advice."
            applied.append("section_builder:preserved")
            return out, applied, flags, meta
        out = "Here’s a simple way to think about it:\n"
        for b in bullets:
            out += f"• {b}\n"
        out += "\n" + self.cta_line
        applied.append("section_builder:simple")
        return out, applied, flags, meta


def _to_bullets(text: str, max_bullets: int = 5) -> List[str]:
    lines = [ln.strip(" •-\t") for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
        lines = parts
    return lines[:max_bullets]
