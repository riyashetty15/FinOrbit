from __future__ import annotations

from typing import Any, Dict, List, Tuple


class EmojiController:
    """Adds a light emoji prefix based on module."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.module_prefix = self.cfg.get("module_prefix", {}) or {}

    def apply(self, text: str, context: Dict[str, Any]) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
        if not self.enabled:
            return text, [], [], {}

        applied: List[str] = []
        flags: List[str] = []
        meta: Dict[str, Any] = {}

        module = (context.get("module") or "GENERIC").upper()
        prefix = self.module_prefix.get(module)

        if prefix and (text or "").strip() and not (text or "").strip().startswith(prefix):
            text = f"{prefix} {text}"
            applied.append("emoji_controller:prefix")
            meta["prefix"] = prefix

        return text or "", applied, flags, meta
