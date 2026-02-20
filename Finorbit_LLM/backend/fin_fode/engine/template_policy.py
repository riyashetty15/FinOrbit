from __future__ import annotations

from typing import Any, Dict


class TemplatePolicy:
    """Chooses a formatting style for SectionBuilder."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.default_style = self.cfg.get("default_style", "simple")
        self.module_styles = self.cfg.get("module_styles", {}) or {}

    def pick_style(self, context: Dict[str, Any]) -> str:
        module = (context.get("module") or "GENERIC").upper()
        return self.module_styles.get(module, self.default_style)
