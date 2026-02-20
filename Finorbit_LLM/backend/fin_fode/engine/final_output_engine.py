from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from backend.fin_fode.core.models import FODEInput, FODEOutput
from backend.fin_fode.engine.red_zone_filter import RedZoneFilter
from backend.fin_fode.engine.tone_filter import ToneFilter
from backend.fin_fode.engine.emoji_controller import EmojiController
from backend.fin_fode.engine.template_policy import TemplatePolicy
from backend.fin_fode.engine.section_builder import SectionBuilder


class FinalOutputEngine:
    """Final Output Design Engine (FODE).

    Strong default order (do not change casually):
      1) Red-zone filter (hard safety / override)
      2) Template policy + section builder (structure)
      3) Tone filter (disclaimer + soft sanitization)
      4) Emoji controller (light cosmetic prefix)

    Designed to be run AFTER post-validation and AFTER compliance.
    Fail-open: If any step errors, it returns a best-effort answer.
    """

    def __init__(self, config_dir: Optional[str] = None):
        cfg_dir = Path(config_dir) if config_dir else (Path(__file__).resolve().parents[1] / "config")
        self.cfg_dir = cfg_dir

        self.modules_cfg = _load_yaml(cfg_dir / "modules.yaml")
        self.red_zone_cfg = _load_yaml(cfg_dir / "red_zone.yaml")
        self.tone_cfg = _load_yaml(cfg_dir / "tone_rules.yaml")

        self.red_zone = RedZoneFilter(self.red_zone_cfg)
        self.policy = TemplatePolicy((self.modules_cfg.get("template_policy") or {}))
        self.sections = SectionBuilder((self.modules_cfg.get("section_builder") or {}))
        self.tone = ToneFilter(self.tone_cfg)
        self.emoji = EmojiController((self.tone_cfg.get("emoji") or {}))

    def run(self, payload: Dict[str, Any] | FODEInput) -> Dict[str, Any]:
        fin = payload if isinstance(payload, FODEInput) else FODEInput(
            raw_answer=(payload or {}).get("raw_answer", "") or "",
            context=(payload or {}).get("context", {}) or {},
        )

        ctx = fin.context or {}
        raw = fin.raw_answer or ""

        applied: list[str] = []
        flags: list[str] = []
        meta: Dict[str, Any] = {
            "module": ctx.get("module"),
            "channel": ctx.get("channel"),
        }

        # 1) Red-zone filter (hard)
        try:
            t, a, f, m = self.red_zone.apply(raw, ctx)
            raw = t
            applied += a
            flags += f
            meta["red_zone"] = m
            if "FODE_REDZONE_BLOCK" in flags:
                out = FODEOutput(final_answer=raw, applied=applied, flags=flags, meta=meta)
                return asdict(out)
        except Exception as e:
            flags.append("FODE_ERR_REDZONE")
            meta["red_zone_error"] = str(e)

        # 2) Template policy + section builder
        try:
            style = self.policy.pick_style(ctx)

            # Only force clarify layout for hard blocks (partial/refuse).
            # For "clarify", let the section_builder decide whether to preserve the real response.
            if ctx.get("recommended_action") in ("partial", "refuse"):
                style = "clarify"
            elif ctx.get("recommended_action") == "clarify" or ctx.get("needs_clarification"):
                style = "clarify"

            t, a, f, m = self.sections.apply(raw, ctx, style=style)
            raw = t
            applied += a
            flags += f
            meta["sections"] = m
        except Exception as e:
            flags.append("FODE_ERR_SECTIONS")
            meta["sections_error"] = str(e)

        # 3) Tone filter
        try:
            t, a, f, m = self.tone.apply(raw, ctx)
            raw = t
            applied += a
            flags += f
            meta["tone"] = m
        except Exception as e:
            flags.append("FODE_ERR_TONE")
            meta["tone_error"] = str(e)

        # 4) Emoji controller
        try:
            t, a, f, m = self.emoji.apply(raw, ctx)
            raw = t
            applied += a
            flags += f
            meta["emoji"] = m
        except Exception as e:
            flags.append("FODE_ERR_EMOJI")
            meta["emoji_error"] = str(e)

        # 5) Sources / citations (deterministic)
        try:
            raw, a, f, m = _append_sources(raw, ctx)
            applied += a
            flags += f
            if m:
                meta["sources"] = m
        except Exception as e:
            flags.append("FODE_ERR_SOURCES")
            meta["sources_error"] = str(e)

        out = FODEOutput(final_answer=raw, applied=applied, flags=flags, meta=meta)
        return asdict(out)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _append_sources(text: str, context: Dict[str, Any]) -> tuple[str, list[str], list[str], Dict[str, Any]]:
    """Append a simple 'Sources' section if retrieved passages exist.

    Expected shape: context['retrieved_passages'] = list[dict] with keys like
    document, chunk_index, similarity_score, excerpt.
    """
    applied: list[str] = []
    flags: list[str] = []
    meta: Dict[str, Any] = {}

    passages = (context or {}).get("retrieved_passages") or []
    if not isinstance(passages, list) or not passages:
        return text or "", applied, flags, meta

    # Dedupe by (document, chunk_index)
    seen: set[tuple[str, Any]] = set()
    items: list[dict] = []
    for p in passages:
        if not isinstance(p, dict):
            continue
        doc = p.get("document")
        idx = p.get("chunk_index")
        if not doc:
            continue
        key = (str(doc), idx)
        if key in seen:
            continue
        seen.add(key)
        items.append(p)

    if not items:
        return text or "", applied, flags, meta

    lines: list[str] = ["Sources"]
    for i, p in enumerate(items, 1):
        doc = str(p.get("document"))
        idx = p.get("chunk_index")
        score = p.get("similarity_score")

        suffix_parts: list[str] = []
        if idx is not None:
            suffix_parts.append(f"chunk {idx}")
        if isinstance(score, (int, float)):
            suffix_parts.append(f"score {score:.2f}")
        suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
        lines.append(f"{i}. {doc}{suffix}")

    out = (text or "").rstrip() + "\n\n" + "\n".join(lines)
    applied.append("sources:append")
    meta["count"] = len(items)
    return out, applied, flags, meta
