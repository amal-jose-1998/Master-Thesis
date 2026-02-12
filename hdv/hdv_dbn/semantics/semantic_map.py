from dataclasses import dataclass
from pathlib import Path

import re
import yaml


def _slugify(s):
    """
    Convert a string into a safe identifier for filenames/CSV columns.
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass(frozen=True)
class StyleInfo:
    key: str            # e.g., "s0"
    name: str           # e.g., "free_flow_low_interaction"
    description: str
    slug: str           # safe identifier


@dataclass(frozen=True)
class ActionInfo:
    style_key: str      # e.g., "s0"
    action_key: str     # e.g., "a2"
    name: str
    description: str
    slug: str           # safe identifier (style+action+name)


class SemanticMap:
    """
    Loads a YAML semantic mapping and exposes helpers to:
      - get style/action names and descriptions
      - produce stable 'slugs' for output files
      - create human-readable labels for plots/logging
    """

    def __init__(self, raw):
        self._raw = raw

        styles = raw.get("styles", {})
        actions_by_style = raw.get("actions_by_style", {})

        self._styles = {}
        for sk, sv in styles.items():
            name = str(sv.get("name", sk))
            desc = str(sv.get("description", "")).strip()
            self._styles[sk] = StyleInfo(
                key=sk,
                name=name,
                description=desc,
                slug=_slugify(name) if name else _slugify(sk),
            )

        self._actions = {}
        for sk, actions in actions_by_style.items():
            for ak, av in (actions or {}).items():
                name = str(av.get("name", ak))
                desc = str(av.get("description", "")).strip()
                combined_slug = _slugify(f"{sk}_{ak}_{name}")
                self._actions[(sk, ak)] = ActionInfo(
                    style_key=sk,
                    action_key=ak,
                    name=name,
                    description=desc,
                    slug=combined_slug,
                )

    # ----------------------------
    # Construction
    # ----------------------------
    @staticmethod
    def load(path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return SemanticMap(raw)

    # ----------------------------
    # Accessors
    # ----------------------------
    def style(self, s):#
        """Return a StyleInfo object for a style."""
        sk = f"s{s}" if isinstance(s, int) else str(s)
        if sk not in self._styles:
            # fall back to minimal info 
            return StyleInfo(key=sk, name=sk, description="", slug=_slugify(sk))
        return self._styles[sk]

    def action(self, s, a):
        """Return an ActionInfo object for an action conditioned on style."""
        sk = f"s{s}" if isinstance(s, int) else str(s)
        ak = f"a{a}" if isinstance(a, int) else str(a)
        key = (sk, ak)
        if key not in self._actions:
            # fall back to minimal info 
            nm = f"{sk}_{ak}"
            return ActionInfo(style_key=sk, action_key=ak, name=nm, description="", slug=_slugify(nm))
        return self._actions[key]

    # ----------------------------
    # Labels: formatting helpers that return strings for plots, logs, CSVs.
    # ----------------------------
    def style_label(self, s):
        info = self.style(s)
        return f"{info.key}: {info.name}"

    def action_label(self, s, a):
        sinfo = self.style(s)
        ainfo = self.action(s, a)
        return f"{sinfo.key}/{ainfo.action_key}: {ainfo.name}"

    def state_label(self, s, a):
        sinfo = self.style(s)
        ainfo = self.action(s, a)
        return f"{sinfo.key}({sinfo.name}) · {ainfo.action_key}({ainfo.name})"

    def describe(self, s, a):
        """
        Human-readable description for:
          - a style (if a is None)
          - a (style, action) pair
        """
        sinfo = self.style(s)
        if a is None:
            return f"{sinfo.key} — {sinfo.name}\n{sinfo.description}".strip()

        ainfo = self.action(s, a)
        lines = [
            f"{sinfo.key}/{ainfo.action_key} — {sinfo.name} / {ainfo.name}",
            f"Style: {sinfo.description}" if sinfo.description else "",
            f"Action: {ainfo.description}" if ainfo.description else "",
        ]
        return "\n".join([ln for ln in lines if ln]).strip()

    # ----------------------------
    # Output-safe identifiers
    # ----------------------------
    def style_slug(self, s):
        return self.style(s).slug

    def action_slug(self, s, a):
        return self.action(s, a).slug

    def state_slug(self, s, a):
        sinfo = self.style(s)
        ainfo = self.action(s, a)
        return _slugify(f"{sinfo.key}_{sinfo.name}__{ainfo.action_key}_{ainfo.name}")

    # ----------------------------
    # Introspection / iteration
    # ----------------------------
    def all_styles(self):
        """Returns a copy of the internal styles dictionary"""
        return dict(self._styles)

    def all_actions(self):
        """Returns a copy of the internal actions dictionary"""
        return dict(self._actions)
