"""Runtime-editable agent personas (requirement 8).

Tone must change weekly without a deployment, so the persona is data: a YAML
file read at the start of every turn, keyed on modification time.

The read is defensive, because a non-developer is editing it. A malformed
file keeps the last good version in service, an unknown name falls back to
the default, and numeric fields are clamped.

A persona controls tone, length and shape. It cannot switch off PII masking,
because the guard is not in the prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

#: Persona fields are clamped to sane bounds. A typo of 50000 in max_words
#: must not turn into a runaway generation bill.
_MAX_WORDS_CEILING = 1200
_MAX_WORDS_FLOOR = 60


@dataclass(slots=True)
class Persona:
    name: str
    label: str = ""
    description: str = ""
    tone: str = ""
    guidance: str = ""
    default_shape: str = "prose_with_table"
    max_words: int = 350
    include_sql: bool = False
    #: Populated when the file failed to load and a fallback is in use.
    load_error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_prompt_section(self) -> str:
        """Render the persona for inclusion in a system prompt."""
        lines = [f"Persona: {self.label or self.name}"]
        if self.tone:
            lines.append(f"Tone: {self.tone.strip()}")
        lines.append(f"Preferred shape: {self.default_shape}")
        lines.append(f"Length limit: about {self.max_words} words.")
        if self.guidance:
            lines.append("Style guidance:\n" + self.guidance.strip())
        return "\n".join(lines)


DEFAULT_PERSONA = Persona(
    name="default",
    label="Balanced Analyst",
    tone="Direct and factual. Lead with the number, then the reason.",
    guidance="- Quantify every claim.\n- Say plainly when the data cannot answer.",
)


class PersonaLoader:
    """Reads persona files, caching on modification time."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._cache: dict[str, tuple[float, Persona]] = {}

    def available(self) -> list[str]:
        if not self.directory.is_dir():
            return [DEFAULT_PERSONA.name]
        return sorted(p.stem for p in self.directory.glob("*.yaml"))

    def load(self, name: str) -> Persona:
        """Return the persona, re-reading the file if it changed on disk."""
        path = self.directory / f"{name}.yaml"
        if not path.is_file():
            if name != DEFAULT_PERSONA.name:
                log.warning("persona %r not found; using default", name)
                return self.load(DEFAULT_PERSONA.name)
            return DEFAULT_PERSONA

        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self._cached_or_default(name)

        cached = self._cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError("persona file must contain a mapping")
            persona = _build(name, data)
        except Exception as exc:  # noqa: BLE001
            # A non-developer just saved a broken file. Keep serving the last
            # good version and make the reason visible.
            log.error("persona %r failed to load: %s", name, exc)
            previous = self._cache.get(name)
            fallback = previous[1] if previous else DEFAULT_PERSONA
            return replace(fallback, load_error=f"{type(exc).__name__}: {exc}")

        self._cache[name] = (mtime, persona)
        return persona

    def _cached_or_default(self, name: str) -> Persona:
        cached = self._cache.get(name)
        return cached[1] if cached else DEFAULT_PERSONA


def _build(name: str, data: dict[str, Any]) -> Persona:
    """Validate and clamp one persona definition."""
    fmt = data.get("format") or {}
    if not isinstance(fmt, dict):
        fmt = {}

    try:
        max_words = int(fmt.get("max_words", 350))
    except (TypeError, ValueError):
        max_words = 350

    return Persona(
        name=str(data.get("name", name)),
        label=str(data.get("label", "")),
        description=str(data.get("description", "")),
        tone=str(data.get("tone", "")),
        guidance=str(data.get("guidance", "")),
        default_shape=str(fmt.get("default_shape", "prose_with_table")),
        max_words=max(_MAX_WORDS_FLOOR, min(_MAX_WORDS_CEILING, max_words)),
        include_sql=bool(fmt.get("include_sql", False)),
        raw=data,
    )
