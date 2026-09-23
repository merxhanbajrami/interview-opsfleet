"""Last line of defence on the way out.

The SQL guard should make this redundant, but it is not the only path to the
user's screen: an error can quote a row, a model can hallucinate an address,
a future contributor can add a code path that bypasses the guard.

Two jobs. Customer ids become stable HMAC pseudonyms, so "top customers"
works while personal identity does not survive. And identifier patterns are
redacted, with card numbers Luhn-checked first, because in a tool full of
large numbers a false positive would corrupt real revenue figures.
"""

from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass, field
from hashlib import blake2b, sha256

import pandas as pd

from insight_agent.data.catalog import pseudonym_columns

#: Per-deployment key. Generated once and stored with the app state; rotating
#: it changes every pseudonym, which is the intended behaviour on key
#: compromise. In production this is a KMS-held secret.
_KEY_ENV = "INSIGHT_PSEUDONYM_KEY"


def _key() -> bytes:
    raw = os.environ.get(_KEY_ENV)
    if raw:
        return raw.encode()
    # Deterministic within a process so pseudonyms stay stable for a session
    # even when no key is configured.
    return sha256(b"insight-agent-dev-key").digest()


# --- Patterns ---------------------------------------------------------------
# Ordered most specific first. Each carries the label used in the replacement
# so redactions are self-documenting in the transcript.

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Card numbers are Luhn-checked before redaction. This is a data-analysis
    # tool: a naive 13-to-19-digit pattern would silently corrupt legitimate
    # revenue figures, which is a worse failure than the one it prevents.
    ("CARD", re.compile(r"(?<!\d)(?<!\d\.)(?:\d[ -]?){12,18}\d(?!\.?\d)")),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?<!\d\.)(?:\+?1[ -.])?(?:\(\d{3}\)|\d{3})[ -.]\d{3}[ -.]\d{4}(?!\.?\d)"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
            r"Court|Ct|Way|Terrace|Ter|Place|Pl)\b\.?",
            re.I,
        ),
    ),
    ("COORD", re.compile(r"\b-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}\b")),
)


@dataclass(slots=True)
class ScrubReport:
    """What the scrubber changed. Emitted as a metric; a non-zero count here
    means an earlier layer let something through and should be investigated."""

    text: str
    redactions: dict[str, int] = field(default_factory=dict)
    pseudonyms: int = 0

    @property
    def clean(self) -> bool:
        return not self.redactions

    @property
    def total_redactions(self) -> int:
        return sum(self.redactions.values())


def pseudonymize(value: object) -> str:
    """Stable, non-reversible token for one customer identifier."""
    digest = hmac.new(_key(), str(value).encode(), blake2b).hexdigest()
    return f"CUST-{digest[:8]}"


def scrub_text(text: str) -> ScrubReport:
    """Redact identifier patterns from free text."""
    if not text:
        return ScrubReport(text=text)
    redactions: dict[str, int] = {}
    out = text
    for label, pattern in _PATTERNS:
        if label == "CARD":
            out, count = pattern.subn(_redact_if_luhn, out)
        else:
            out, count = pattern.subn(f"[{label} REDACTED]", out)
            count = count if count else 0
        if label == "CARD":
            count = out.count("[CARD REDACTED]")
        if count:
            redactions[label] = redactions.get(label, 0) + count
    return ScrubReport(text=out, redactions=redactions)


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum, used to tell a card number from a big integer."""
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_if_luhn(match: re.Match[str]) -> str:
    """Redact only sequences that actually check out as card numbers."""
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    return "[CARD REDACTED]" if _luhn_valid(digits) else raw


def pseudonymize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace customer identifiers in a result set with stable pseudonyms.

    Matching is by column name against the catalog's pseudonym classification,
    so a column aliased in SQL as ``user_id`` is caught wherever it came from.
    """
    if frame.empty:
        return frame, 0

    targets = set()
    for names in pseudonym_columns().values():
        targets |= set(names)

    out = frame.copy()
    replaced = 0
    for column in out.columns:
        bare = str(column).lower().split(".")[-1]
        if bare in targets:
            out[column] = out[column].map(
                lambda v: pseudonymize(v) if pd.notna(v) else v
            )
            replaced += 1
    return out, replaced


def scrub_records(records: list[dict], ) -> list[dict]:
    """Scrub a list of row dicts before they enter the model's context.

    Applied to the sample rows handed to the analyst node: whatever the model
    never sees, it cannot repeat.
    """
    targets = set()
    for names in pseudonym_columns().values():
        targets |= set(names)

    out: list[dict] = []
    for row in records:
        clean: dict = {}
        for key, value in row.items():
            bare = str(key).lower().split(".")[-1]
            if bare in targets and value is not None:
                clean[key] = pseudonymize(value)
            elif isinstance(value, str):
                clean[key] = scrub_text(value).text
            else:
                clean[key] = value
        out.append(clean)
    return out
