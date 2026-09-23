"""Learned user preferences (requirement 4, user level).

"User A prefers tables" is not something to ask about. It is something to
notice, from explicit statements and from repeated requests.

Confidence matters because a preference applied too eagerly is worse than
none: one offhand "as bullets please" should not change every future report.

Storage is a small closed key/value set rather than a free-text memory blob,
so a model cannot invent a key and the user can inspect and correct it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from insight_agent.store.db import transaction, utcnow

#: Below this, a preference is remembered but not acted on.
APPLY_THRESHOLD = 0.6
#: Confidence assigned to a preference the user stated outright.
EXPLICIT_CONFIDENCE = 0.95
#: Confidence gained per repeated implicit observation.
IMPLICIT_STEP = 0.25
#: Confidence lost when the user asks for the opposite.
CONTRADICTION_PENALTY = 0.4

#: The preference keys the agent understands. A closed set, so a model cannot
#: invent a key that silently changes behaviour no one designed.
KNOWN_KEYS: dict[str, str] = {
    "output_shape": "Preferred answer shape: table, bullets, prose, or chart.",
    "detail_level": "How deep the analysis should go: brief, standard, or deep.",
    "include_sql": "Whether to show the SQL used: always or never.",
    "include_actions": "Whether reports should end with action items: always or never.",
    "persona": "Preferred report persona.",
}


@dataclass(slots=True)
class Preference:
    key: str
    value: str
    confidence: float
    evidence: int
    updated_at: str

    @property
    def is_applied(self) -> bool:
        return self.confidence >= APPLY_THRESHOLD

    def describe(self) -> str:
        state = "applied" if self.is_applied else "watching"
        return (
            f"{self.key} = {self.value}  "
            f"({state}, confidence {self.confidence:.0%}, {self.evidence} observation"
            f"{'s' if self.evidence != 1 else ''})"
        )


class PreferenceStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def all_for_user(self, user_id: str) -> list[Preference]:
        rows = self.conn.execute(
            "SELECT key, value, confidence, evidence, updated_at FROM preferences "
            "WHERE user_id = ? ORDER BY key",
            (user_id,),
        ).fetchall()
        return [Preference(*row) for row in rows]

    def applied(self, user_id: str) -> dict[str, str]:
        """Only the preferences confident enough to act on."""
        return {p.key: p.value for p in self.all_for_user(user_id) if p.is_applied}

    def observe(
        self, user_id: str, key: str, value: str, *, explicit: bool = False
    ) -> Preference | None:
        """Record one observation about a user's preference.

        Returns the updated preference, or ``None`` if the key is not one the
        agent knows how to act on.
        """
        if key not in KNOWN_KEYS:
            return None

        existing = next((p for p in self.all_for_user(user_id) if p.key == key), None)

        if existing is None:
            confidence = EXPLICIT_CONFIDENCE if explicit else IMPLICIT_STEP
            evidence = 1
        elif existing.value == value:
            # Same signal again: more confident, capped below certainty so a
            # contradiction can still move it.
            confidence = min(
                0.99, max(existing.confidence, EXPLICIT_CONFIDENCE if explicit else 0.0)
                + (0.0 if explicit else IMPLICIT_STEP)
            )
            evidence = existing.evidence + 1
        else:
            # Opposite signal. An explicit statement overrides immediately; an
            # implicit one erodes the old belief first.
            if explicit:
                confidence, evidence = EXPLICIT_CONFIDENCE, 1
            else:
                reduced = existing.confidence - CONTRADICTION_PENALTY
                if reduced > APPLY_THRESHOLD:
                    # Old belief still stands; record the doubt, keep the value.
                    self._write(user_id, key, existing.value, reduced, existing.evidence)
                    return Preference(
                        key, existing.value, reduced, existing.evidence, utcnow()
                    )
                confidence, evidence = IMPLICIT_STEP, 1

        self._write(user_id, key, value, confidence, evidence)
        return Preference(key, value, confidence, evidence, utcnow())

    def set_explicit(self, user_id: str, key: str, value: str) -> Preference | None:
        return self.observe(user_id, key, value, explicit=True)

    def forget(self, user_id: str, key: str) -> bool:
        with transaction(self.conn):
            cursor = self.conn.execute(
                "DELETE FROM preferences WHERE user_id = ? AND key = ?", (user_id, key)
            )
        return cursor.rowcount > 0

    def forget_all(self, user_id: str) -> int:
        with transaction(self.conn):
            cursor = self.conn.execute(
                "DELETE FROM preferences WHERE user_id = ?", (user_id,)
            )
        return cursor.rowcount

    def as_prompt_section(self, user_id: str) -> str:
        """Render applied preferences for the formatter's system prompt."""
        applied = self.applied(user_id)
        if not applied:
            return ""
        lines = ["Known preferences for this user:"]
        lines += [f"- {key}: {value}" for key, value in sorted(applied.items())]
        return "\n".join(lines)

    def _write(
        self, user_id: str, key: str, value: str, confidence: float, evidence: int
    ) -> None:
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO preferences (user_id, key, value, confidence, evidence, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
                "confidence = excluded.confidence, evidence = excluded.evidence, "
                "updated_at = excluded.updated_at",
                (user_id, key, value, round(confidence, 3), evidence, utcnow()),
            )
