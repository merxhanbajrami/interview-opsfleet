"""Structured tracing.

Knowing that the agent is failing needs counters. Knowing why needs the full
message correspondence for one turn, so every turn gets a trace_id and every
node emits ordered events under it.

Payloads are scrubbed before they are written. A trace store that accumulates
personal data is a second copy of the problem the guard exists to prevent.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insight_agent.security.scrubber import scrub_text
from insight_agent.store.db import utcnow

log = logging.getLogger(__name__)

#: Payload keys whose values are truncated rather than stored whole. Prompts
#: and report bodies are useful for debugging but large.
_TRUNCATE_KEYS = frozenset({"prompt", "response", "body", "rows", "messages"})
_TRUNCATE_AT = 2000


@dataclass(slots=True)
class TraceEvent:
    trace_id: str
    seq: int
    ts: str
    node: str
    event: str
    duration_ms: float | None = None
    user_id: str | None = None
    thread_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        return json.dumps(
            {
                "trace_id": self.trace_id, "seq": self.seq, "ts": self.ts,
                "node": self.node, "event": self.event,
                "duration_ms": self.duration_ms, "user_id": self.user_id,
                "thread_id": self.thread_id, "payload": self.payload,
            },
            default=str,
        )


class Tracer:
    """Emits ordered events for one conversation turn."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        log_path: Path,
        *,
        user_id: str | None = None,
        thread_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.conn = conn
        self.log_path = log_path
        self.user_id = user_id
        self.thread_id = thread_id
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self._seq = 0
        #: Rolled up at the end of the turn for the metrics line.
        self.counters: dict[str, float] = {}

    # --- Emission ---------------------------------------------------------

    def emit(
        self,
        node: str,
        event: str,
        *,
        duration_ms: float | None = None,
        **payload: Any,
    ) -> TraceEvent:
        self._seq += 1
        record = TraceEvent(
            trace_id=self.trace_id, seq=self._seq, ts=utcnow(), node=node,
            event=event, duration_ms=duration_ms, user_id=self.user_id,
            thread_id=self.thread_id, payload=_sanitize(payload),
        )
        self._persist(record)
        return record

    def count(self, name: str, value: float = 1.0) -> None:
        """Accumulate a per-turn metric."""
        self.counters[name] = self.counters.get(name, 0.0) + value

    @contextmanager
    def span(self, node: str, **payload: Any) -> Iterator[dict[str, Any]]:
        """Time a node and record its outcome either way.

        The failure path emits too. A node that raises is precisely the one a
        debugger needs a record of.
        """
        started = time.perf_counter()
        self.emit(node, "start", **payload)
        extra: dict[str, Any] = {}
        try:
            yield extra
        except Exception as exc:
            self.emit(
                node, "error",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=type(exc).__name__, error=str(exc)[:500], **extra,
            )
            self.count("errors")
            raise
        else:
            self.emit(
                node, "end",
                duration_ms=(time.perf_counter() - started) * 1000, **extra,
            )

    # --- Persistence ------------------------------------------------------

    def _persist(self, record: TraceEvent) -> None:
        """Write the event. Tracing never breaks the request it is observing."""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, seq, ts, user_id, thread_id, "
                "node, event, duration_ms, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.trace_id, record.seq, record.ts, record.user_id,
                    record.thread_id, record.node, record.event, record.duration_ms,
                    json.dumps(record.payload, default=str),
                ),
            )
            self.conn.commit()
        except Exception:  # noqa: BLE001
            log.debug("trace write to sqlite failed", exc_info=True)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(record.as_json() + "\n")
        except Exception:  # noqa: BLE001
            log.debug("trace write to jsonl failed", exc_info=True)

    def summary(self) -> dict[str, float]:
        return dict(self.counters)


def _sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    """Scrub and truncate a payload before it is stored."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            text = scrub_text(value).text
            if key in _TRUNCATE_KEYS and len(text) > _TRUNCATE_AT:
                text = text[:_TRUNCATE_AT] + f"… [{len(text) - _TRUNCATE_AT} more chars]"
            clean[key] = text
        elif isinstance(value, (int, float, bool)) or value is None:
            clean[key] = value
        else:
            rendered = scrub_text(str(value)).text
            clean[key] = rendered[:_TRUNCATE_AT]
    return clean


# --- Read side: metrics and replay -----------------------------------------


def recent_traces(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    """One row per turn, newest first, for `insight traces`."""
    rows = conn.execute(
        """
        SELECT trace_id,
               MIN(ts)                                             AS started,
               COALESCE(user_id, '')                               AS user_id,
               SUM(CASE WHEN event = 'error' THEN 1 ELSE 0 END)    AS errors,
               SUM(COALESCE(duration_ms, 0))                       AS total_ms,
               COUNT(*)                                            AS events
        FROM traces
        GROUP BY trace_id
        ORDER BY started DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def load_trace(conn: sqlite3.Connection, trace_id: str) -> list[dict[str, Any]]:
    """Full ordered event list for one turn: the message correspondence."""
    rows = conn.execute(
        "SELECT * FROM traces WHERE trace_id LIKE ? ORDER BY seq",
        (f"{trace_id}%",),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        try:
            record["payload"] = json.loads(record["payload"])
        except (TypeError, ValueError):
            record["payload"] = {}
        out.append(record)
    return out


def agent_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """The agent-level metrics named in requirement 7.

    Deliberately small. Each one maps to a question an on-call engineer asks:
    is it up, is it slow, is it self-correcting, is it being probed.
    """
    turns = conn.execute("SELECT COUNT(DISTINCT trace_id) FROM traces").fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(DISTINCT trace_id) FROM traces WHERE event = 'error'"
    ).fetchone()[0]
    latency = conn.execute(
        "SELECT AVG(t), MAX(t) FROM (SELECT SUM(COALESCE(duration_ms,0)) t "
        "FROM traces GROUP BY trace_id)"
    ).fetchone()

    def _count(node: str, event: str) -> int:
        return conn.execute(
            "SELECT COUNT(*) FROM traces WHERE node = ? AND event = ?", (node, event)
        ).fetchone()[0]

    guard_rejections = _count("sql_guard", "reject")
    repairs = _count("sql_repair", "start")
    repairs_ok = _count("sql_repair", "recovered")
    blocked_inputs = _count("input_guard", "reject")

    return {
        "turns": turns,
        "turns_failed": failed,
        "success_rate": round(1 - (failed / turns), 3) if turns else 1.0,
        "avg_turn_ms": round(latency[0] or 0, 1),
        "max_turn_ms": round(latency[1] or 0, 1),
        "guard_rejections": guard_rejections,
        "input_rejections": blocked_inputs,
        "sql_repairs_attempted": repairs,
        "sql_repairs_recovered": repairs_ok,
        "repair_success_rate": round(repairs_ok / repairs, 3) if repairs else None,
    }
