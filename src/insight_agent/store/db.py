"""SQLite-backed application state.

One file holds saved reports, user preferences, the audit log and the trace
store.  SQLite is the right choice for a prototype that must run on a
reviewer's machine with no services to start, and it is not a toy: the schema
below is the same shape we would deploy on Cloud SQL for Postgres, and the
access layer is narrow enough that swapping the driver is a contained change.

Design notes that matter beyond the prototype:

* Reports are **soft-deleted**.  A destructive action that cannot be examined
  afterwards is not auditable, and requirement 3 is about oversight.
* Every mutation writes an audit row in the same transaction.
* ``deletion_batches`` records the exact set of ids a confirmation applied to.
  LangGraph re-executes a node from the start when an interrupt resumes, so a
  delete must be idempotent; the batch id makes replay a no-op.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reports (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL,
    summary       TEXT NOT NULL DEFAULT '',
    sql_used      TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    deleted_at    TEXT,
    deleted_by    TEXT,
    delete_batch  TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_user   ON reports(user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_reports_thread ON reports(thread_id, deleted_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    thread_id   TEXT,
    action      TEXT NOT NULL,
    target      TEXT,
    outcome     TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id, ts);

-- Records that a confirmed destructive action has already been applied, so
-- replaying the node after an interrupt resume does not delete twice.
CREATE TABLE IF NOT EXISTS deletion_batches (
    batch_id    TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    report_ids  TEXT NOT NULL,
    confirmed_at TEXT NOT NULL,
    applied_at  TEXT
);

CREATE TABLE IF NOT EXISTS preferences (
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    evidence    INTEGER NOT NULL DEFAULT 1,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    ts          TEXT NOT NULL,
    user_id     TEXT,
    thread_id   TEXT,
    node        TEXT NOT NULL,
    event       TEXT NOT NULL,
    duration_ms REAL,
    payload     TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (trace_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_traces_ts     ON traces(ts);
CREATE INDEX IF NOT EXISTS idx_traces_thread ON traces(thread_id);
"""

_lock = threading.Lock()
_connections: dict[str, sqlite3.Connection] = {}


def utcnow() -> str:
    """ISO-8601 UTC timestamp. One format everywhere."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    """Return the process-wide connection for ``path``, creating the schema."""
    key = str(path)
    with _lock:
        conn = _connections.get(key)
        if conn is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(SCHEMA)
            conn.commit()
            _connections[key] = conn
        return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Group related writes so a report and its audit row commit together."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def record_audit(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    action: str,
    outcome: str,
    thread_id: str | None = None,
    target: str | None = None,
    detail: dict | None = None,
) -> None:
    """Append one immutable audit row.

    Called for refusals as well as for actions. A blocked PII attempt is
    exactly the event a security reviewer will come looking for.
    """
    conn.execute(
        "INSERT INTO audit_log (ts, user_id, thread_id, action, target, outcome, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (utcnow(), user_id, thread_id, action, target, outcome, json.dumps(detail or {})),
    )
