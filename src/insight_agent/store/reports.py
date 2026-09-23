"""The Saved Reports library and its destructive-operation path.

The shape that satisfies requirement 3 is preview, confirm, apply.

Three properties make it safe rather than merely polite. Nothing is inferred
at apply time, because the batch stores the resolved ids. Applying twice is a
no-op, because LangGraph re-runs a node body when an interrupt resumes. And
deletes are soft, with an audit row for every action.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass

from insight_agent.store.db import record_audit, transaction, utcnow


@dataclass(slots=True)
class Report:
    id: str
    user_id: str
    thread_id: str
    title: str
    body: str
    summary: str
    sql_used: str
    created_at: str

    def short(self) -> str:
        return f"{self.id[:8]}  {self.created_at[:16]}  {self.title}"


@dataclass(slots=True)
class DeletionBatch:
    """A resolved, confirmed-pending set of reports."""

    batch_id: str
    user_id: str
    thread_id: str
    reports: list[Report]
    criterion: str

    @property
    def count(self) -> int:
        return len(self.reports)

    @property
    def is_empty(self) -> bool:
        return not self.reports


class ReportStore:
    """All access to saved reports goes through here."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # --- Creation and reading --------------------------------------------

    def save(
        self,
        *,
        user_id: str,
        thread_id: str,
        title: str,
        body: str,
        summary: str = "",
        sql_used: str = "",
    ) -> Report:
        report = Report(
            id=uuid.uuid4().hex,
            user_id=user_id,
            thread_id=thread_id,
            title=title,
            body=body,
            summary=summary,
            sql_used=sql_used,
            created_at=utcnow(),
        )
        with transaction(self.conn):
            self.conn.execute(
                "INSERT INTO reports (id, user_id, thread_id, title, body, summary, "
                "sql_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    report.id, report.user_id, report.thread_id, report.title,
                    report.body, report.summary, report.sql_used, report.created_at,
                ),
            )
            record_audit(
                self.conn, user_id=user_id, thread_id=thread_id,
                action="report.create", target=report.id, outcome="ok",
                detail={"title": title},
            )
        return report

    def list_for_user(self, user_id: str, limit: int = 50) -> list[Report]:
        rows = self.conn.execute(
            "SELECT * FROM reports WHERE user_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [_to_report(r) for r in rows]

    def get(self, report_id: str, user_id: str) -> Report | None:
        """Fetch one report, scoped to its owner.

        Ownership is part of the query, not a check afterwards, so there is no
        code path that reads another user's report at all.
        """
        row = self.conn.execute(
            "SELECT * FROM reports WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            (report_id, user_id),
        ).fetchone()
        return _to_report(row) if row else None

    # --- Destructive path -------------------------------------------------

    def find_deletion_candidates(
        self,
        *,
        user_id: str,
        thread_id: str,
        mentions: str | None = None,
        this_thread_only: bool = False,
        report_ids: list[str] | None = None,
    ) -> DeletionBatch:
        """Resolve a deletion request into an explicit list.

        Every branch filters on ``user_id``. A user cannot delete, or even
        enumerate, another user's reports, which is what makes it safe to let
        them delete their own without an approval step.
        """
        sql = ["SELECT * FROM reports WHERE user_id = ? AND deleted_at IS NULL"]
        params: list[object] = [user_id]
        criterion_parts: list[str] = []

        if report_ids:
            placeholders = ", ".join("?" for _ in report_ids)
            sql.append(f"AND id IN ({placeholders})")
            params.extend(report_ids)
            criterion_parts.append(f"{len(report_ids)} report(s) by id")

        if this_thread_only:
            sql.append("AND thread_id = ?")
            params.append(thread_id)
            criterion_parts.append("created in this conversation")

        if mentions:
            # Case-insensitive substring across the text the user can see.
            sql.append("AND (LOWER(title) LIKE ? OR LOWER(body) LIKE ? OR LOWER(summary) LIKE ?)")
            needle = f"%{mentions.lower()}%"
            params.extend([needle, needle, needle])
            criterion_parts.append(f"mentioning '{mentions}'")

        sql.append("ORDER BY created_at DESC")
        rows = self.conn.execute(" ".join(sql), params).fetchall()

        batch = DeletionBatch(
            batch_id=uuid.uuid4().hex,
            user_id=user_id,
            thread_id=thread_id,
            reports=[_to_report(r) for r in rows],
            criterion=", ".join(criterion_parts) or "all your reports",
        )
        record_audit(
            self.conn, user_id=user_id, thread_id=thread_id,
            action="report.delete.preview", target=batch.batch_id, outcome="pending",
            detail={"criterion": batch.criterion, "count": batch.count},
        )
        self.conn.commit()
        return batch

    def register_confirmation(self, batch: DeletionBatch) -> None:
        """Freeze the confirmed set before anything is deleted.

        Written in its own transaction so that a crash between confirmation and
        application leaves a record of what was agreed to.
        """
        with transaction(self.conn):
            self.conn.execute(
                "INSERT OR IGNORE INTO deletion_batches "
                "(batch_id, user_id, thread_id, report_ids, confirmed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    batch.batch_id, batch.user_id, batch.thread_id,
                    json.dumps([r.id for r in batch.reports]), utcnow(),
                ),
            )

    def apply_deletion(self, batch_id: str, user_id: str) -> tuple[int, bool]:
        """Apply a confirmed batch. Returns ``(count, was_replay)``.

        Idempotent by design: LangGraph replays the node body when an interrupt
        resumes, so this can legitimately be called twice for one confirmation.
        """
        row = self.conn.execute(
            "SELECT * FROM deletion_batches WHERE batch_id = ? AND user_id = ?",
            (batch_id, user_id),
        ).fetchone()
        if row is None:
            raise PermissionError("No confirmed deletion batch with that id for this user.")

        report_ids: list[str] = json.loads(row["report_ids"])

        if row["applied_at"]:
            # Already done. Report the original outcome rather than deleting again.
            return len(report_ids), True

        if not report_ids:
            with transaction(self.conn):
                self.conn.execute(
                    "UPDATE deletion_batches SET applied_at = ? WHERE batch_id = ?",
                    (utcnow(), batch_id),
                )
            return 0, False

        placeholders = ", ".join("?" for _ in report_ids)
        now = utcnow()
        with transaction(self.conn):
            cursor = self.conn.execute(
                f"UPDATE reports SET deleted_at = ?, deleted_by = ?, delete_batch = ? "
                f"WHERE id IN ({placeholders}) AND user_id = ? AND deleted_at IS NULL",
                [now, user_id, batch_id, *report_ids, user_id],
            )
            self.conn.execute(
                "UPDATE deletion_batches SET applied_at = ? WHERE batch_id = ?",
                (now, batch_id),
            )
            record_audit(
                self.conn, user_id=user_id, thread_id=row["thread_id"],
                action="report.delete.apply", target=batch_id, outcome="ok",
                detail={"count": cursor.rowcount, "report_ids": report_ids},
            )
        return cursor.rowcount, False

    def record_cancellation(self, batch: DeletionBatch) -> None:
        with transaction(self.conn):
            record_audit(
                self.conn, user_id=batch.user_id, thread_id=batch.thread_id,
                action="report.delete.cancel", target=batch.batch_id, outcome="cancelled",
                detail={"criterion": batch.criterion, "count": batch.count},
            )

    # --- Recovery ---------------------------------------------------------

    def restore_batch(self, batch_id: str, user_id: str) -> int:
        """Undo a soft delete. The reason soft deletes are worth the column."""
        with transaction(self.conn):
            cursor = self.conn.execute(
                "UPDATE reports SET deleted_at = NULL, deleted_by = NULL, delete_batch = NULL "
                "WHERE delete_batch = ? AND user_id = ?",
                (batch_id, user_id),
            )
            record_audit(
                self.conn, user_id=user_id, action="report.delete.restore",
                target=batch_id, outcome="ok", detail={"count": cursor.rowcount},
            )
        return cursor.rowcount


def _to_report(row: sqlite3.Row) -> Report:
    return Report(
        id=row["id"], user_id=row["user_id"], thread_id=row["thread_id"],
        title=row["title"], body=row["body"], summary=row["summary"],
        sql_used=row["sql_used"], created_at=row["created_at"],
    )
