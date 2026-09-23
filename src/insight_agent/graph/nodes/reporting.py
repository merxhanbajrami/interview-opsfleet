"""Saved reports, and the confirmation flow for deleting them.

Requirement 3 asks for a strict confirmation before a destructive action, and
for that confirmation not to damage the user experience.  The tension is real:
a modal "are you sure?" on every request is strict but tiresome, and a
free-text "yes" that the model interprets is pleasant but not strict at all.

The resolution is to split the action across two nodes with a graph interrupt
between them:

``deletion_preview`` resolves the phrase into an explicit, ownership-filtered
list of reports and puts it in state.  Nothing is deleted.  If the phrase
matches nothing, the turn ends there and the user is simply told so — no
confirmation prompt for a no-op.

``deletion_confirm`` begins with ``interrupt()``.  The graph stops, the exact
list is shown, and the conversation state is checkpointed to SQLite.  The
process can restart between the question and the answer and the pending
deletion survives.  Only an explicit approval resumes it.

Two details make this correct rather than merely nice:

* ``interrupt()`` re-executes its node from the top when resumed.  Everything
  before the ``interrupt`` call therefore runs twice.  That is why resolution
  lives in the *previous* node and why ``interrupt`` is the first statement
  here — nothing with a side effect runs before it.
* The confirmed set is the one the user was shown, read back from state.  A
  report created between the preview and the approval is not swept up by it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.types import interrupt

from insight_agent.graph.prompts import REPORT_SYSTEM, report_prompt
from insight_agent.graph.services import Services
from insight_agent.graph.state import AgentState
from insight_agent.llm.client import LLMUnavailableError
from insight_agent.resilience.breaker import CircuitOpenError
from insight_agent.store.reports import DeletionBatch, Report

log = logging.getLogger(__name__)


# --- Writing and saving reports --------------------------------------------


def make_report_composer(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Turn an analysis into a saved report with action items."""

    def report_composer(state: AgentState) -> dict[str, Any]:
        if state.get("answer"):
            return {}

        analysis = state.get("analysis", "")
        if not analysis and not state.get("rows"):
            return {
                "answer": (
                    "I don't have any analysis to write up yet. Ask me a "
                    "question about the data first, then I'll turn the answer "
                    "into a report."
                )
            }

        with services.tracer.span("report_composer"):
            try:
                response = services.llm.complete(
                    report_prompt(
                        state["user_input"], analysis, state.get("rows", []),
                        state.get("sql", ""),
                    ),
                    system=REPORT_SYSTEM,
                    temperature=services.settings.llm_temperature_prose,
                )
            except (LLMUnavailableError, CircuitOpenError) as exc:
                return {
                    "analysis": analysis,
                    "degraded": True,
                    "warnings": [f"Report generation unavailable: {exc}"],
                }

        body = response.text.strip()
        title = _title_of(body, fallback=state["user_input"])

        report = services.reports.save(
            user_id=state["user_id"], thread_id=state["thread_id"],
            title=title, body=body,
            summary=analysis[:400], sql_used=state.get("sql", ""),
        )
        services.tracer.emit("report_composer", "saved", report_id=report.id, title=title)
        services.tracer.count("reports_created")

        return {
            "analysis": body,
            "saved_report_id": report.id,
            "warnings": [f"Saved as report {report.id[:8]} — \"{title}\"."],
        }

    return report_composer


def make_list_reports(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    def list_reports(state: AgentState) -> dict[str, Any]:
        reports = services.reports.list_for_user(state["user_id"])
        services.tracer.emit("list_reports", "listed", count=len(reports))
        if not reports:
            return {"answer": "You have no saved reports yet."}
        lines = [f"You have {len(reports)} saved report(s):", ""]
        lines += [f"- `{r.id[:8]}` · {r.created_at[:10]} · {r.title}" for r in reports]
        return {"answer": "\n".join(lines)}

    return list_reports


# --- Destructive path -------------------------------------------------------


def make_deletion_preview(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Resolve a deletion request into an explicit list. Deletes nothing."""

    def deletion_preview(state: AgentState) -> dict[str, Any]:
        request = state.get("pending_deletion") or {}
        mentions = (request.get("mentions") or "").strip()
        this_thread = bool(request.get("this_conversation"))

        batch = services.reports.find_deletion_candidates(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            mentions=mentions or None,
            this_thread_only=this_thread,
        )

        services.tracer.emit(
            "deletion_preview", "resolved", count=batch.count,
            criterion=batch.criterion,
        )

        if batch.is_empty:
            # No confirmation prompt for something that would do nothing.
            services.tracer.emit("deletion_preview", "no_match")
            return {
                "pending_deletion": {},
                "answer": (
                    f"I found no saved reports matching {batch.criterion}. "
                    "Nothing to delete."
                ),
            }

        return {
            "pending_deletion": {
                "batch_id": batch.batch_id,
                "criterion": batch.criterion,
                "reports": [
                    {"id": r.id, "title": r.title, "created_at": r.created_at}
                    for r in batch.reports
                ],
            }
        }

    return deletion_preview


def make_deletion_confirm(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Ask, then act. ``interrupt`` is the first statement for a reason."""

    def deletion_confirm(state: AgentState) -> dict[str, Any]:
        pending = state.get("pending_deletion") or {}
        reports = pending.get("reports", [])

        # Everything above this line re-runs when the graph resumes, so
        # nothing above this line may have a side effect.
        decision = interrupt(
            {
                "type": "confirm_deletion",
                "batch_id": pending.get("batch_id", ""),
                "criterion": pending.get("criterion", ""),
                "count": len(reports),
                "reports": reports,
                "prompt": (
                    f"Delete {len(reports)} report"
                    f"{'s' if len(reports) != 1 else ''} matching "
                    f"{pending.get('criterion', 'your request')}?"
                ),
            }
        )

        approved = _is_approval(decision)
        batch = DeletionBatch(
            batch_id=pending.get("batch_id", ""),
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            reports=[
                Report(
                    id=r["id"], user_id=state["user_id"],
                    thread_id=state["thread_id"], title=r.get("title", ""),
                    body="", summary="", sql_used="",
                    created_at=r.get("created_at", ""),
                )
                for r in reports
            ],
            criterion=pending.get("criterion", ""),
        )

        if not approved:
            services.reports.record_cancellation(batch)
            services.tracer.emit("deletion_confirm", "cancelled", count=len(reports))
            services.tracer.count("deletions_cancelled")
            return {
                "pending_deletion": {},
                "deletion_outcome": "cancelled",
                "answer": "Cancelled. Nothing was deleted.",
            }

        services.reports.register_confirmation(batch)
        try:
            count, replayed = services.reports.apply_deletion(
                batch.batch_id, state["user_id"]
            )
        except PermissionError as exc:
            services.tracer.emit("deletion_confirm", "denied", error=str(exc))
            return {
                "pending_deletion": {},
                "deletion_outcome": "denied",
                "answer": "I couldn't apply that deletion: it is not yours to delete.",
            }

        services.tracer.emit(
            "deletion_confirm", "applied", count=count, replayed=replayed,
            batch_id=batch.batch_id,
        )
        services.tracer.count("reports_deleted", count)

        return {
            "pending_deletion": {},
            "deletion_outcome": "applied",
            "answer": (
                f"Deleted {count} report{'s' if count != 1 else ''}. "
                f"They are recoverable — say \"restore batch "
                f"{batch.batch_id[:8]}\" if that was a mistake."
            ),
        }

    return deletion_confirm


def route_after_preview(state: AgentState) -> str:
    """Skip the confirmation entirely when nothing matched."""
    return "confirm" if state.get("pending_deletion", {}).get("reports") else "done"


# --- Helpers ----------------------------------------------------------------

#: Accepted affirmatives. Deliberately a closed list rather than a model call:
#: interpreting consent for a destructive action is not a job to delegate to a
#: probabilistic classifier, and anything unrecognised is treated as "no".
_APPROVALS = frozenset(
    {"yes", "y", "confirm", "confirmed", "approve", "approved", "delete",
     "do it", "go ahead", "proceed", "ok", "okay"}
)


def _is_approval(decision: Any) -> bool:
    """Default to refusal for anything not clearly an approval."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        if "approved" in decision:
            return bool(decision["approved"])
        decision = decision.get("value", "")
    if isinstance(decision, str):
        return decision.strip().lower() in _APPROVALS
    return False


def _title_of(body: str, fallback: str) -> str:
    """First heading or first line of the report, trimmed."""
    for line in body.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return fallback[:120]
