"""The analysis pipeline: question in, grounded explanation out.

This is the part of the graph that is a cycle rather than a chain.  SQL
generation can fail in two different ways — rejected by the guard, or rejected
by BigQuery — and both feed the same repair node, which loops back.  The cycle
is bounded by ``sql_attempts``, which is the only thing standing between a
confused model and an infinite loop, so it is checked in the routing function
rather than inside a node where an early return could skip it.

The failure taxonomy earns its keep here.  Three different outcomes that all
look like "the query didn't work" are handled differently:

* **Repairable** (bad column, syntax, over budget) — show the model the error
  and let it try again, at most ``max_sql_repair_attempts`` times.
* **Terminal** (no permission, circuit open) — stop immediately. Retrying
  cannot succeed and costs the user time to reach the same failure.
* **Empty but valid** — not a failure at all. The query was right and the
  answer is "nothing matched". Retrying this is the classic wasteful loop, so
  it routes straight to interpretation instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from insight_agent.data.executor import (
    QueryCostError,
    QueryError,
    QueryPermissionError,
    QueryResult,
)
from insight_agent.graph.prompts import (
    ANALYST_SYSTEM,
    EMPTY_RESULT_SYSTEM,
    REPAIR_SYSTEM,
    SCHEMA_SYSTEM,
    analyst_prompt,
    repair_prompt,
    schema_prompt,
    sql_prompt,
    sql_system,
)
from insight_agent.graph.services import Services
from insight_agent.graph.state import AgentState
from insight_agent.llm.client import LLMUnavailableError
from insight_agent.resilience.breaker import CircuitOpenError
from insight_agent.security.scrubber import pseudonymize_frame
from insight_agent.store.db import record_audit

log = logging.getLogger(__name__)

#: How many result rows are handed to the model. Enough to reason over a
#: ranking or a trend; small enough to keep the context bounded and the cost
#: predictable regardless of how many rows the query returned.
MODEL_ROW_SAMPLE = 40


# --- Golden Bucket ----------------------------------------------------------


def make_golden_retriever(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Fetch analyst precedent for this question (requirement 1)."""

    def golden_retriever(state: AgentState) -> dict[str, Any]:
        question = state.get("user_input", "")
        try:
            examples = services.golden.examples(question, limit=3)
        except Exception as exc:  # noqa: BLE001
            # Precedent improves the answer; it is not required for one.
            log.warning("golden bucket lookup failed: %s", exc)
            services.tracer.emit("golden", "degraded", error=str(exc)[:200])
            return {"golden_examples": []}

        services.tracer.emit(
            "golden", "retrieved", count=len(examples),
            matched=" | ".join(e["question"][:50] for e in examples),
        )
        services.tracer.count("golden_examples_used", len(examples))
        return {"golden_examples": examples}

    return golden_retriever


# --- SQL generation ---------------------------------------------------------


def make_sql_generator(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    def sql_generator(state: AgentState) -> dict[str, Any]:
        principal = services.principal(state["user_id"])
        scope_note = (
            f"This user may only analyse {principal.scope_description()}. The "
            "scope filter is applied automatically; do not add it yourself."
            if not principal.is_unrestricted
            else ""
        )

        with services.tracer.span("sql_generator", model=services.settings.model_primary):
            try:
                response = services.llm.complete(
                    sql_prompt(
                        state["user_input"],
                        plan=state.get("plan", ""),
                        golden_examples=state.get("golden_examples") or [],
                        scope_note=scope_note,
                    ),
                    system=sql_system(services.settings.bq_dataset),
                )
            except (LLMUnavailableError, CircuitOpenError) as exc:
                services.tracer.count("llm_unavailable")
                return {
                    "sql": "",
                    "answer": (
                        "I can't reach the language model right now, so I can't "
                        "build a query for that. Please try again shortly."
                    ),
                    "degraded": True,
                    "warnings": [f"Model unavailable: {exc}"],
                }

        sql = _strip_fence(response.text)
        services.tracer.emit(
            "sql_generator", "generated", sql=sql, tokens=response.total_tokens,
            attempts=response.attempts, fell_back=response.fell_back,
        )
        return {"sql": sql, "sql_attempts": state.get("sql_attempts", 0) + 1}

    return sql_generator


def make_sql_guard_node(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Validate and rewrite. The enforcement point for requirement 2."""

    def sql_guard_node(state: AgentState) -> dict[str, Any]:
        sql = state.get("sql", "")
        if not sql:
            return {"guard_violations": ["No query was produced."]}

        principal = services.principal(state["user_id"])
        result = services.guard.check(sql, principal)

        if not result.ok:
            services.tracer.emit("sql_guard", "reject", reason=result.reason(), sql=sql)
            services.tracer.count("guard_rejections")
            _audit(
                services, state, "sql_guard.reject", result.reason()[:160], "blocked"
            )
            return {
                "guard_violations": result.violations,
                "sql_failures": [
                    *state.get("sql_failures", []),
                    {"sql": sql, "error": f"Rejected: {result.reason()}"},
                ],
            }

        services.tracer.emit(
            "sql_guard", "pass", scope_applied=result.scope_applied,
            row_limit=result.limit_applied,
        )
        return {
            "sql": result.sql,
            "guard_violations": [],
            "scope_applied": result.scope_applied,
            "warnings": result.warnings,
        }

    return sql_guard_node


# --- Execution --------------------------------------------------------------


def make_executor_node(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    def executor_node(state: AgentState) -> dict[str, Any]:
        sql = state["sql"]
        try:
            with services.tracer.span("bq_executor") as span:
                result: QueryResult = services.executor.execute(sql)
                span["rows"] = result.row_count
                span["bytes_billed"] = result.bytes_billed
                span["cache_hit"] = result.cache_hit
        except CircuitOpenError as exc:
            services.tracer.count("breaker_open")
            return _terminal(
                "The data warehouse is not responding, so I can't run that "
                "query right now. Nothing was changed. Please try again in a "
                "minute.",
                str(exc),
            )
        except QueryPermissionError as exc:
            # Never repaired and never retried: the query is fine, the access
            # is not, and another attempt reaches the same wall more slowly.
            services.tracer.count("permission_errors")
            _audit(services, state, "bq.permission_denied", str(exc)[:160], "error")
            return _terminal(
                "I don't have access to run that query. This needs a "
                "permissions change rather than a different question.",
                str(exc),
            )
        except QueryError as exc:
            kind = "cost" if isinstance(exc, QueryCostError) else "query"
            services.tracer.count(f"{kind}_errors")
            return {
                "sql_failures": [
                    *state.get("sql_failures", []),
                    {"sql": sql, "error": str(exc)},
                ],
                "result_meta": {"last_error": str(exc), "repairable": exc.repairable},
            }
        except Exception as exc:  # noqa: BLE001
            # Unclassified. Treat as terminal rather than looping on something
            # we do not understand.
            log.exception("unexpected executor failure")
            return _terminal(
                "Something went wrong running that query. The error has been "
                "recorded and nothing was changed.",
                str(exc),
            )

        frame, masked_columns = pseudonymize_frame(result.rows)
        rows = frame.head(MODEL_ROW_SAMPLE).to_dict(orient="records")

        services.tracer.emit(
            "bq_executor", "result", rows=result.row_count,
            bytes_billed=result.bytes_billed,
            cost_usd=round(result.estimated_cost_usd(), 6),
            masked_columns=masked_columns, cache_hit=result.cache_hit,
        )
        services.tracer.count("bytes_billed", result.bytes_billed)
        services.tracer.count("rows_returned", result.row_count)

        return {
            "rows": rows,
            "row_count": result.row_count,
            "result_empty": result.is_empty,
            "result_meta": {
                "bytes_billed": result.bytes_billed,
                "cost_usd": round(result.estimated_cost_usd(), 6),
                "duration_ms": round(result.duration_ms, 1),
                "cache_hit": result.cache_hit,
                "truncated": result.truncated,
                "job_id": result.job_id,
            },
            "sql_failures": [],
        }

    return executor_node


def make_sql_repair(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Rewrite a failed query, having been shown why it failed."""

    def sql_repair(state: AgentState) -> dict[str, Any]:
        failures = state.get("sql_failures", [])
        attempt = state.get("sql_attempts", 0)
        services.tracer.emit(
            "sql_repair", "start", attempt=attempt,
            last_error=failures[-1]["error"][:200] if failures else "",
        )

        with services.tracer.span("sql_repair_call"):
            try:
                response = services.llm.complete(
                    repair_prompt(
                        state["user_input"], failures,
                        sql_system(services.settings.bq_dataset),
                    ),
                    system=REPAIR_SYSTEM,
                )
            except (LLMUnavailableError, CircuitOpenError) as exc:
                return {
                    "answer": (
                        "That query failed and I can't reach the model to fix "
                        "it. Please try again shortly."
                    ),
                    "degraded": True,
                    "warnings": [str(exc)],
                }

        repaired = _strip_fence(response.text)
        services.tracer.emit("sql_repair", "recovered", sql=repaired, attempt=attempt)
        services.tracer.count("sql_repairs")
        return {"sql": repaired, "sql_attempts": attempt + 1, "guard_violations": []}

    return sql_repair


# --- Interpretation ---------------------------------------------------------


def make_analyst(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    def analyst(state: AgentState) -> dict[str, Any]:
        if state.get("answer"):
            return {}  # a terminal failure already produced the reply

        empty = state.get("result_empty", False)
        system = EMPTY_RESULT_SYSTEM if empty else ANALYST_SYSTEM

        with services.tracer.span("analyst", empty=empty):
            try:
                response = services.llm.complete(
                    analyst_prompt(
                        state["user_input"], state.get("sql", ""),
                        state.get("rows", []), state.get("row_count", 0),
                        bool(state.get("result_meta", {}).get("truncated")),
                    ),
                    system=system,
                    temperature=services.settings.llm_temperature_prose,
                )
            except (LLMUnavailableError, CircuitOpenError) as exc:
                # The data is in hand; only the prose failed. Show the numbers
                # rather than nothing.
                return {
                    "analysis": _fallback_table(state),
                    "degraded": True,
                    "warnings": [f"Explanation unavailable: {exc}"],
                }

        services.tracer.emit("analyst", "explained", tokens=response.total_tokens)
        return {"analysis": response.text.strip()}

    return analyst


def make_schema_agent(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Answer 'what data is available' without touching the warehouse."""

    def schema_agent(state: AgentState) -> dict[str, Any]:
        principal = services.principal(state["user_id"])
        with services.tracer.span("schema_agent"):
            try:
                response = services.llm.complete(
                    schema_prompt(
                        state["user_input"], services.settings.bq_dataset,
                        principal.scope_description(),
                    ),
                    system=SCHEMA_SYSTEM,
                    temperature=services.settings.llm_temperature_prose,
                )
            except (LLMUnavailableError, CircuitOpenError):
                from insight_agent.data.catalog import render_for_prompt

                return {
                    "analysis": (
                        "Here is the data available to you:\n\n"
                        + render_for_prompt(services.settings.bq_dataset)
                    ),
                    "degraded": True,
                }
        return {"analysis": response.text.strip()}

    return schema_agent


# --- Routing functions ------------------------------------------------------


def route_after_guard(state: AgentState) -> str:
    """Execute, repair, or give up. The bound on the cycle lives here."""
    from insight_agent.config import get_settings

    if state.get("answer"):
        return "give_up"
    if not state.get("guard_violations"):
        return "execute"
    if state.get("sql_attempts", 0) <= get_settings().max_sql_repair_attempts:
        return "repair"
    return "give_up"


def route_after_execute(state: AgentState) -> str:
    """Interpret, repair, or give up.

    An empty result goes to interpretation, not to repair. The query was
    valid; "no rows matched" is the answer, and retrying it is the loop that
    burns budget for nothing.
    """
    from insight_agent.config import get_settings

    if state.get("answer"):
        return "give_up"
    failures = state.get("sql_failures", [])
    if not failures:
        return "interpret"
    meta = state.get("result_meta", {})
    if not meta.get("repairable", True):
        return "give_up"
    if state.get("sql_attempts", 0) <= get_settings().max_sql_repair_attempts:
        return "repair"
    return "give_up"


def make_give_up(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Fail honestly.

    Requirement 5 says the system must attempt self-correction *before giving
    up*, which means giving up has to be a real state with a real message,
    not an exception that reaches the user as a stack trace.
    """

    def give_up(state: AgentState) -> dict[str, Any]:
        if state.get("answer"):
            return {}

        failures = state.get("sql_failures", [])
        attempts = state.get("sql_attempts", 0)
        services.tracer.emit("give_up", "exhausted", attempts=attempts,
                             failures=len(failures))
        services.tracer.count("give_ups")
        _audit(services, state, "analysis.give_up", state["user_input"][:120], "failed")

        detail = failures[-1]["error"] if failures else "no query could be built"
        return {
            "answer": (
                f"I couldn't answer that one. I tried {attempts} "
                f"{'query' if attempts == 1 else 'different queries'} and the "
                f"last problem was: {detail}\n\n"
                "Rephrasing it more specifically usually helps — naming the "
                "time period, the product category, or the metric you want."
            ),
            "degraded": True,
        }

    return give_up


# --- Helpers ----------------------------------------------------------------


def _strip_fence(text: str) -> str:
    """Remove a markdown fence if the model added one despite instructions."""
    import re

    cleaned = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.S | re.I)
    if fenced:
        cleaned = fenced.group(1)
    return cleaned.strip().rstrip(";").strip()


def _terminal(message: str, error: str) -> dict[str, Any]:
    return {"answer": message, "degraded": True, "warnings": [error[:300]]}


def _fallback_table(state: AgentState) -> str:
    """Render results without the model, for when only the prose call failed."""
    rows = state.get("rows", [])
    if not rows:
        return "The query returned no rows."
    headers = list(rows[0].keys())
    lines = [" | ".join(headers), "-|-".join("-" * len(h) for h in headers)]
    lines += [" | ".join(str(r.get(h, "")) for h in headers) for r in rows[:20]]
    return (
        "I retrieved the data but couldn't generate the written analysis:\n\n"
        + "\n".join(lines)
    )


def _audit(
    services: Services, state: AgentState, action: str, target: str, outcome: str
) -> None:
    try:
        record_audit(
            services.conn, user_id=state.get("user_id", "unknown"),
            thread_id=state.get("thread_id"), action=action, target=target,
            outcome=outcome, detail={"trace_id": state.get("trace_id", "")},
        )
        services.conn.commit()
    except Exception:  # noqa: BLE001
        log.debug("audit write failed", exc_info=True)
