"""The state carried through the graph.

One ``TypedDict`` for the whole turn.  Keeping it flat and explicit matters
more than it looks: this structure is what gets checkpointed to SQLite, so it
is also the thing that has to survive a process restart and be resumed after a
human confirmation.  Anything not in here does not survive an interrupt.

Reducers are used only where two branches can legitimately both write.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

#: What the router decided the turn is.
Intent = Literal[
    "analysis",    # a data question that needs SQL
    "schema",      # a question about what data exists
    "report",      # produce and save a written report
    "delete",      # destructive action on the reports library
    "list_reports",
    "followup",    # discussion of the previous result, no new query needed
    "refused",     # off-topic, or an attempt to subvert the agent
]


class AgentState(TypedDict, total=False):
    """Everything one conversation turn needs."""

    # --- Conversation -----------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str

    # --- Identity and context --------------------------------------------
    user_id: str
    thread_id: str
    trace_id: str
    persona: str

    # --- Routing ----------------------------------------------------------
    intent: Intent
    route_reason: str
    refusal: str

    # --- Analysis pipeline ------------------------------------------------
    plan: str
    golden_examples: list[dict[str, Any]]
    sql: str
    #: Bounded. The self-correction cycle reads this to decide whether to
    #: try again or give up, which is what stops the graph looping forever.
    sql_attempts: int
    #: Every failed attempt, kept so a repair prompt can avoid repeating one.
    #: Deliberately NOT an accumulating reducer. A successful execution has to
    #: be able to clear it, and with `operator.add` returning [] appends
    #: nothing instead of resetting, which would leave a stale failure in
    #: state and send a query that already succeeded back into the repair
    #: cycle. Nodes append explicitly instead.
    sql_failures: list[dict[str, str]]
    guard_violations: list[str]
    scope_applied: bool

    # --- Results ----------------------------------------------------------
    rows: list[dict[str, Any]]
    row_count: int
    result_empty: bool
    result_meta: dict[str, Any]

    # --- Findings and output ---------------------------------------------
    analysis: str
    answer: str
    saved_report_id: str

    # --- Destructive path -------------------------------------------------
    #: Serialised DeletionBatch. Must be plain data: it is checkpointed and
    #: read back after the human confirmation, possibly in another process.
    pending_deletion: dict[str, Any]
    deletion_outcome: str

    # --- Diagnostics ------------------------------------------------------
    warnings: Annotated[list[str], operator.add]
    degraded: bool


def new_turn(
    *, user_input: str, user_id: str, thread_id: str, trace_id: str, persona: str
) -> AgentState:
    """Fresh per-turn fields.

    Conversation history is restored by the checkpointer; everything derived
    from a single question starts empty so that a previous turn's SQL or rows
    can never be mistaken for this one's.
    """
    return AgentState(
        user_input=user_input,
        user_id=user_id,
        thread_id=thread_id,
        trace_id=trace_id,
        persona=persona,
        intent="analysis",
        route_reason="",
        refusal="",
        plan="",
        golden_examples=[],
        sql="",
        sql_attempts=0,
        guard_violations=[],
        scope_applied=False,
        rows=[],
        row_count=0,
        result_empty=False,
        result_meta={},
        analysis="",
        answer="",
        saved_report_id="",
        pending_deletion={},
        deletion_outcome="",
        degraded=False,
    )
