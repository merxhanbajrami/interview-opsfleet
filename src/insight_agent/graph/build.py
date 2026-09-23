"""Graph assembly.

The shape of this graph is the design decision worth defending.  It is a
directed graph with branches and one bounded cycle, not a chain, and the
difference is not cosmetic:

* **Branches** mean a schema question never pays for SQL generation, and a
  deletion never touches the warehouse.  In a linear chain every turn runs
  every step and the irrelevant ones are told to do nothing.
* **The cycle** is what self-correction is.  ``sql_repair`` routes back to
  ``sql_guard``, so a repaired query is re-validated rather than trusted —
  a repair that introduced a PII column would otherwise walk straight past
  the control that exists to stop it.
* **Convergence** on ``output_guard`` means every path out of the graph passes
  the final scrub. A new branch added later inherits that automatically,
  because the edge into ``END`` is not available anywhere else.

The checkpointer is not an optimisation. ``interrupt()`` requires persisted
state, so the confirmation flow in requirement 3 depends on it, and durable
conversation state across restarts falls out of the same mechanism.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from insight_agent.graph.nodes.analysis import (
    make_analyst,
    make_executor_node,
    make_give_up,
    make_golden_retriever,
    make_schema_agent,
    make_sql_generator,
    make_sql_guard_node,
    make_sql_repair,
    route_after_execute,
    route_after_guard,
)
from insight_agent.graph.nodes.output import (
    make_followup,
    make_formatter,
    make_output_guard,
    make_preference_learner,
    make_refusal,
)
from insight_agent.graph.nodes.reporting import (
    make_deletion_confirm,
    make_deletion_preview,
    make_list_reports,
    make_report_composer,
    route_after_preview,
)
from insight_agent.graph.nodes.routing import (
    make_input_guard,
    make_router,
    route_from_intent,
)
from insight_agent.graph.services import Services
from insight_agent.graph.state import AgentState

#: Applied to nodes whose only realistic failure is a transient network
#: problem. Jittered backoff, so a rate limit does not turn into a thundering
#: herd of retries from concurrent turns.
_NETWORK_RETRY = RetryPolicy(
    initial_interval=0.5, backoff_factor=2.0, max_interval=8.0,
    max_attempts=2, jitter=True,
)


def build_graph(services: Services, checkpointer=None):
    """Wire the nodes into the agent graph and compile it."""
    graph = StateGraph(AgentState)

    # --- Nodes ------------------------------------------------------------
    graph.add_node("input_guard", make_input_guard(services))
    graph.add_node("preference_learner", make_preference_learner(services),
                   retry_policy=_NETWORK_RETRY)
    graph.add_node("router", make_router(services), retry_policy=_NETWORK_RETRY)

    graph.add_node("golden", make_golden_retriever(services))
    graph.add_node("sql_generator", make_sql_generator(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("sql_guard", make_sql_guard_node(services))
    graph.add_node("executor", make_executor_node(services))
    graph.add_node("sql_repair", make_sql_repair(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("analyst", make_analyst(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("give_up", make_give_up(services))

    graph.add_node("schema_agent", make_schema_agent(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("followup", make_followup(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("report_composer", make_report_composer(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("list_reports", make_list_reports(services))

    graph.add_node("deletion_preview", make_deletion_preview(services))
    graph.add_node("deletion_confirm", make_deletion_confirm(services))

    graph.add_node("refusal", make_refusal(services))
    graph.add_node("formatter", make_formatter(services), retry_policy=_NETWORK_RETRY)
    graph.add_node("output_guard", make_output_guard(services))

    # --- Entry ------------------------------------------------------------
    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "preference_learner")
    graph.add_edge("preference_learner", "router")

    # --- The branch -------------------------------------------------------
    graph.add_conditional_edges(
        "router",
        route_from_intent,
        {
            "analysis": "golden",
            "schema": "schema_agent",
            "delete": "deletion_preview",
            "list_reports": "list_reports",
            "followup": "followup",
            "refused": "refusal",
        },
    )

    # --- Analysis pipeline, with the self-correction cycle -----------------
    graph.add_edge("golden", "sql_generator")
    graph.add_edge("sql_generator", "sql_guard")

    graph.add_conditional_edges(
        "sql_guard",
        route_after_guard,
        {"execute": "executor", "repair": "sql_repair", "give_up": "give_up"},
    )
    graph.add_conditional_edges(
        "executor",
        route_after_execute,
        {"interpret": "analyst", "repair": "sql_repair", "give_up": "give_up"},
    )
    # The cycle. A repaired query is re-validated, never trusted.
    graph.add_edge("sql_repair", "sql_guard")

    # An analysis turn that was really a report request continues into
    # composition; everything else goes straight to presentation.
    graph.add_conditional_edges(
        "analyst",
        lambda state: "report" if state.get("intent") == "report" else "present",
        {"report": "report_composer", "present": "formatter"},
    )
    graph.add_edge("report_composer", "formatter")

    # --- Other branches ---------------------------------------------------
    graph.add_edge("schema_agent", "formatter")
    graph.add_edge("followup", "formatter")
    graph.add_edge("give_up", "output_guard")
    graph.add_edge("list_reports", "output_guard")
    graph.add_edge("refusal", "output_guard")

    graph.add_conditional_edges(
        "deletion_preview",
        route_after_preview,
        {"confirm": "deletion_confirm", "done": "output_guard"},
    )
    graph.add_edge("deletion_confirm", "output_guard")

    # --- Single exit ------------------------------------------------------
    graph.add_edge("formatter", "output_guard")
    graph.add_edge("output_guard", END)

    return graph.compile(checkpointer=checkpointer)


def build_checkpointer(path: Path) -> SqliteSaver:
    """Durable conversation state.

    Constructed from an explicit connection rather than the documented
    ``from_conn_string`` context manager, which closes the database when the
    block exits — fine for a script, wrong for a long-running CLI session.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def build_agent(services: Services):
    """The compiled graph with its checkpointer, ready to run."""
    return build_graph(services, checkpointer=build_checkpointer(services.settings.checkpoint_db))
