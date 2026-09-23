"""Everything the graph nodes depend on, assembled once.

Nodes are closures over a Services instance, which is what makes the graph
testable: a test builds Services with a recorded executor and a stub model,
and the same node code runs with no network.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from insight_agent.config import Settings, get_settings
from insight_agent.data.executor import QueryExecutor
from insight_agent.golden.retriever import GoldenBucket
from insight_agent.llm.client import LLMClient
from insight_agent.obs.tracing import Tracer
from insight_agent.personas.loader import Persona, PersonaLoader
from insight_agent.security.identity import Principal, get_principal
from insight_agent.security.sql_guard import SqlGuard, build_guard
from insight_agent.store.db import connect
from insight_agent.store.preferences import PreferenceStore
from insight_agent.store.reports import ReportStore

log = logging.getLogger(__name__)


@dataclass
class Services:
    """Dependency container passed to every node."""

    settings: Settings
    conn: sqlite3.Connection
    llm: LLMClient
    guard: SqlGuard
    reports: ReportStore
    preferences: PreferenceStore
    personas: PersonaLoader
    golden: GoldenBucket
    tracer: Tracer
    _executor: QueryExecutor | None = field(default=None, repr=False)
    _executor_error: str = field(default="", repr=False)

    # --- Lazily constructed data plane ------------------------------------

    @property
    def executor(self) -> QueryExecutor:
        """Build the warehouse client on first use.

        Deferred so that an unconfigured warehouse does not stop the process
        from starting; schema questions, saved reports and the confirmation
        flow all work without it.
        """
        if self._executor is None:
            self._executor = self._build_executor()
        return self._executor

    @property
    def executor_available(self) -> bool:
        try:
            _ = self.executor
            return True
        except Exception:  # noqa: BLE001
            return False

    def _build_executor(self) -> QueryExecutor:
        if self.settings.executor == "recorded":
            from insight_agent.data.recorded import RecordedExecutor

            return RecordedExecutor(self.settings.fixtures_dir)
        from insight_agent.data.bigquery import BigQueryExecutor

        return BigQueryExecutor(self.settings)

    # --- Per-turn context --------------------------------------------------

    def principal(self, user_id: str) -> Principal:
        return get_principal(user_id)

    def persona_for(self, user_id: str, override: str = "") -> Persona:
        """Resolve the persona: explicit override, learned preference, or role default."""
        if override:
            return self.personas.load(override)
        preferred = self.preferences.applied(user_id).get("persona")
        if preferred:
            return self.personas.load(preferred)
        return self.personas.load(self.principal(user_id).persona)

    def new_tracer(self, *, user_id: str, thread_id: str) -> Tracer:
        self.tracer = Tracer(
            self.conn, self.settings.trace_log, user_id=user_id, thread_id=thread_id
        )
        return self.tracer


def build_services(settings: Settings | None = None, **overrides: Any) -> Services:
    """Assemble the container. ``overrides`` lets tests substitute components."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    conn = connect(settings.app_db)

    services = Services(
        settings=settings,
        conn=conn,
        llm=overrides.get("llm") or LLMClient(settings),
        guard=overrides.get("guard")
        or build_guard(
            settings.bq_dataset,
            default_limit=settings.default_row_limit,
            max_limit=settings.max_row_limit,
        ),
        reports=ReportStore(conn),
        preferences=PreferenceStore(conn),
        personas=PersonaLoader(settings.personas_dir),
        golden=overrides.get("golden") or GoldenBucket(settings.golden_dir),
        tracer=Tracer(conn, settings.trace_log),
    )
    if "executor" in overrides:
        services._executor = overrides["executor"]
    return services
