"""Shared test fixtures.

The agent is exercised end to end with a scripted model and a fixture-backed
warehouse, so the full graph — routing, guarding, the repair cycle, the
confirmation interrupt — runs with no network and no cloud credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from insight_agent.config import Settings
from insight_agent.data.recorded import RecordedExecutor, fingerprint
from insight_agent.golden.retriever import GoldenBucket
from insight_agent.graph.build import build_graph
from insight_agent.graph.services import build_services
from insight_agent.llm.client import LLMResponse, UsageLedger


class StubLLM:
    """Scripted stand-in for :class:`LLMClient`.

    Responses are keyed by a marker found in the system prompt, so a test
    controls what each node receives without knowing the call order.
    """

    def __init__(self, script: dict[str, Any] | None = None) -> None:
        self.script = script or {}
        self.calls: list[tuple[str, str]] = []
        self.ledger = UsageLedger()
        self.fail_with: Exception | None = None

    def _key(self, system: str | None) -> str:
        text = (system or "").lower()
        for marker in (
            "classify one message", "bigquery standard sql", "fixing a bigquery",
            "interpret query results", "matched no rows", "explain what data",
            "short analytical report", "presentation preferences",
            "detect whether the user",
        ):
            if marker in text:
                return marker
        return "default"

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> LLMResponse:
        if self.fail_with:
            raise self.fail_with
        key = self._key(system)
        self.calls.append((key, prompt[:80]))
        value = self.script.get(key, "")
        if callable(value):
            value = value(prompt)
        if isinstance(value, list):
            value = value.pop(0) if value else ""
        return LLMResponse(text=str(value), model="stub", input_tokens=10, output_tokens=5)

    def complete_json(
        self, prompt: str, *, system: str | None = None, **kwargs: Any
    ) -> tuple[dict, LLMResponse]:
        response = self.complete(prompt, system=system, **kwargs)
        try:
            return json.loads(response.text), response
        except (ValueError, TypeError):
            return {}, response


@pytest.fixture
def fixtures_dir(tmp_path) -> Path:
    directory = tmp_path / "fixtures"
    directory.mkdir()
    return directory


def write_fixture(directory: Path, sql: str, rows: list[dict], **extra: Any) -> None:
    """Record a query result so the executor can replay it."""
    path = directory / f"{fingerprint(sql)}.json"
    path.write_text(
        json.dumps(
            {"fingerprint": fingerprint(sql), "sql": sql, "rows": rows,
             "bytes_processed": 1_000_000, "bytes_billed": 1_000_000, **extra},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "state",
        golden_dir=Path("golden_bucket/trios"),
        personas_dir=Path("personas"),
        fixtures_dir=tmp_path / "fixtures",
        executor="recorded",
        llm_provider="google",
        google_api_key="test-key",
        gcp_project="test-project",
        max_sql_repair_attempts=2,
    )


@pytest.fixture
def make_agent(settings, fixtures_dir):
    """Build a compiled agent with a scripted model and replayed warehouse."""

    def _build(script: dict[str, Any] | None = None, *, checkpointer=None):
        settings.fixtures_dir = fixtures_dir
        llm = StubLLM(script)
        services = build_services(
            settings,
            llm=llm,
            executor=RecordedExecutor(fixtures_dir),
            golden=GoldenBucket(Path("golden_bucket/trios")),
        )
        if checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()
        agent = build_graph(services, checkpointer=checkpointer)
        return agent, services, llm

    return _build


@pytest.fixture
def run_turn():
    """Invoke one turn and return the resulting state."""

    def _run(agent, services, text: str, *, user: str = "ceo", thread: str = "t1"):
        from insight_agent.graph.state import new_turn

        tracer = services.new_tracer(user_id=user, thread_id=thread)
        state = new_turn(
            user_input=text, user_id=user, thread_id=thread,
            trace_id=tracer.trace_id, persona="",
        )
        return agent.invoke(state, {"configurable": {"thread_id": thread}})

    return _run
