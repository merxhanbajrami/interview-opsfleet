"""Final nodes: learn, present, scrub.

Order matters and is not arbitrary.  Formatting runs before scrubbing, so the
scrubber sees exactly the bytes the user will see.  Scrubbing first and then
letting a model rewrite the text would put an unchecked generation step after
the last control, which is the same mistake as trusting the prompt.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from insight_agent.graph.prompts import (
    FORMATTER_SYSTEM,
    PREFERENCE_SYSTEM,
    formatter_prompt,
)
from insight_agent.graph.services import Services
from insight_agent.graph.state import AgentState
from insight_agent.llm.client import LLMUnavailableError
from insight_agent.resilience.breaker import CircuitOpenError
from insight_agent.security.scrubber import scrub_text
from insight_agent.store.db import record_audit
from insight_agent.store.preferences import KNOWN_KEYS

log = logging.getLogger(__name__)

#: Words that suggest the user is talking about presentation rather than data.
_PREFERENCE_HINT = re.compile(
    r"\b(table|tabular|bullet|bullets|list|chart|graph|prose|paragraph|"
    r"short|shorter|brief|briefly|concise|detail|detailed|deep|depth|"
    r"summar\w+|sql|query|prefer|always|from now on|stop showing|"
    r"action items?|format)\b",
    re.I,
)


def make_preference_learner(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Notice lasting presentation preferences (requirement 4, user level).

    Runs on the fast model and never blocks the answer: a failure here costs
    a missed observation, nothing more.
    """

    def preference_learner(state: AgentState) -> dict[str, Any]:
        text = state.get("user_input", "")
        if len(text) < 8 or state.get("intent") == "refused":
            return {}
        # Cheap pre-filter. Most turns contain no preference signal at all, and
        # a model call on every turn to discover that would be a standing tax
        # on latency and cost for a rare event.
        if not _PREFERENCE_HINT.search(text):
            return {}

        try:
            parsed, _ = services.llm.complete_json(
                f"User message:\n{text}",
                system=PREFERENCE_SYSTEM,
                model=services.settings.model_fast,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("preference detection skipped: %s", exc)
            return {}

        learned: list[str] = []
        for item in parsed.get("preferences", []) or []:
            if not isinstance(item, dict):
                continue
            key, value = str(item.get("key", "")), str(item.get("value", ""))
            if key not in KNOWN_KEYS or not value:
                continue
            preference = services.preferences.observe(
                state["user_id"], key, value, explicit=bool(item.get("explicit"))
            )
            if preference:
                learned.append(preference.describe())

        if learned:
            services.tracer.emit("preference_learner", "observed", learned=" ; ".join(learned))
            services.tracer.count("preferences_observed", len(learned))
        return {}

    return preference_learner


def make_formatter(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Apply persona and learned preferences to the finished content."""

    def formatter(state: AgentState) -> dict[str, Any]:
        # A refusal or a terminal failure is already final. Re-writing it
        # through a persona risks softening a refusal into a maybe.
        if state.get("refusal"):
            return {"answer": state["refusal"]}
        if state.get("answer"):
            return {}

        content = state.get("analysis", "")
        if not content:
            return {"answer": "I wasn't able to produce an answer for that."}

        persona = services.persona_for(state["user_id"], state.get("persona", ""))
        preference_section = services.preferences.as_prompt_section(state["user_id"])

        if persona.load_error:
            services.tracer.emit("formatter", "persona_error", error=persona.load_error)

        # Nothing to apply, and a model call that changes nothing is waste.
        if not preference_section and not persona.tone and not persona.guidance:
            return {"answer": content}

        with services.tracer.span("formatter", persona=persona.name):
            try:
                response = services.llm.complete(
                    formatter_prompt(content, persona.as_prompt_section(), preference_section),
                    system=FORMATTER_SYSTEM,
                    temperature=services.settings.llm_temperature_prose,
                )
            except (LLMUnavailableError, CircuitOpenError):
                # Presentation is the least important thing to get right.
                # Ship the correct content unformatted.
                return {
                    "answer": content,
                    "warnings": ["Applied default formatting; persona unavailable."],
                }

        return {"answer": response.text.strip()}

    return formatter


def make_output_guard(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Scrub the exact text about to be shown.

    A redaction here means an earlier layer let something through, so it is
    counted and audited rather than silently fixed.
    """

    def output_guard(state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        if not answer:
            return {}

        report = scrub_text(answer)
        if not report.clean:
            services.tracer.emit(
                "output_guard", "redacted",
                redactions=str(report.redactions),
                total=report.total_redactions,
            )
            services.tracer.count("output_redactions", report.total_redactions)
            try:
                record_audit(
                    services.conn, user_id=state.get("user_id", "unknown"),
                    thread_id=state.get("thread_id"), action="output_guard.redact",
                    target=state.get("trace_id", ""), outcome="redacted",
                    detail={"redactions": report.redactions},
                )
                services.conn.commit()
            except Exception:  # noqa: BLE001
                log.debug("audit write failed", exc_info=True)
            return {"answer": report.text}

        services.tracer.emit("output_guard", "clean")
        return {}

    return output_guard


def make_followup(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Discuss the previous result without running a new query.

    Requirement: executives must be able to *discuss* the data, not only
    query it. Re-querying for "what do you mean by that" is slow and costs
    money for an answer already in context.
    """

    def followup(state: AgentState) -> dict[str, Any]:
        from insight_agent.graph.prompts import ANALYST_SYSTEM

        rows = state.get("rows", [])
        if not rows:
            # Nothing to discuss; treat it as a fresh question instead.
            return {"intent": "analysis"}

        import json

        prompt = (
            f"Earlier result ({state.get('row_count', 0)} rows) from this query:\n"
            f"{state.get('sql', '')}\n\n"
            f"{json.dumps(rows[:25], indent=2, default=str)}\n\n"
            f"Previous explanation:\n{state.get('analysis', '')}\n\n"
            f"Follow-up question: {state['user_input']}"
        )
        with services.tracer.span("followup"):
            try:
                response = services.llm.complete(
                    prompt, system=ANALYST_SYSTEM,
                    temperature=services.settings.llm_temperature_prose,
                )
            except (LLMUnavailableError, CircuitOpenError) as exc:
                return {
                    "answer": "I can't reach the model to answer that follow-up.",
                    "degraded": True, "warnings": [str(exc)],
                }
        return {"analysis": response.text.strip()}

    return followup


def make_refusal(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    def refusal(state: AgentState) -> dict[str, Any]:
        return {"answer": state.get("refusal") or "I can't help with that one."}

    return refusal
