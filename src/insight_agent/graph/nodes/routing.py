"""Entry nodes: what kind of turn is this, and is it allowed at all.

Refusal here is a convenience, not a security control. A classifier can be
talked around; the SQL guard and output scrubber run regardless. What this
node buys is cost: an off-topic turn pays for one cheap classification
instead of the whole analysis pipeline.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from insight_agent.graph.prompts import ROUTER_SYSTEM, router_prompt
from insight_agent.graph.services import Services
from insight_agent.graph.state import AgentState
from insight_agent.store.db import record_audit

log = logging.getLogger(__name__)

#: Cheap pre-filter for the most common subversion patterns. Catching these
#: without a model call keeps an obvious attack from costing anything.
_SUBVERSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+"
               r"(system\s+)?(prompt|instructions?|rules)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(a|an)\b", re.I),
    re.compile(r"\bdisregard\s+(your|all|the)\b", re.I),
    re.compile(r"\b(developer|admin|root)\s+mode\b", re.I),
    re.compile(r"\bpretend\s+(you|to\s+be)\b", re.I),
)

#: Direct requests for personal data. The SQL guard would block the query
#: anyway; refusing here gives the user a clear reason instead of a
#: confusing "that column does not exist".
_PII_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(email|e-mail)\s*(address(es)?)?\b", re.I),
    re.compile(r"\b(home\s+)?address(es)?\b", re.I),
    re.compile(r"\b(full|first|last|customer)\s+names?\b", re.I),
    re.compile(r"\bphone\s+numbers?\b", re.I),
    re.compile(r"\bpostal\s+code|zip\s*code\b", re.I),
    re.compile(r"\bwho\s+(is|are)\s+(the\s+)?customers?\s+(called|named)\b", re.I),
)

REFUSAL_OFF_TOPIC = (
    "I can only help with analysis of the retail sales, product and customer "
    "data in this dataset. Ask me about revenue, products, customers or "
    "trends and I will dig into it."
)

REFUSAL_SUBVERSION = (
    "I can't change how I work or share my configuration. I'm here to analyse "
    "the retail data. Ask me a question about sales, products or customers."
)

REFUSAL_PII = (
    "I can't return personal details about individual customers: names, email "
    "addresses, postal addresses or locations are masked at the data layer. I "
    "can analyse customer behaviour in aggregate, and I can rank customers by "
    "value using anonymous identifiers. Would either of those help?"
)


def make_input_guard(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Cheap pattern checks before any model call."""

    def input_guard(state: AgentState) -> dict[str, Any]:
        text = state.get("user_input", "")
        tracer = services.tracer

        for pattern in _SUBVERSION_PATTERNS:
            if pattern.search(text):
                tracer.emit("input_guard", "reject", kind="subversion",
                            pattern=pattern.pattern[:60])
                tracer.count("input_rejections")
                _audit(services, state, "input_guard.subversion", pattern.pattern[:80])
                return {"intent": "refused", "refusal": REFUSAL_SUBVERSION}

        for pattern in _PII_REQUEST_PATTERNS:
            if pattern.search(text):
                tracer.emit("input_guard", "reject", kind="pii_request",
                            pattern=pattern.pattern[:60])
                tracer.count("input_rejections")
                _audit(services, state, "input_guard.pii_request", pattern.pattern[:80])
                return {"intent": "refused", "refusal": REFUSAL_PII}

        tracer.emit("input_guard", "pass")
        return {}

    return input_guard


def make_router(services: Services) -> Callable[[AgentState], dict[str, Any]]:
    """Classify the turn with the fast model."""

    def router(state: AgentState) -> dict[str, Any]:
        if state.get("intent") == "refused":
            return {}

        tracer = services.tracer
        text = state.get("user_input", "")
        had_result = bool(state.get("rows"))

        with tracer.span("router", model=services.settings.model_fast):
            try:
                parsed, response = services.llm.complete_json(
                    router_prompt(text, had_result),
                    system=ROUTER_SYSTEM,
                    model=services.settings.model_fast,
                )
            except Exception as exc:  # noqa: BLE001
                # The router failing must not end the turn. Analysis is the
                # safe default: it is the most common intent, and every
                # dangerous path has its own gate further down.
                log.warning("router failed, defaulting to analysis: %s", exc)
                tracer.emit("router", "degraded", error=str(exc)[:200])
                return {
                    "intent": "analysis",
                    "route_reason": "router unavailable, defaulted",
                    "degraded": True,
                    "warnings": ["Intent classification was unavailable."],
                }

        intent = str(parsed.get("intent", "analysis")).strip().lower()
        if intent not in {
            "analysis", "schema", "report", "delete", "list_reports",
            "followup", "refused",
        }:
            intent = "analysis"

        update: dict[str, Any] = {
            "intent": intent,
            "route_reason": str(parsed.get("reason", ""))[:120],
        }

        if intent == "refused":
            update["refusal"] = REFUSAL_OFF_TOPIC
            tracer.count("input_rejections")
            _audit(services, state, "router.off_topic", text[:120])

        if intent == "delete":
            # Carry the parsed target through; the deletion node re-resolves it
            # against the store rather than trusting these strings.
            update["pending_deletion"] = {
                "mentions": str(parsed.get("mentions", "")).strip(),
                "this_conversation": bool(parsed.get("this_conversation", False)),
            }

        tracer.emit("router", "classified", intent=intent,
                    reason=update["route_reason"], tokens=response.total_tokens)
        return update

    return router


def route_from_intent(state: AgentState) -> str:
    """Conditional edge: the branch of the DAG this turn takes."""
    intent = state.get("intent", "analysis")
    if intent == "refused":
        return "refused"
    if intent == "schema":
        return "schema"
    if intent == "delete":
        return "delete"
    if intent == "list_reports":
        return "list_reports"
    if intent == "followup":
        return "followup"
    return "analysis"


def _audit(services: Services, state: AgentState, action: str, target: str) -> None:
    try:
        record_audit(
            services.conn,
            user_id=state.get("user_id", "unknown"),
            thread_id=state.get("thread_id"),
            action=action,
            target=target,
            outcome="refused",
            detail={"trace_id": state.get("trace_id", "")},
        )
        services.conn.commit()
    except Exception:  # noqa: BLE001
        log.debug("audit write failed", exc_info=True)
