"""Model access with the failure handling requirement 5 asks for.

Three defences, in order of cost: retry with jittered backoff on transient
failures only, step down a chain of fallback models, then open a circuit
breaker so a dead dependency fails fast instead of costing every turn the
full retry budget.

Token usage is recorded per call, because "without inflating costs" cannot be
verified without measuring it.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from insight_agent.config import Settings, get_settings
from insight_agent.llm.providers import build_chat_model
from insight_agent.resilience.breaker import REGISTRY, CircuitOpenError

log = logging.getLogger(__name__)

#: Substrings marking a failure that is worth another attempt.
_TRANSIENT_MARKERS = (
    "429", "rate limit", "resource exhausted", "quota",
    "503", "unavailable", "500", "internal",
    "deadline", "timeout", "timed out", "connection",
)


class LLMUnavailableError(RuntimeError):
    """Every strategy was exhausted. The caller must degrade, not crash."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    fell_back: bool = False
    duration_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageLedger:
    """Running token count for one turn, surfaced in the metrics line."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fallbacks: int = 0
    retries: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.retries += response.attempts - 1
        if response.fell_back:
            self.fallbacks += 1
        self.by_model[response.model] = self.by_model.get(response.model, 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _is_quota_exhausted(exc: Exception) -> bool:
    """Distinguish an exhausted quota from a momentary rate limit.

    Both surface as 429. A burst limit clears in seconds and is worth a
    retry; a daily or per-model quota does not, and the only useful response
    is to try a different model.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return "resource_exhausted" in text or "exceeded your current quota" in text


def is_transient(exc: Exception) -> bool:
    """Decide whether another attempt could plausibly succeed."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class LLMClient:
    """Thin, resilient wrapper over the Gemini chat models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.ledger = UsageLedger()
        self._models: dict[str, Any] = {}

    def _breaker_for(self, model: str):
        """One breaker per model, not one per provider.

        Found in live testing: models are rate-limited independently. A
        provider-wide breaker meant that exhausting the primary model's quota
        also refused calls to the fallback model, which was healthy, so the
        fallback path could never fire in exactly the situation it exists for.
        """
        return REGISTRY.get(
            f"llm:{self.settings.llm_provider}:{model}",
            failure_threshold=self.settings.breaker_failure_threshold,
            reset_timeout=self.settings.breaker_reset_seconds,
        )

    # --- Model construction ----------------------------------------------

    def _model(self, name: str, temperature: float):
        """Construct (and cache) a chat model for the configured provider."""
        key = f"{name}:{temperature}"
        if key not in self._models:
            self._models[key] = build_chat_model(
                self.settings.provider_spec,
                name,
                temperature=temperature,
                timeout=self.settings.llm_timeout_seconds,
                api_key=self.settings.llm_api_key,
            )
        return self._models[key]

    # --- Public API -------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        history: list[BaseMessage] | None = None,
    ) -> LLMResponse:
        """One completion, with retry, fallback and breaker applied."""
        primary = model or self.settings.model_primary
        temp = (
            temperature
            if temperature is not None
            else self.settings.llm_temperature_precise
        )

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.extend(history or [])
        messages.append(HumanMessage(content=prompt))

        try:
            response = self._attempt_with_retries(primary, temp, messages)
        except LLMUnavailableError as first_failure:
            # Step down the chain. On a shared free tier several models can be
            # rate-limited at once, so stopping after one alternative gives up
            # while a working model is still available.
            response = None
            for candidate in self.settings.provider_spec.fallbacks_for(primary):
                log.warning("falling back from %s to %s", primary, candidate)
                try:
                    response = self._attempt_with_retries(candidate, temp, messages)
                except LLMUnavailableError:
                    continue
                response.fell_back = True
                break
            if response is None:
                raise LLMUnavailableError(
                    f"No model could answer. Tried {primary} and "
                    f"{len(self.settings.provider_spec.fallbacks_for(primary))} "
                    f"fallback(s). Last error: {first_failure}"
                ) from first_failure

        self.ledger.record(response)
        return response

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        history: list[BaseMessage] | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Completion parsed as JSON.

        Returns ``({}, response)`` rather than raising when the model emits
        unparseable output, so a formatting slip degrades into a fallback path
        instead of a crash.
        """
        import json
        import re

        response = self.complete(
            prompt, system=system, model=model,
            temperature=self.settings.llm_temperature_precise, history=history,
        )
        text = response.text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if fenced:
            text = fenced.group(1).strip()
        try:
            parsed = json.loads(text)
            return (parsed if isinstance(parsed, dict) else {"value": parsed}), response
        except (ValueError, TypeError):
            log.warning("model returned unparseable JSON: %.200s", text)
            return {}, response

    # --- Retry machinery --------------------------------------------------

    def _attempt_with_retries(
        self, model_name: str, temperature: float, messages: list[BaseMessage]
    ) -> LLMResponse:
        last: Exception | None = None
        started = time.perf_counter()

        for attempt in range(1, self.settings.llm_max_attempts + 1):
            try:
                raw = self._breaker_for(model_name).call(
                    self._model(model_name, temperature).invoke, messages
                )
            except CircuitOpenError as exc:
                raise LLMUnavailableError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001
                last = exc
                if _is_quota_exhausted(exc):
                    # A daily quota does not refill in eight seconds. Retrying
                    # burns the user's time to reach the same error; the
                    # fallback model is the only thing that can help.
                    log.warning("%s has no remaining quota; falling back", model_name)
                    break
                if not is_transient(exc) or attempt == self.settings.llm_max_attempts:
                    break
                delay = min(
                    self.settings.llm_timeout_seconds,
                    (2 ** (attempt - 1)) * 0.75 * (1 + random.random()),
                )
                log.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    model_name, attempt, self.settings.llm_max_attempts,
                    type(exc).__name__, delay,
                )
                time.sleep(delay)
                continue

            return LLMResponse(
                text=_text_of(raw),
                model=model_name,
                input_tokens=_usage(raw, "input_tokens"),
                output_tokens=_usage(raw, "output_tokens"),
                attempts=attempt,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        raise LLMUnavailableError(
            f"{model_name} failed after {self.settings.llm_max_attempts} attempts: {last}"
        ) from last


def _text_of(message: Any) -> str:
    """Extract text whether the model returns a string or content blocks."""
    if isinstance(message, AIMessage) or hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts) or str(content)
    return str(message)


def _usage(message: Any, field_name: str) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    try:
        return int(usage.get(field_name, 0))
    except (AttributeError, TypeError, ValueError):
        return 0
