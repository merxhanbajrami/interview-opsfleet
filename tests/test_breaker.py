"""Circuit breaker behaviour (requirement 5: third-party failure resilience)."""

from __future__ import annotations

import pytest

from insight_agent.resilience.breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


def _boom():
    raise RuntimeError("dependency down")


def test_starts_closed_and_passes_calls_through():
    breaker = CircuitBreaker("bq", failure_threshold=3)
    assert breaker.call(lambda: 42) == 42
    assert breaker.state is CircuitState.CLOSED


def test_opens_after_the_threshold():
    breaker = CircuitBreaker("bq", failure_threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)
    assert breaker.state is CircuitState.OPEN


def test_open_circuit_refuses_without_calling():
    breaker = CircuitBreaker("bq", failure_threshold=1)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)

    calls = []
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: calls.append(1))
    assert not calls, "open circuit still invoked the dependency"


def test_a_success_resets_the_failure_count():
    breaker = CircuitBreaker("bq", failure_threshold=3)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)
    breaker.call(lambda: "ok")
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    assert breaker.state is CircuitState.CLOSED


def test_half_opens_after_the_cooldown():
    breaker = CircuitBreaker("bq", failure_threshold=1, reset_timeout=0.0)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    assert breaker.state is CircuitState.HALF_OPEN


def test_a_failed_trial_call_reopens_immediately():
    """The trial call is one chance, not a reset of the failure budget."""
    breaker = CircuitBreaker("bq", failure_threshold=5, reset_timeout=0.0)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    # Threshold is 5, so a single failure has not opened it yet.
    assert breaker.state is CircuitState.CLOSED

    breaker = CircuitBreaker("bq", failure_threshold=1, reset_timeout=0.0)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    assert breaker.state is CircuitState.HALF_OPEN  # zero cooldown, trial allowed

    # Lengthen the cooldown so the reopened state is observable.
    breaker.reset_timeout = 60.0
    with pytest.raises(RuntimeError):
        breaker.call(_boom)  # the trial call fails
    assert breaker.state is CircuitState.OPEN


def test_a_successful_trial_call_closes_the_circuit():
    breaker = CircuitBreaker("bq", failure_threshold=1, reset_timeout=0.0)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state is CircuitState.CLOSED


def test_the_error_reports_when_to_retry():
    breaker = CircuitBreaker("gemini", failure_threshold=1, reset_timeout=30.0)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    with pytest.raises(CircuitOpenError) as exc:
        breaker.call(lambda: None)
    assert exc.value.retry_after > 0
    assert "gemini" in str(exc.value)


def test_state_changes_are_reported_for_metrics():
    seen = []
    breaker = CircuitBreaker(
        "bq", failure_threshold=1, on_state_change=lambda n, s: seen.append((n, s))
    )
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    assert ("bq", CircuitState.OPEN) in seen


# --- Per-model isolation ----------------------------------------------------


def test_breakers_are_isolated_per_model():
    """Found in live testing against the Gemini free tier.

    Models are rate-limited independently. A provider-wide breaker meant that
    exhausting the primary model's quota also refused calls to the fallback
    model, which was healthy -- so the fallback path could never fire in
    exactly the situation it exists for.
    """
    from insight_agent.resilience.breaker import BreakerRegistry

    registry = BreakerRegistry()
    primary = registry.get("llm:google:gemini-3.8-flash", failure_threshold=2)
    fallback = registry.get("llm:google:gemini-3.5-flash", failure_threshold=2)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            primary.call(_boom)

    assert primary.state is CircuitState.OPEN
    assert fallback.state is CircuitState.CLOSED, "an unrelated model was cut off"
    assert fallback.call(lambda: "answered") == "answered"


def test_an_exhausted_quota_is_not_retried():
    """A daily quota does not refill in eight seconds.

    Retrying it burns the user's time to reach the same error; only a
    different model can help.
    """
    from insight_agent.llm.client import _is_quota_exhausted, is_transient

    quota = RuntimeError(
        "429 RESOURCE_EXHAUSTED. You exceeded your current quota, "
        "please check your plan and billing details."
    )
    burst = RuntimeError("429 Too Many Requests: rate limit exceeded, slow down")

    assert _is_quota_exhausted(quota)
    assert not _is_quota_exhausted(burst)
    # Both are still transient, so neither is treated as a permanent failure.
    assert is_transient(quota) and is_transient(burst)
