"""Circuit breaker for outbound dependencies.

Retries handle a dependency that is briefly unwell.  They make things worse
when it is properly down: every user turn pays the full retry budget in
latency before failing anyway, and the retries themselves keep load on the
failing service.

The breaker converts that into a fast, cheap failure.  After
``failure_threshold`` consecutive failures the circuit opens and calls are
refused immediately.  After ``reset_timeout`` one trial call is allowed
through; success closes the circuit, failure re-opens it.

Refusal raises ``CircuitOpenError``, which callers catch to fall back — to a
second model, to cached data, or to an honest message — rather than to crash.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """The dependency is known to be failing; the call was not attempted."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"{name} is unavailable (circuit open). Retrying in {retry_after:.0f}s."
        )


class CircuitBreaker:
    """One breaker per dependency."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 4,
        reset_timeout: float = 60.0,
        on_state_change: Callable[[str, CircuitState], None] | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._on_state_change = on_state_change
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        """Promote OPEN to HALF_OPEN once the cooldown has elapsed."""
        if (
            self._state is CircuitState.OPEN
            and time.monotonic() - self._opened_at >= self.reset_timeout
        ):
            self._set(CircuitState.HALF_OPEN)

    def _set(self, state: CircuitState) -> None:
        if state is not self._state:
            self._state = state
            if self._on_state_change:
                self._on_state_change(self.name, state)

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``func`` unless the circuit is open."""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                raise CircuitOpenError(self.name, max(0.0, self.reset_timeout - elapsed))

        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._set(CircuitState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            # A failed trial call in HALF_OPEN re-opens immediately: the
            # dependency had its chance.
            if (
                self._state is CircuitState.HALF_OPEN
                or self._failures >= self.failure_threshold
            ):
                self._opened_at = time.monotonic()
                self._set(CircuitState.OPEN)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._set(CircuitState.CLOSED)

    def snapshot(self) -> dict[str, Any]:
        """State for the health line in the CLI and for metrics."""
        with self._lock:
            self._maybe_half_open()
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
            }


class BreakerRegistry:
    """Named breakers, so the CLI can report on all dependencies at once."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(
        self, name: str, *, failure_threshold: int = 4, reset_timeout: float = 60.0
    ) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(
                    name, failure_threshold=failure_threshold, reset_timeout=reset_timeout
                )
                self._breakers[name] = breaker
            return breaker

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [b.snapshot() for b in self._breakers.values()]

    def reset_all(self) -> None:
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


REGISTRY = BreakerRegistry()
