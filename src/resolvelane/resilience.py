"""Circuit breaker for calls to external providers.

Closed: calls go through. After enough consecutive failures the breaker opens and calls are
skipped, so customers do not wait for timeouts while a provider is down. When the cooldown has
passed, a single probe call is let through: success closes the breaker, failure reopens it.
"""

import logging
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("resolvelane.resilience")

_BREAKERS: "weakref.WeakSet[CircuitBreaker]" = weakref.WeakSet()


def breakers() -> list["CircuitBreaker"]:
    """The breakers in use, for monitoring."""
    return sorted(_BREAKERS, key=lambda breaker: breaker.name)


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class Ticket:
    """Permission for one call. Results of calls admitted before the breaker last opened are ignored."""
    generation: int
    probe: bool


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int, cooldown_s: float,
                 clock: Callable[[], float] = time.monotonic):
        if failure_threshold < 1 or cooldown_s <= 0:
            raise ValueError("failure_threshold must be at least 1 and cooldown_s positive")
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._generation = 0
        self._opened_at: float | None = None
        self._probe_started_at: float | None = None
        _BREAKERS.add(self)

    @property
    def state(self) -> BreakerState:
        with self._lock:
            if self._opened_at is None:
                return BreakerState.CLOSED
            return BreakerState.HALF_OPEN if self._probe_started_at is not None else BreakerState.OPEN

    def acquire(self) -> Ticket | None:
        """Return a ticket if a call may be made now, or None if it should be skipped."""
        with self._lock:
            if self._opened_at is None:
                return Ticket(self._generation, probe=False)
            now = self._clock()
            if now - self._opened_at < self.cooldown_s:
                return None
            # One probe at a time; a probe that never reported back is replaced after a cooldown.
            if self._probe_started_at is not None and now - self._probe_started_at < self.cooldown_s:
                return None
            self._probe_started_at = now
            return Ticket(self._generation, probe=True)

    def succeeded(self, ticket: Ticket) -> None:
        with self._lock:
            if ticket.generation != self._generation:
                return  # admitted before the breaker opened; says nothing about the provider now
            if self._opened_at is not None and not ticket.probe:
                return
            if self._opened_at is not None:
                logger.info("circuit_closed breaker=%s", self.name)
            self._failures = 0
            self._opened_at = None
            self._probe_started_at = None

    def failed(self, ticket: Ticket) -> None:
        with self._lock:
            if ticket.generation != self._generation:
                return
            if self._opened_at is not None and not ticket.probe:
                return
            self._failures += 1
            if ticket.probe or self._failures >= self.failure_threshold:
                self._generation += 1
                self._opened_at = self._clock()
                self._probe_started_at = None
                logger.warning("circuit_opened breaker=%s failures=%d cooldown_s=%s",
                               self.name, self._failures, self.cooldown_s)
