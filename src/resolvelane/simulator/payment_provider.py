"""A simulated payment provider that refunds to the customer's original payment method.

It behaves like a real provider in the ways that matter for safety:

- Requests carry an idempotency key. A repeated key returns the first result instead of
  refunding again; the same key with a different amount is refused.
- A request can time out after the provider has made the refund, so the caller cannot tell
  whether it happened. ``find_refund`` looks a refund up by its key, so the caller can
  check before sending again.
- A refund can be declined, for example when the original card was closed.

Faults are scripted for tests and demos: each create_refund call takes the next fault from
the list, if any. Nothing here moves real money.
"""

import itertools
import logging
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

logger = logging.getLogger("resolvelane.payment_provider")

TIMEOUT_BEFORE = "timeout_before"  # no answer, and nothing was refunded
TIMEOUT_AFTER = "timeout_after"    # no answer, but the refund was made
UNAVAILABLE = "unavailable"        # refused before processing
DECLINE = "decline"                # processed and declined
FAULTS = (TIMEOUT_BEFORE, TIMEOUT_AFTER, UNAVAILABLE, DECLINE)


class ProviderTimeout(Exception):
    """No answer in time: the refund may or may not have been made."""


class ProviderUnavailable(Exception):
    """The provider refused the request before processing it."""


class ProviderKeyMismatch(Exception):
    """The idempotency key was used before for a different refund."""


@dataclass(frozen=True)
class ProviderRefund:
    provider_refund_id: str
    idempotency_key: str
    amount_minor: int
    currency: str
    reference: str
    succeeded: bool
    failure_reason: str = ""


class SimulatedPaymentProvider:
    def __init__(self, faults: Iterable[str] = ()):
        self.faults = list(faults)
        unknown = set(self.faults) - set(FAULTS)
        if unknown:
            raise ValueError(f"Unknown payment faults: {', '.join(sorted(unknown))}")
        self._refunds: dict[str, ProviderRefund] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.create_calls = 0

    def create_refund(self, idempotency_key: str, amount_minor: int, currency: str,
                      reference: str) -> ProviderRefund:
        with self._lock:
            self.create_calls += 1
            fault = self.faults.pop(0) if self.faults else None
            existing = self._refunds.get(idempotency_key)
            if existing is not None:
                if (existing.amount_minor, existing.currency) != (amount_minor, currency):
                    raise ProviderKeyMismatch(idempotency_key)
                return existing
            if fault == UNAVAILABLE:
                raise ProviderUnavailable("payment provider unavailable")
            if fault == TIMEOUT_BEFORE:
                raise ProviderTimeout("payment provider timed out")
            refund = ProviderRefund(f"sim-refund-{next(self._ids):04d}", idempotency_key, amount_minor, currency,
                                    reference, fault != DECLINE,
                                    "The original payment method can no longer receive refunds"
                                    if fault == DECLINE else "")
            self._refunds[idempotency_key] = refund
            logger.info("provider_refund key=%s id=%s succeeded=%s", idempotency_key, refund.provider_refund_id,
                        refund.succeeded)
            if fault == TIMEOUT_AFTER:
                raise ProviderTimeout("payment provider timed out")
            return refund

    def find_refund(self, idempotency_key: str) -> ProviderRefund | None:
        with self._lock:
            return self._refunds.get(idempotency_key)

    def refunds(self) -> list[ProviderRefund]:
        with self._lock:
            return list(self._refunds.values())


def load_payment_provider(environ: Mapping[str, str] = os.environ) -> SimulatedPaymentProvider:
    """RESOLVELANE_PAYMENT_FAULTS, e.g. "timeout_after,unavailable", scripts faults for a demo."""
    faults = [item.strip() for item in environ.get("RESOLVELANE_PAYMENT_FAULTS", "").split(",") if item.strip()]
    if faults:
        logger.warning("simulated payment provider will fail with: %s", ",".join(faults))
    return SimulatedPaymentProvider(faults)
