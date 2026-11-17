"""Sending approved refunds to the payment provider.

Approving a refund writes its payment in the same transaction as the approval: an outbox.
Nothing is sent while the case is locked. A dispatcher sends due payments afterwards, in
three steps, like customer messages:

1. Under the lock: claim the payment by counting the attempt and moving its next attempt
   forward (a lease), so it is not picked up again while this attempt runs.
2. Without the lock: ask the provider. The payment ID is the idempotency key, so the
   provider refunds at most once however often it is asked.
3. Under the lock: record the outcome.

An attempt that gets no definite answer, such as a timeout, may have refunded the customer
or not. The next attempt therefore first looks the refund up at the provider by its key
(reconciliation) and sends it again only if the provider has no record of it. The same
happens after a crash between sending and recording: the lease expires and the next attempt
finds the provider's record.

After MAX_ATTEMPTS without a definite answer, the payment is unconfirmed: it is never sent
again but still looked up, and the case goes back to the support agents' queue. A declined
payment also goes back to the queue. The customer is told when the refund has been sent.

Set RESOLVELANE_PAYMENTS=paused to keep approved payments in the outbox without sending
them, for example during a provider incident; they are sent once it is unset.
"""

import logging
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from resolvelane import metrics
from resolvelane.case_store import CaseStore
from resolvelane.domain.case import Case, CaseMessage, CaseStatus, MessageRole
from resolvelane.domain.refund import FINAL_PAYMENT_STATUSES, Payment, PaymentStatus, Refund
from resolvelane.refunds import describe, money
from resolvelane.simulator.payment_provider import ProviderRefund

logger = logging.getLogger("resolvelane.payments")

LEASE = timedelta(seconds=60)          # longer than any provider call may take
RETRY_DELAYS = tuple(timedelta(seconds=seconds) for seconds in (2, 8, 30, 120))
MAX_ATTEMPTS = len(RETRY_DELAYS) + 1
CHECK_UNCONFIRMED_EVERY = timedelta(minutes=10)
POLL_SECONDS = 2.0


class PaymentProvider(Protocol):
    def create_refund(self, idempotency_key: str, amount_minor: int, currency: str,
                      reference: str) -> ProviderRefund: ...

    def find_refund(self, idempotency_key: str) -> ProviderRefund | None: ...


def new_payment(refund: Refund, now: datetime) -> Payment:
    version = refund.decided
    return Payment(f"payment-{uuid.uuid4().hex}", refund.refund_id, version.amount_minor, refund.currency, now, now)


@dataclass(frozen=True)
class _Claim:
    case_id: str
    payment: Payment   # a copy, as claimed
    reconcile: bool    # an earlier attempt got no definite answer
    send: bool         # False for an unconfirmed payment: only look it up


def _find(case: Case, payment_id: str) -> tuple[Refund, Payment]:
    refund = next(item for item in case.refunds if item.payment and item.payment.payment_id == payment_id)
    return refund, refund.payment


def _claim(cases: CaseStore, case_id: str, payment_id: str) -> _Claim | None:
    with cases.lock:
        case = cases.get(case_id)
        _, payment = _find(case, payment_id)
        now = cases.clock()
        if payment.status in FINAL_PAYMENT_STATUSES or payment.next_attempt_at > now:
            return None  # handled meanwhile
        claim = _Claim(case_id, Payment(**vars(payment)), payment.attempts > 0,
                       payment.status is PaymentStatus.PENDING)
        if claim.send:
            payment.attempts += 1
        payment.next_attempt_at = now + LEASE
        cases.commit(case)
        return claim


def _ask(provider: PaymentProvider, claim: _Claim) -> tuple[ProviderRefund | None, str]:
    """The provider's refund, or None and why there is no definite answer."""
    payment = claim.payment
    started = time.perf_counter()
    try:
        if claim.reconcile or not claim.send:
            found = provider.find_refund(payment.payment_id)
            if found is not None or not claim.send:
                return found, "" if found else "no record at the provider"
        return provider.create_refund(payment.payment_id, payment.amount_minor, payment.currency,
                                      payment.refund_id), ""
    except Exception as error:  # a timeout, an outage, or anything else: the outcome is not known
        return None, f"{type(error).__name__}: {error}"
    finally:
        metrics.PROVIDER_SECONDS.observe(time.perf_counter() - started)


def _to_review(case: Case, now: datetime, note: str) -> None:
    case.review_notes.append(note)
    case.move_to(CaseStatus.NEEDS_REVIEW, now, "system", None, note)


def _record(cases: CaseStore, claim: _Claim, result: ProviderRefund | None, error: str) -> PaymentStatus:
    with cases.lock:
        case = cases.get(claim.case_id)
        refund, payment = _find(case, claim.payment.payment_id)
        now = cases.clock()
        if payment.status in FINAL_PAYMENT_STATUSES:
            return payment.status
        if result is not None and result.succeeded:
            payment.status, payment.provider_refund_id, payment.completed_at = (
                PaymentStatus.SUCCEEDED, result.provider_refund_id, now)
            case.messages.append(CaseMessage(
                MessageRole.SYSTEM, f"Your refund of {money(payment.amount_minor, payment.currency)} for "
                                    f"{describe(refund.decided)} has been sent to your original payment method.", now))
        elif result is not None:
            payment.status, payment.provider_refund_id, payment.completed_at = (
                PaymentStatus.FAILED, result.provider_refund_id, now)
            payment.last_error = result.failure_reason
            _to_review(case, now, f"Refund payment declined by the provider: {result.failure_reason}")
        elif payment.status is PaymentStatus.UNCONFIRMED:
            payment.next_attempt_at = now + CHECK_UNCONFIRMED_EVERY
        else:
            payment.last_error = error
            if payment.attempts >= MAX_ATTEMPTS:
                payment.status = PaymentStatus.UNCONFIRMED
                payment.next_attempt_at = now + CHECK_UNCONFIRMED_EVERY
                _to_review(case, now, f"Refund payment unconfirmed after {payment.attempts} attempts ({error}); "
                                      "it is checked with the provider but not sent again")
            else:
                payment.next_attempt_at = now + RETRY_DELAYS[payment.attempts - 1]
        cases.commit(case)
        metrics.PAYMENT_ATTEMPTS.labels(
            "succeeded" if result is not None and result.succeeded else "declined" if result is not None
            else "no_answer" if claim.send else "checked").inc()
        logger.info("payment case=%s payment=%s attempt=%d status=%s reconciled=%s error=%s", case.case_id,
                    payment.payment_id, payment.attempts, payment.status.value, claim.reconcile, error or "-")
        return payment.status


def send_due(cases: CaseStore, provider: PaymentProvider, limit: int = 20) -> int:
    """Attempt every payment that is due; returns how many were attempted."""
    attempted = 0
    for case_id, payment_id in cases.due_payments(cases.clock(), limit):
        claim = _claim(cases, case_id, payment_id)
        if claim is None:
            continue
        result, error = _ask(provider, claim)
        _record(cases, claim, result, error)
        attempted += 1
    return attempted


class PaymentDispatcher:
    """Sends due payments on a background thread, when poked and every few seconds."""

    def __init__(self, cases: CaseStore, provider: PaymentProvider, poll_seconds: float = POLL_SECONDS):
        self.cases = cases
        self.provider = provider
        self.poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name="payments", daemon=True)
        self._thread.start()

    def poke(self) -> None:
        self._wake.set()

    def close(self) -> None:
        self._stopped.set()
        self._wake.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                send_due(self.cases, self.provider)
            except Exception:
                logger.exception("payment_dispatch_failed")
            self._wake.wait(self.poll_seconds)
            self._wake.clear()


def load_payment_dispatcher(cases: CaseStore, provider: PaymentProvider,
                            environ: Mapping[str, str] = os.environ) -> PaymentDispatcher | None:
    if environ.get("RESOLVELANE_PAYMENTS", "").strip().lower() == "paused":
        logger.warning("RESOLVELANE_PAYMENTS is paused; approved refunds wait in the outbox")
        return None
    return PaymentDispatcher(cases, provider)
