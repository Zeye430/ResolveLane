"""Refunds: proposed by the assistant or a support agent, decided by a support agent.

A refund is a series of immutable versions. Revising a proposal adds a version; nothing that
was proposed is overwritten, so the record shows what each author put forward. A support
agent approves or rejects one exact version, the current one, and the decision is final.

Amounts are computed by code from the order record (resolvelane.refunds), never taken from
the model or the request.

An approved refund gets one payment: the instruction to the payment provider, written in the
same transaction as the approval (an outbox) and sent afterwards (resolvelane.payments).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RefundStatus(str, Enum):
    PROPOSED = "proposed"  # waiting for a support agent's decision
    APPROVED = "approved"
    REJECTED = "rejected"


class PaymentStatus(str, Enum):
    PENDING = "pending"          # to be sent, or sent without a definite answer yet
    SUCCEEDED = "succeeded"      # the provider made the refund
    FAILED = "failed"            # the provider declined it; a support agent decides what next
    UNCONFIRMED = "unconfirmed"  # no definite answer after every attempt; checked, never sent again


FINAL_PAYMENT_STATUSES = (PaymentStatus.SUCCEEDED, PaymentStatus.FAILED)


@dataclass
class Payment:
    payment_id: str            # also the idempotency key sent to the provider
    refund_id: str
    amount_minor: int          # the approved version's amount
    currency: str
    created_at: datetime
    next_attempt_at: datetime
    status: PaymentStatus = PaymentStatus.PENDING
    attempts: int = 0
    provider_refund_id: str | None = None
    last_error: str = ""
    completed_at: datetime | None = None


class RefundDecided(Exception):
    """The refund was already approved or rejected."""


@dataclass(frozen=True)
class RefundLine:
    item_id: str
    name: str
    quantity: int
    amount_minor: int  # quantity x the unit price on the order record


@dataclass(frozen=True)
class RefundVersion:
    version: int
    lines: tuple[RefundLine, ...]
    delivery_fee_minor: int  # 0, or the order's whole delivery fee
    reason: str
    author_role: str         # assistant or support_agent
    author_id: str | None    # the support agent; None for the assistant
    created_at: datetime

    @property
    def amount_minor(self) -> int:
        return sum(line.amount_minor for line in self.lines) + self.delivery_fee_minor

    def same_content(self, other: "RefundVersion") -> bool:
        return (self.lines, self.delivery_fee_minor) == (other.lines, other.delivery_fee_minor)


@dataclass
class Refund:
    refund_id: str
    case_id: str
    order_id: str
    currency: str
    created_at: datetime
    versions: list[RefundVersion] = field(default_factory=list)
    status: RefundStatus = RefundStatus.PROPOSED
    decided_version: int | None = None
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_note: str = ""
    payment: Payment | None = None  # created when the refund is approved

    @property
    def current(self) -> RefundVersion:
        return self.versions[-1]

    @property
    def decided(self) -> RefundVersion | None:
        return self.versions[self.decided_version - 1] if self.decided_version else None

    def revise(self, version: RefundVersion) -> None:
        if self.status is not RefundStatus.PROPOSED:
            raise RefundDecided
        self.versions.append(version)

    def decide(self, status: RefundStatus, version: int, at: datetime, agent_id: str, note: str) -> None:
        if self.status is not RefundStatus.PROPOSED:
            raise RefundDecided
        self.status, self.decided_version = status, version
        self.decided_at, self.decided_by, self.decision_note = at, agent_id, note
