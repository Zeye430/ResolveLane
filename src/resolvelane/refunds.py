"""Pricing a refund from the order record.

Whoever proposes a refund, the assistant or a support agent, names only order items and
quantities, and whether the delivery fee is included. The amount is computed here from the
unit prices on the order, so no one can propose an amount the order does not support.

What one order can be refunded is limited across all of its approved refunds, in every
case: each item up to the quantity ordered, and the delivery fee once. Tax is not refunded
in this simulation; a real marketplace would compute it with its tax provider.
"""

import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from resolvelane.domain.delivery import Order, OrderStatus
from resolvelane.domain.refund import Refund, RefundLine, RefundStatus, RefundVersion


class RefundInvalid(Exception):
    """The proposal does not fit the order; the message says why and is safe to show."""


@dataclass(frozen=True)
class Refunded:
    """What approved refunds of an order already cover."""
    quantities: Mapping[str, int] = field(default_factory=dict)
    delivery_fee: bool = False

    @classmethod
    def of(cls, versions: Iterable[RefundVersion]) -> "Refunded":
        quantities: Counter[str] = Counter()
        fee = False
        for version in versions:
            for line in version.lines:
                quantities[line.item_id] += line.quantity
            fee |= version.delivery_fee_minor > 0
        return cls(dict(quantities), fee)


@dataclass(frozen=True)
class RefundProposal:
    """What a proposer asks for; priced by ``price_refund``."""
    items: tuple[tuple[str, int], ...]  # (item_id, quantity)
    include_delivery_fee: bool
    reason: str


def price_refund(order: Order, proposal: RefundProposal, refunded: Refunded) -> tuple[tuple[RefundLine, ...], int]:
    """The refund lines and delivery fee for ``proposal``, or RefundInvalid."""
    if order.status is not OrderStatus.DELIVERED:
        raise RefundInvalid(f"Order {order.order_id} has not been delivered; only delivered orders are refunded here")
    if not proposal.items and not proposal.include_delivery_fee:
        raise RefundInvalid("Name at least one item or include the delivery fee")
    ids = [item_id for item_id, _ in proposal.items]
    if len(ids) != len(set(ids)):
        raise RefundInvalid("Name each item once")
    ordered = {item.item_id: item for item in order.items}
    lines = []
    for item_id, quantity in proposal.items:
        item = ordered.get(item_id)
        if item is None:
            raise RefundInvalid(f"Order {order.order_id} has no item {item_id}")
        remaining = item.quantity - refunded.quantities.get(item_id, 0)
        if not 1 <= quantity <= remaining:
            raise RefundInvalid(f"{remaining} x {item.name} ({item_id}) can still be refunded, not {quantity}")
        lines.append(RefundLine(item_id, item.name, quantity, quantity * item.unit_price_minor))
    if proposal.include_delivery_fee and refunded.delivery_fee:
        raise RefundInvalid("The delivery fee was already refunded")
    return tuple(lines), order.delivery_fee_minor if proposal.include_delivery_fee else 0


def new_version(order: Order, proposal: RefundProposal, refunded: Refunded, number: int, author_role: str,
                author_id: str | None, now: datetime) -> RefundVersion:
    lines, fee = price_refund(order, proposal, refunded)
    return RefundVersion(number, lines, fee, proposal.reason, author_role, author_id, now)


def new_refund(case_id: str, order: Order, version: RefundVersion, now: datetime) -> Refund:
    return Refund(f"refund-{uuid.uuid4().hex}", case_id, order.order_id, order.currency, now, [version])


def open_refund(refunds: Sequence[Refund]) -> Refund | None:
    """The case's refund waiting for a decision; a case has at most one."""
    return next((refund for refund in refunds if refund.status is RefundStatus.PROPOSED), None)


def payment_state(refund: Refund) -> str | None:
    """How an approved refund's payment stands, as customers are told: sent, processing (being
    sent), or delayed (declined or unconfirmed, so a support agent is looking into it)."""
    if refund.payment is None:
        return None
    return {"succeeded": "sent", "pending": "processing"}.get(refund.payment.status.value, "delayed")


def money(minor: int, currency: str) -> str:
    return f"{minor / 100:.2f} {currency}"


def describe(version: RefundVersion) -> str:
    """The refunded items in words, e.g. '1 x Vegetable wrap and the delivery fee'."""
    parts = [f"{line.quantity} x {line.name}" for line in version.lines]
    if version.delivery_fee_minor:
        parts.append("the delivery fee")
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
