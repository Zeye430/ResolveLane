"""First version of the support assistant: keyword issue detection and replies built from order facts.

The customer's message is treated as a report, not a fact. Replies state what the order
record shows, and anything that needs a decision is flagged for a support agent.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from resolvelane.domain.delivery import Order, OrderStatus


class Issue(str, Enum):
    LATE_DELIVERY = "late_delivery"
    MISSING_ITEM = "missing_item"
    WRONG_ITEM = "wrong_item"
    DAMAGED_ITEM = "damaged_item"
    CANCELLATION = "cancellation"
    DELIVERY_STATUS = "delivery_status"
    UNKNOWN = "unknown"


_PATTERNS = {
    Issue.LATE_DELIVERY: r"\blate\b|\bdelayed\b|took (too long|forever)",
    Issue.MISSING_ITEM: r"\bmissing\b|didn't (get|receive)|did not (get|receive)|\bforgot\b",
    Issue.WRONG_ITEM: r"\bwrong\b|\bincorrect\b|not what i ordered",
    Issue.DAMAGED_ITEM: r"\bdamaged\b|\bspilled\b|\bcrushed\b|\bbroken\b",
    Issue.CANCELLATION: r"\bcancel",
    Issue.DELIVERY_STATUS: r"where is|where's|\bstatus\b|on time|when will|\btrack",
}

_ITEM_ISSUES = {Issue.MISSING_ITEM: "missing", Issue.WRONG_ITEM: "wrong", Issue.DAMAGED_ITEM: "damaged"}

HELP_TEXT = ("I can help with delivery status, late deliveries, cancellations, "
             "and missing, wrong or damaged items.")


def detect_issues(message: str) -> list[Issue]:
    text = message.lower()
    issues = [issue for issue, pattern in _PATTERNS.items() if re.search(pattern, text)]
    # A specific problem ("arrived late") also matches generic status words; keep the specific one.
    specific = [issue for issue in issues if issue is not Issue.DELIVERY_STATUS]
    return specific or issues or [Issue.UNKNOWN]


@dataclass(frozen=True)
class AssistantReply:
    reply: str
    issues: list[Issue]
    needs_human_review: bool


def _time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%H:%M UTC on %b %d")


def _minutes_late(order: Order) -> int:
    """Minutes between the estimate and delivery; zero or negative means on time. Delivered orders only."""
    delivered_at = order.delivery_events[-1].occurred_at
    return int((delivered_at - order.estimated_delivery_at).total_seconds() // 60)


def _status_sentence(order: Order) -> str:
    last = order.delivery_events[-1]
    if order.status is OrderStatus.CANCELLED:
        return f"Order {order.order_id} was cancelled at {_time(last.occurred_at)}."
    estimate = _time(order.estimated_delivery_at)
    if order.status is OrderStatus.DELIVERED:
        minutes_late = _minutes_late(order)
        timing = (f"{minutes_late} minutes after the estimated time of {estimate}" if minutes_late > 0
                  else f"within the estimated time of {estimate}")
        return f"Order {order.order_id} was delivered at {_time(last.occurred_at)}, {timing}."
    step = last.event_type.value.replace("_", " ")
    return (f"Order {order.order_id} is on its way. Latest update: {step} at {_time(last.occurred_at)}. "
            f"Estimated delivery is {estimate}.")


def _late_delivery(order: Order) -> tuple[str, bool]:
    late = order.status is OrderStatus.DELIVERED and _minutes_late(order) > 0
    return _status_sentence(order), late


def _cancellation(order: Order) -> tuple[str, bool]:
    if order.status is OrderStatus.CANCELLED:
        return _status_sentence(order) + " The record does not include a reason.", True
    return f"Order {order.order_id} has not been cancelled. " + _status_sentence(order), False


def _item_report(order: Order, issues: list[Issue]) -> tuple[str, bool]:
    if order.status is not OrderStatus.DELIVERED:
        return f"Order {order.order_id} has not been delivered yet. " + _status_sentence(order), False
    kinds = " and ".join(_ITEM_ISSUES[issue] for issue in issues)
    contents = ", ".join(f"{item.quantity} x {item.name}" for item in order.items)
    return f"I've noted your report of a {kinds} item. The order contained: {contents}.", True


def respond(message: str, order_id: str | None, order: Order | None) -> AssistantReply:
    """Build a reply. ``order`` is None when no order was given or it is not available to the customer."""
    issues = detect_issues(message)

    if order_id is None:
        prefix = "" if issues != [Issue.UNKNOWN] else HELP_TEXT + " "
        return AssistantReply(prefix + "Please share your order number so I can check it.", issues, False)
    if order is None:
        return AssistantReply(f"I couldn't find order {order_id} on your account. "
                              "Please check the order number.", issues, False)

    parts: list[str] = []
    needs_review = False
    if Issue.LATE_DELIVERY in issues:
        text, review = _late_delivery(order)
        parts.append(text)
        needs_review |= review
    if Issue.CANCELLATION in issues:
        text, review = _cancellation(order)
        parts.append(text)
        needs_review |= review
    item_issues = [issue for issue in issues if issue in _ITEM_ISSUES]
    if item_issues:
        text, review = _item_report(order, item_issues)
        parts.append(text)
        needs_review |= review
    if issues == [Issue.DELIVERY_STATUS]:
        parts.append(_status_sentence(order))
    if issues == [Issue.UNKNOWN]:
        parts += [_status_sentence(order), HELP_TEXT]

    if needs_review:
        parts.append("I've passed this to a support agent for review. Refunds are not issued automatically.")
    return AssistantReply(" ".join(parts), issues, needs_review)
