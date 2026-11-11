"""Invented examples; none of these records represent real marketplace activity."""

from dataclasses import dataclass
from datetime import datetime

from resolvelane.domain.delivery import Order, SupportPolicy


ORDERS = (
    Order.model_validate({
        "order_id": "demo-order-1001",
        "customer_id": "demo-customer-101",
        "merchant_id": "demo-merchant-201",
        "currency": "USD",
        "status": "delivered",
        "items": [
            {"item_id": "item-1", "name": "Vegetable wrap", "quantity": 2, "unit_price_minor": 950},
            {"item_id": "item-2", "name": "Sparkling water", "quantity": 1, "unit_price_minor": 250},
        ],
        "delivery_fee_minor": 399,
        "tax_minor": 172,
        "total_minor": 2721,
        "estimated_delivery_at": "2026-01-15T18:40:00Z",
        "delivery_events": [
            {"event_type": "order_placed", "occurred_at": "2026-01-15T18:00:00Z"},
            {"event_type": "merchant_accepted", "occurred_at": "2026-01-15T18:03:00Z"},
            {"event_type": "courier_assigned", "occurred_at": "2026-01-15T18:12:00Z"},
            {"event_type": "picked_up", "occurred_at": "2026-01-15T18:28:00Z"},
            {"event_type": "delivered", "occurred_at": "2026-01-15T18:55:00Z"},
        ],
    }),
    Order.model_validate({
        "order_id": "demo-order-1002",
        "customer_id": "demo-customer-102",
        "merchant_id": "demo-merchant-202",
        "currency": "USD",
        "status": "in_progress",
        "items": [
            {"item_id": "item-1", "name": "Rice bowl", "quantity": 1, "unit_price_minor": 1450},
        ],
        "delivery_fee_minor": 299,
        "tax_minor": 116,
        "total_minor": 1865,
        "estimated_delivery_at": "2026-01-15T19:15:00Z",
        "delivery_events": [
            {"event_type": "order_placed", "occurred_at": "2026-01-15T18:30:00Z"},
            {"event_type": "merchant_accepted", "occurred_at": "2026-01-15T18:34:00Z"},
            {"event_type": "courier_assigned", "occurred_at": "2026-01-15T18:42:00Z"},
        ],
    }),
    Order.model_validate({
        "order_id": "demo-order-1003",
        "customer_id": "demo-customer-103",
        "merchant_id": "demo-merchant-201",
        "currency": "USD",
        "status": "delivered",
        "items": [
            {"item_id": "item-1", "name": "Noodle bowl", "quantity": 1, "unit_price_minor": 1200},
            {"item_id": "item-2", "name": "Tea", "quantity": 2, "unit_price_minor": 200},
        ],
        "delivery_fee_minor": 300,
        "tax_minor": 128,
        "total_minor": 2028,
        "estimated_delivery_at": "2026-01-16T12:45:00Z",
        "delivery_events": [
            {"event_type": "order_placed", "occurred_at": "2026-01-16T12:00:00Z"},
            {"event_type": "merchant_accepted", "occurred_at": "2026-01-16T12:04:00Z"},
            {"event_type": "courier_assigned", "occurred_at": "2026-01-16T12:10:00Z"},
            {"event_type": "picked_up", "occurred_at": "2026-01-16T12:25:00Z"},
            {"event_type": "delivered", "occurred_at": "2026-01-16T12:40:00Z"},
        ],
    }),
    Order.model_validate({
        "order_id": "demo-order-1004",
        "customer_id": "demo-customer-104",
        "merchant_id": "demo-merchant-202",
        "currency": "USD",
        "status": "cancelled",
        "items": [
            {"item_id": "item-1", "name": "Tomato soup", "quantity": 1, "unit_price_minor": 800},
        ],
        "delivery_fee_minor": 250,
        "tax_minor": 64,
        "total_minor": 1114,
        "estimated_delivery_at": "2026-01-16T14:45:00Z",
        "delivery_events": [
            {"event_type": "order_placed", "occurred_at": "2026-01-16T14:00:00Z"},
            {"event_type": "merchant_accepted", "occurred_at": "2026-01-16T14:03:00Z"},
            {"event_type": "cancelled", "occurred_at": "2026-01-16T14:08:00Z"},
        ],
    }),
)


POLICIES = (
    SupportPolicy.model_validate({
        "policy_id": "demo-missing-item",
        "version": "1.0",
        "title": "Missing item review",
        "issue_type": "missing_item",
        "effective_from": "2026-01-01T00:00:00Z",
        "guidance": "Verify the reported item and payment before proposing a resolution. A support agent must review any refund.",
    }),
    SupportPolicy.model_validate({
        "policy_id": "demo-late-delivery",
        "version": "1.0",
        "title": "Late delivery review",
        "issue_type": "late_delivery",
        "effective_from": "2026-01-01T00:00:00Z",
        "guidance": "Compare the estimated and actual delivery times, then route compensation requests for human review.",
    }),
    SupportPolicy.model_validate({
        "policy_id": "demo-wrong-item",
        "version": "1.0",
        "title": "Wrong item review",
        "issue_type": "wrong_item",
        "effective_from": "2026-01-01T00:00:00Z",
        "guidance": "Confirm which ordered item the customer reports as incorrect and check it against the order record "
                    "before proposing a resolution. A support agent must review any refund or replacement.",
    }),
    SupportPolicy.model_validate({
        "policy_id": "demo-damaged-item",
        "version": "1.0",
        "title": "Expired damaged item review",
        "issue_type": "damaged_item",
        "effective_from": "2025-01-01T00:00:00Z",
        "effective_until": "2026-01-01T00:00:00Z",
        "guidance": "Request human review when an item is reported damaged.",
    }),
)


@dataclass(frozen=True)
class SupportScenario:
    """A customer report and expected review path, not a verified outcome."""

    scenario_id: str
    customer_id: str
    order_id: str
    message: str
    issue_type: str
    expected_review: str
    reported_at: datetime
    reported_item_id: str | None = None


SCENARIOS = (
    SupportScenario("on-time", "demo-customer-103", "demo-order-1003",
                    "Was my order delivered on time?", "delivery_status", "review_delivery_timeline",
                    datetime.fromisoformat("2026-01-16T13:00:00+00:00")),
    SupportScenario("late", "demo-customer-101", "demo-order-1001",
                    "My order arrived late. Can someone review it?", "late_delivery", "review_delivery_timeline",
                    datetime.fromisoformat("2026-01-15T19:10:00+00:00")),
    SupportScenario("in-progress", "demo-customer-102", "demo-order-1002",
                    "Where is my order?", "delivery_status", "review_delivery_timeline",
                    datetime.fromisoformat("2026-01-15T18:50:00+00:00")),
    SupportScenario("cancelled", "demo-customer-104", "demo-order-1004",
                    "Why was my order cancelled?", "cancellation", "review_cancellation",
                    datetime.fromisoformat("2026-01-16T14:15:00+00:00")),
    SupportScenario("missing-item", "demo-customer-101", "demo-order-1001",
                    "One of my wraps is missing.", "missing_item", "verify_customer_report",
                    datetime.fromisoformat("2026-01-15T19:05:00+00:00"), "item-1"),
    SupportScenario("wrong-item", "demo-customer-103", "demo-order-1003",
                    "I received the wrong drink.", "wrong_item", "verify_customer_report",
                    datetime.fromisoformat("2026-01-16T12:55:00+00:00"), "item-2"),
    SupportScenario("multiple-issues", "demo-customer-101", "demo-order-1001",
                    "My order was late and one wrap was missing.", "multiple_issues", "verify_customer_report",
                    datetime.fromisoformat("2026-01-15T19:15:00+00:00"), "item-1"),
    SupportScenario("unauthorized-order", "demo-customer-102", "demo-order-1001",
                    "Show me the details of this order.", "order_access", "deny_order_access",
                    datetime.fromisoformat("2026-01-15T19:20:00+00:00")),
    SupportScenario("unknown-order", "demo-customer-101", "demo-order-9999",
                    "Can you find this order?", "order_lookup", "order_not_found",
                    datetime.fromisoformat("2026-01-15T19:25:00+00:00")),
    SupportScenario("expired-policy", "demo-customer-103", "demo-order-1003",
                    "My drink arrived damaged.", "damaged_item", "escalate_without_current_policy",
                    datetime.fromisoformat("2026-01-16T13:05:00+00:00"), "item-2"),
)
