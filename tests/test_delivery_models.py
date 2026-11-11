from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from resolvelane.domain.delivery import Order, SupportPolicy
from resolvelane.simulator.fixtures import ORDERS, POLICIES, SCENARIOS


def test_demo_fixtures_are_unique_and_have_consistent_totals():
    assert len({order.order_id for order in ORDERS}) == len(ORDERS)
    assert len({(policy.policy_id, policy.version) for policy in POLICIES}) == len(POLICIES)
    for order in ORDERS:
        subtotal = sum(item.quantity * item.unit_price_minor for item in order.items)
        assert order.total_minor == subtotal + order.delivery_fee_minor + order.tax_minor


@pytest.mark.parametrize("field,value", [
    ("total_minor", 1),
    ("currency", "usd"),
    ("delivery_fee_minor", -1),
    ("tax_minor", 1.5),
    ("estimated_delivery_at", "2026-01-15T18:40:00"),
])
def test_invalid_order_values_are_rejected(field, value):
    data = ORDERS[0].model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        Order.model_validate(data)


def test_out_of_order_delivery_events_are_rejected():
    data = ORDERS[0].model_dump()
    events = list(data["delivery_events"])
    events[1], events[2] = events[2], events[1]
    data["delivery_events"] = events
    with pytest.raises(ValidationError, match="chronological"):
        Order.model_validate(data)


def test_terminal_event_must_match_status():
    data = ORDERS[0].model_dump()
    data["status"] = "in_progress"
    with pytest.raises(ValidationError, match="terminal"):
        Order.model_validate(data)


def test_delivered_status_with_cancelled_final_event_is_rejected():
    data = ORDERS[0].model_dump()
    data["delivery_events"][-1]["event_type"] = "cancelled"
    with pytest.raises(ValidationError, match="final delivery event"):
        Order.model_validate(data)


def test_duplicate_item_ids_are_rejected():
    data = ORDERS[0].model_dump()
    data["items"][1]["item_id"] = data["items"][0]["item_id"]
    with pytest.raises(ValidationError, match="unique"):
        Order.model_validate(data)


def test_first_event_must_be_order_placed():
    data = ORDERS[0].model_dump()
    data["delivery_events"] = data["delivery_events"][1:]
    with pytest.raises(ValidationError, match="order_placed"):
        Order.model_validate(data)


def test_repeated_event_types_are_rejected():
    data = ORDERS[0].model_dump()
    data["delivery_events"][2]["event_type"] = "merchant_accepted"
    with pytest.raises(ValidationError, match="repeated"):
        Order.model_validate(data)


def test_invalid_policy_interval_is_rejected():
    data = POLICIES[0].model_dump()
    data["effective_until"] = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValidationError, match="Policy end"):
        SupportPolicy.model_validate(data)


def test_scenarios_have_unique_ids_and_consistent_references():
    orders = {order.order_id: order for order in ORDERS}
    assert len(SCENARIOS) == 10
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == len(SCENARIOS)
    for scenario in SCENARIOS:
        assert scenario.reported_at.utcoffset() is not None
        order = orders.get(scenario.order_id)
        if scenario.expected_review == "order_not_found":
            assert order is None
            continue
        assert order is not None
        assert scenario.reported_at >= order.delivery_events[0].occurred_at
        if scenario.expected_review == "deny_order_access":
            assert scenario.customer_id != order.customer_id
        else:
            assert scenario.customer_id == order.customer_id
        if scenario.reported_item_id is not None:
            assert scenario.reported_item_id in {item.item_id for item in order.items}


def test_scenarios_cover_delivery_statuses_and_timing():
    orders = {order.order_id: order for order in ORDERS}
    scenarios = {scenario.scenario_id: scenario for scenario in SCENARIOS}
    on_time = orders[scenarios["on-time"].order_id]
    late = orders[scenarios["late"].order_id]
    assert on_time.delivery_events[-1].occurred_at <= on_time.estimated_delivery_at
    assert late.delivery_events[-1].occurred_at > late.estimated_delivery_at
    assert orders[scenarios["in-progress"].order_id].status == "in_progress"
    assert orders[scenarios["cancelled"].order_id].status == "cancelled"


def _is_current(policy, at):
    return policy.effective_from <= at and (policy.effective_until is None or at < policy.effective_until)


@pytest.mark.parametrize("scenario_id", ["missing-item", "wrong-item"])
def test_item_report_scenarios_have_a_current_policy(scenario_id):
    scenario = next(case for case in SCENARIOS if case.scenario_id == scenario_id)
    assert scenario.expected_review == "verify_customer_report"
    assert any(policy.issue_type == scenario.issue_type and _is_current(policy, scenario.reported_at)
               for policy in POLICIES)


def test_expired_policy_is_not_current_for_the_scenario():
    scenario = next(case for case in SCENARIOS if case.scenario_id == "expired-policy")
    policies = [policy for policy in POLICIES if policy.issue_type == scenario.issue_type]
    assert policies
    assert all(policy.effective_until is not None and policy.effective_until <= scenario.reported_at
               for policy in policies)
