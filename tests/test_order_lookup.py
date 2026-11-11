import pytest

from resolvelane.services.order_lookup import OrderAccess, OrderLookupResult, lookup_order
from resolvelane.simulator.fixtures import ORDERS, SCENARIOS
from resolvelane.simulator.order_source import InMemoryOrderSource


@pytest.fixture
def source():
    return InMemoryOrderSource(ORDERS)


def test_owner_can_read_their_order(source):
    result = lookup_order(source, "demo-customer-101", "demo-order-1001")
    assert result.access is OrderAccess.GRANTED
    assert result.order is not None
    assert result.order.order_id == "demo-order-1001"
    assert result.customer_visible_access is OrderAccess.GRANTED


def test_unknown_order_is_not_found(source):
    result = lookup_order(source, "demo-customer-101", "demo-order-9999")
    assert result.access is OrderAccess.NOT_FOUND
    assert result.order is None


def test_another_customers_order_is_denied_without_details(source):
    result = lookup_order(source, "demo-customer-102", "demo-order-1001")
    assert result.access is OrderAccess.DENIED
    assert result.order is None


def test_customer_cannot_distinguish_denied_from_unknown(source):
    denied = lookup_order(source, "demo-customer-102", "demo-order-1001")
    unknown = lookup_order(source, "demo-customer-102", "demo-order-9999")
    assert denied.customer_visible_access == unknown.customer_visible_access == OrderAccess.NOT_FOUND


@pytest.mark.parametrize("customer_id,order_id", [
    ("", "demo-order-1001"),
    ("demo-customer-101", ""),
])
def test_missing_identifiers_are_rejected(source, customer_id, order_id):
    with pytest.raises(ValueError, match="required"):
        lookup_order(source, customer_id, order_id)


def test_identifiers_are_matched_exactly(source):
    assert lookup_order(source, "DEMO-CUSTOMER-101", "demo-order-1001").access is OrderAccess.DENIED
    assert lookup_order(source, "demo-customer-101", " demo-order-1001").access is OrderAccess.NOT_FOUND


def test_result_cannot_carry_an_order_unless_granted():
    with pytest.raises(ValueError, match="only when access is granted"):
        OrderLookupResult(OrderAccess.DENIED, ORDERS[0])
    with pytest.raises(ValueError, match="only when access is granted"):
        OrderLookupResult(OrderAccess.GRANTED)


def test_duplicate_order_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate order ID"):
        InMemoryOrderSource([ORDERS[0], ORDERS[0]])


EXPECTED_ACCESS = {
    "order_not_found": OrderAccess.NOT_FOUND,
    "deny_order_access": OrderAccess.DENIED,
}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.scenario_id)
def test_scenarios_resolve_to_the_expected_access(source, scenario):
    result = lookup_order(source, scenario.customer_id, scenario.order_id)
    assert result.access is EXPECTED_ACCESS.get(scenario.expected_review, OrderAccess.GRANTED)
