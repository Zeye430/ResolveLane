"""Read-only order lookup with an explicit customer ownership check.

The requester identifier is supplied by the caller. In the simulator it is a
fictional customer ID, not an authenticated identity; production use requires
the API layer to derive it from a trusted authentication mechanism.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from resolvelane.domain.delivery import Order


class OrderSource(Protocol):
    """Adapter boundary for order data, such as the simulator or a future database."""

    def get(self, order_id: str) -> Order | None: ...

    def for_customer(self, customer_id: str) -> list[Order]:
        """The customer's orders, newest estimated delivery first."""
        ...


class OrderAccess(str, Enum):
    GRANTED = "granted"
    NOT_FOUND = "not_found"
    DENIED = "denied"


@dataclass(frozen=True)
class OrderLookupResult:
    """Internal lookup outcome.

    ``access`` distinguishes a missing order from another customer's order so that
    audit records and evaluations can tell them apart. Customers must not see that
    difference, so callers responding to a customer use ``customer_visible_access``.
    """

    access: OrderAccess
    order: Order | None = None

    def __post_init__(self):
        if (self.access is OrderAccess.GRANTED) != (self.order is not None):
            raise ValueError("An order is returned only when access is granted")

    @property
    def customer_visible_access(self) -> OrderAccess:
        if self.access is OrderAccess.DENIED:
            return OrderAccess.NOT_FOUND
        return self.access


def lookup_order(source: OrderSource, requester_customer_id: str, order_id: str) -> OrderLookupResult:
    if not requester_customer_id or not order_id:
        raise ValueError("Requester customer ID and order ID are required")

    order = source.get(order_id)
    if order is None:
        return OrderLookupResult(OrderAccess.NOT_FOUND)
    if order.customer_id != requester_customer_id:
        return OrderLookupResult(OrderAccess.DENIED)
    return OrderLookupResult(OrderAccess.GRANTED, order)
