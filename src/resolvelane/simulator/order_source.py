"""In-memory order source backed by simulated fixtures."""

from collections.abc import Iterable

from resolvelane.domain.delivery import Order


class InMemoryOrderSource:
    def __init__(self, orders: Iterable[Order]):
        self._orders: dict[str, Order] = {}
        for order in orders:
            if order.order_id in self._orders:
                raise ValueError(f"Duplicate order ID: {order.order_id}")
            self._orders[order.order_id] = order

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)
