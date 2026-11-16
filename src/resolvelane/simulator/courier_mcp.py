"""A simulated courier tracking service, reached over MCP (stdio).

It stands in for a logistics partner's system: separate from the marketplace's order records,
with its own view of where a delivery is. All data is invented. Run it with
``python -m resolvelane.simulator.courier_mcp``; the application starts it itself when it is
listed in the MCP configuration (see mcp_servers.example.json).

``reassign_courier`` changes data and is marked so; it exists to show that such a tool is never
offered to the reply agent, even when an operator lists it.
"""

from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel

server = MCPServer("Simulated courier tracking", version="0.1.0")


class CourierStatus(BaseModel):
    order_id: str
    stage: Literal["heading_to_restaurant", "waiting_at_restaurant", "on_the_way", "delivered", "not_started"]
    minutes_away: int | None = None
    updated_at: str


# Invented tracking states, keyed by the simulated marketplace's order IDs.
_TRACKING = {
    "demo-order-1001": CourierStatus(order_id="demo-order-1001", stage="delivered", updated_at="2026-01-15T18:55:00Z"),
    "demo-order-1002": CourierStatus(order_id="demo-order-1002", stage="on_the_way", minutes_away=12,
                                     updated_at="2026-01-15T19:08:00Z"),
    "demo-order-1003": CourierStatus(order_id="demo-order-1003", stage="delivered", updated_at="2026-01-16T12:40:00Z"),
}


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def get_courier_status(order_id: str) -> CourierStatus:
    """The courier's latest tracked stage for an order and, while on the way, the minutes to arrival."""
    status = _TRACKING.get(order_id)
    if status is None:
        raise ToolError("No delivery is tracked for this order")
    return status


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True))
def reassign_courier(order_id: str) -> str:
    """Assign a different courier to an order."""
    return f"Courier reassigned for {order_id}"


if __name__ == "__main__":
    server.run(transport="stdio")
