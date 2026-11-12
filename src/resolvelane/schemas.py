"""HTTP response models.

Responses use dedicated view models instead of domain objects, so fields added to the
domain later are not exposed unless they are added here deliberately.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from resolvelane.assistant import Issue
from resolvelane.domain.delivery import DeliveryEventType, OrderStatus


class _View(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)


class OrderItemView(_View):
    item_id: str
    name: str
    quantity: int
    unit_price_minor: int


class DeliveryEventView(_View):
    event_type: DeliveryEventType
    occurred_at: datetime


class CustomerOrderView(_View):
    """An order as its owner may see it; internal customer and merchant IDs are omitted."""

    order_id: str
    status: OrderStatus
    currency: str
    items: tuple[OrderItemView, ...]
    delivery_fee_minor: int
    tax_minor: int
    total_minor: int
    estimated_delivery_at: datetime
    delivery_events: tuple[DeliveryEventView, ...]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    order_id: str | None = Field(default=None, min_length=1, max_length=64)


class ChatResponse(BaseModel):
    reply: str
    issues: list[Issue]
    order_id: str | None
    needs_human_review: bool


class ErrorResponse(BaseModel):
    detail: str
