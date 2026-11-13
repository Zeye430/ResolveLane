"""HTTP response models.

Responses use dedicated view models instead of domain objects, so fields added to the
domain later are not exposed unless they are added here deliberately.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from resolvelane.domain.case import CaseStatus, Issue, MessageRole
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


class CitationView(_View):
    """The help center article version and section a reply relied on."""

    article_id: str
    version: int
    title: str
    section: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    case_id: str | None = Field(default=None, min_length=1, max_length=64,
                                description="Continue an existing case; omit to start a new one")


class ChatResponse(BaseModel):
    case_id: str
    case_status: CaseStatus
    reply: str
    issues: list[Issue] = Field(description="Issues answered by this reply")
    order_id: str | None = Field(description="The order this case is about, once it has been found")
    needs_human_review: bool
    citations: list[CitationView] = Field(default_factory=list,
                                          description="Help center sections the reply relies on")


class CaseMessageView(_View):
    role: MessageRole
    text: str
    sent_at: datetime
    citations: list[CitationView]


class CaseView(_View):
    """A case as its customer may see it; pending issues are internal bookkeeping."""

    case_id: str
    order_id: str | None
    status: CaseStatus
    issues: list[Issue]
    created_at: datetime
    messages: list[CaseMessageView]


class ErrorResponse(BaseModel):
    detail: str
