"""HTTP response models.

Responses use dedicated view models instead of domain objects, so fields added to the
domain later are not exposed unless they are added here deliberately.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from resolvelane.domain.case import CaseStatus, Issue, MessageRole
from resolvelane.domain.delivery import DeliveryEventType, OrderStatus
from resolvelane.domain.refund import RefundStatus


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


class RefundLineView(_View):
    item_id: str
    name: str
    quantity: int
    amount_minor: int


class CustomerRefundView(BaseModel):
    """An approved refund. Proposals and rejections are internal: the support agent tells the customer."""

    refund_id: str
    status: RefundStatus
    currency: str
    amount_minor: int
    lines: list[RefundLineView]
    delivery_fee_minor: int
    approved_at: datetime


class CaseView(_View):
    """A case as its customer may see it; pending issues are internal bookkeeping."""

    case_id: str
    order_id: str | None
    status: CaseStatus
    issues: list[Issue]
    created_at: datetime
    messages: list[CaseMessageView]
    refunds: list[CustomerRefundView] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str


# --- support desk: internal views for support agents --------------------------------

class SupportMessageView(_View):
    role: MessageRole
    text: str
    sent_at: datetime
    citations: list[CitationView]
    author_id: str | None


class TransitionView(_View):
    at: datetime
    from_status: CaseStatus
    to_status: CaseStatus
    actor_role: str
    actor_id: str | None
    note: str


class RefundVersionView(_View):
    version: int
    lines: list[RefundLineView]
    delivery_fee_minor: int
    amount_minor: int
    reason: str
    author_role: str
    author_id: str | None
    created_at: datetime


class RefundView(_View):
    refund_id: str
    order_id: str
    currency: str
    status: RefundStatus
    created_at: datetime
    versions: list[RefundVersionView] = Field(description="Every version proposed; the last one is current")
    decided_version: int | None
    decided_at: datetime | None
    decided_by: str | None
    decision_note: str


class SupportCaseView(_View):
    """Everything support agents need to handle a case, including internal notes."""

    case_id: str
    customer_id: str
    order_id: str | None
    status: CaseStatus
    issues: list[Issue]
    pending_issues: list[Issue]
    review_notes: list[str]
    created_at: datetime
    waiting_since: datetime
    summary: str | None = Field(description="Model-written summary of older messages; derived, not a record")
    messages: list[SupportMessageView]
    transitions: list[TransitionView]
    refunds: list[RefundView]


class QueueEntryView(_View):
    case_id: str
    customer_id: str
    order_id: str | None
    status: CaseStatus
    issues: list[Issue]
    review_notes: list[str]
    waiting_since: datetime
    message_count: int


class QueuePageView(BaseModel):
    cases: list[QueueEntryView]
    next_cursor: str | None = Field(description="Pass as cursor to read the next page; null on the last page")


class SupportReplyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ResolveRequest(BaseModel):
    note: str = Field(min_length=1, max_length=500, description="Internal note on how the case was resolved")


class SupportActionResponse(BaseModel):
    case_id: str
    status: CaseStatus
    message_count: int


class RefundItemRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=64)
    quantity: int = Field(ge=1, le=99)


class RefundProposalRequest(BaseModel):
    items: list[RefundItemRequest] = Field(max_length=50, description="Order items to refund, each once")
    include_delivery_fee: bool = False
    reason: str = Field(min_length=1, max_length=500, description="Internal: why this refund")


class RefundRevisionRequest(RefundProposalRequest):
    revises: int = Field(ge=1, description="The version this revision replaces; it must be the current one")


class RefundDecisionRequest(BaseModel):
    version: int = Field(ge=1, description="The version reviewed; it must be the current one")
    note: str = Field(default="", max_length=500, description="Internal note on the decision")


class RefundRejectionRequest(RefundDecisionRequest):
    note: str = Field(min_length=1, max_length=500, description="Internal note on why the refund was rejected")


class RefundActionResponse(BaseModel):
    refund_id: str
    case_id: str
    status: RefundStatus
    version: int
    amount_minor: int
    currency: str
