"""HTTP response models.

Responses use dedicated view models instead of domain objects, so fields added to the
domain later are not exposed unless they are added here deliberately.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from resolvelane.domain.case import CaseStatus, Issue, MessageRole
from resolvelane.domain.delivery import DeliveryEventType, OrderStatus
from resolvelane.domain.refund import PaymentStatus, RefundStatus


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


class OrderListView(BaseModel):
    orders: list[CustomerOrderView]


class PrincipalView(BaseModel):
    """Who the bearer token belongs to, so a client can show the right screens."""

    subject_id: str
    role: Literal["customer", "support_agent", "operator"]


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
    payment_status: Literal["processing", "sent", "delayed"] = Field(
        description="sent once the payment provider has made the refund; delayed when it could not be sent "
                    "and a support agent is looking into it")
    sent_at: datetime | None


class CaseView(_View):
    """A case as its customer may see it; pending issues are internal bookkeeping."""

    case_id: str
    order_id: str | None
    status: CaseStatus
    issues: list[Issue]
    created_at: datetime
    messages: list[CaseMessageView]
    refunds: list[CustomerRefundView] = Field(default_factory=list)


class CaseListEntryView(_View):
    case_id: str
    order_id: str | None
    status: CaseStatus
    issues: list[Issue]
    opened_at: datetime


class CaseListView(BaseModel):
    cases: list[CaseListEntryView] = Field(description="The caller's cases, newest first")


class ErrorResponse(BaseModel):
    detail: str


# --- support desk: internal views for support agents --------------------------------

class ToolCallView(_View):
    name: str
    ok: bool
    latency_ms: float
    error: str


class ReplyTraceView(_View):
    """How an assistant reply came about. Internal: never part of a customer view."""

    written_by: Literal["agent", "rules"] = Field(
        description="agent when the language model's reply was sent; rules when the rule-based reply was")
    fallback: str = Field(description="Why the agent's reply was not sent, e.g. ungrounded_number, circuit_open "
                                      "or case_changed; none when it was sent or no agent was asked")
    recognition: str = Field(description="The recognition tier that decided the issues: llm, remote_embedding, "
                                         "local_embedding, keywords or none")
    issues: list[Issue] = Field(description="Issues recognized in the customer's message")
    seconds: float = Field(description="Time to recognize the message and write the reply")
    model: str | None
    rounds: int = Field(description="Language model calls for the reply")
    tools: list[ToolCallView]
    skills: list[str] = Field(description="Handling guides given to the agent, as name@version")
    repeat: bool = Field(description="Nearly the same as the assistant's previous reply")


class SupportMessageView(_View):
    role: MessageRole
    text: str
    sent_at: datetime
    citations: list[CitationView]
    author_id: str | None
    trace: ReplyTraceView | None = Field(description="For assistant replies recorded with one")


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


class PaymentView(_View):
    payment_id: str = Field(description="Also the idempotency key sent to the payment provider")
    amount_minor: int
    currency: str
    status: PaymentStatus
    attempts: int
    next_attempt_at: datetime
    provider_refund_id: str | None
    last_error: str
    created_at: datetime
    completed_at: datetime | None


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
    payment: PaymentView | None


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
    order: CustomerOrderView | None = Field(description="The order record the case is about, once it is known")
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


# --- evaluation reports: operators only ---------------------------------------------

class EvalModelsView(_View):
    agent: str
    judge: str | None = Field(description="None when the run had no judge")


class EvalCountsView(_View):
    turns: int = Field(description="Assistant replies evaluated")
    checks: int = Field(description="Code checks made on them")
    judged: int = Field(description="Replies the judge scored")
    judge_failures: int = Field(description="Replies the judge gave no valid answer for")


class EvalSecondsView(_View):
    mean: float
    p95: float


class EvalRunView(_View):
    """One saved evaluation run, without its replies."""

    report_id: str = Field(description="The report's name, or baseline for the saved baseline")
    suite_id: str = Field(description="Hash of the scenarios, judge instructions and models; runs are "
                                      "comparable only with the same suite")
    created_at: datetime
    models: EvalModelsView
    repeat: int = Field(description="How many times each scenario ran")
    commit: str | None = Field(default=None, description="The application's git commit, when recorded")
    metrics: dict[str, float] = Field(description="Pass rates from 0 to 1, by metric name")
    counts: EvalCountsView
    fallbacks: dict[str, int] = Field(description="Rule-based replies, by why the agent's reply was not sent")
    seconds: EvalSecondsView


class EvalReportListView(BaseModel):
    reports: list[EvalRunView] = Field(description="Saved runs, newest first")
    baseline: EvalRunView | None
    tolerance: float = Field(description="How far a pass rate may fall below the baseline before it is "
                                         "a regression")
    unreadable: int = Field(description="Report files that could not be read")


class EvalCheckView(_View):
    name: str
    passed: bool
    detail: str


class EvalVerdictView(_View):
    passed: bool
    reason: str


class EvalJudgementView(_View):
    verdicts: dict[str, EvalVerdictView]
    error: str = Field(description="Why the judge gave no valid answer; verdicts is then empty")


class EvalTurnView(_View):
    """One customer message of a scenario and the assistant's reply, as checked and judged."""

    scenario: str
    run: int
    step: int = Field(description="The step's number in the scenario, from 1")
    message: str
    reply: str
    written_by_agent: bool
    fallback: str
    seconds: float
    checks: list[EvalCheckView]
    judgement: EvalJudgementView | None


class EvalScenarioView(_View):
    id: str
    description: str


class EvalReportView(EvalRunView):
    scenarios: list[EvalScenarioView] = Field(default=[], description="The scenarios' descriptions, when recorded")
    turns: list[EvalTurnView]
