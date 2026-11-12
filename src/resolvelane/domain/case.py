"""Support case: one customer conversation about one order."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum


class Issue(str, Enum):
    LATE_DELIVERY = "late_delivery"
    MISSING_ITEM = "missing_item"
    WRONG_ITEM = "wrong_item"
    DAMAGED_ITEM = "damaged_item"
    CANCELLATION = "cancellation"
    DELIVERY_STATUS = "delivery_status"
    UNKNOWN = "unknown"


def ordered_issues(issues) -> list[Issue]:
    """Deduplicate issues and return them in a stable order."""
    present = set(issues)
    return [issue for issue in Issue if issue in present]


class CaseStatus(str, Enum):
    OPEN = "open"
    NEEDS_REVIEW = "needs_review"


class MessageRole(str, Enum):
    CUSTOMER = "customer"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class CaseMessage:
    role: MessageRole
    text: str
    sent_at: datetime


@dataclass
class Case:
    case_id: str
    customer_id: str
    created_at: datetime
    order_id: str | None = None
    status: CaseStatus = CaseStatus.OPEN
    # Every issue the customer has raised in this case.
    issues: list[Issue] = field(default_factory=list)
    # Issues raised before an accessible order was known; answered once it is.
    pending_issues: list[Issue] = field(default_factory=list)
    messages: list[CaseMessage] = field(default_factory=list)

    def snapshot(self) -> "Case":
        """A copy that later updates to this case do not change."""
        return replace(self, issues=list(self.issues), pending_issues=list(self.pending_issues),
                       messages=list(self.messages))
