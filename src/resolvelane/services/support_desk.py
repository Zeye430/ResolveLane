"""The support agents' side of a case: the review queue, replies to customers, resolution.

Support agents see every case, including internal details customers do not see: review
notes, the conversation summary, who wrote each message and every status change. Their
actions take effect under the case lock and are recorded as status transitions with the
agent's identity.

Replies and resolutions accept an idempotency key, like customer messages, so an agent's
retried request is applied once.
"""

import base64
import binascii
import hashlib
import json
import logging
from dataclasses import dataclass

from resolvelane.case_store import CaseStore, QueueEntry, Receipt
from resolvelane.domain.case import Case, CaseMessage, CaseStatus, CaseSummary, MessageRole

logger = logging.getLogger("resolvelane.desk")

MAX_PAGE = 50


class CaseNotFound(Exception):
    pass


class CaseResolved(Exception):
    """The case is resolved; a reply would reach a closed conversation."""


class InvalidCursor(Exception):
    pass


class IdempotencyKeyReused(Exception):
    pass


@dataclass(frozen=True)
class QueuePage:
    entries: list[QueueEntry]
    next_cursor: str | None


@dataclass(frozen=True)
class DeskResult:
    case_id: str
    status: CaseStatus
    message_count: int
    replayed: bool = False

    def to_json(self) -> str:
        return json.dumps({"case_id": self.case_id, "status": self.status.value, "message_count": self.message_count})

    @classmethod
    def from_json(cls, text: str) -> "DeskResult":
        data = json.loads(text)
        return cls(data["case_id"], CaseStatus(data["status"]), data["message_count"], replayed=True)


def _cursor(entry: QueueEntry) -> str:
    return base64.urlsafe_b64encode(json.dumps([entry.position, entry.case_id]).encode()).decode()


def _parse_cursor(cursor: str) -> tuple[str, str]:
    try:
        position, case_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except (ValueError, TypeError, binascii.Error):
        raise InvalidCursor from None
    if not isinstance(position, str) or not isinstance(case_id, str):
        raise InvalidCursor
    return position, case_id


def queue(cases: CaseStore, status: CaseStatus, limit: int = 20, cursor: str | None = None) -> QueuePage:
    """Cases in ``status``, longest waiting first, a page at a time."""
    limit = max(1, min(limit, MAX_PAGE))
    entries = cases.queue(status, limit + 1, _parse_cursor(cursor) if cursor else None)
    more = len(entries) > limit
    entries = entries[:limit]
    return QueuePage(entries, _cursor(entries[-1]) if more else None)


def read_case(cases: CaseStore, case_id: str) -> tuple[Case, CaseSummary | None]:
    with cases.lock:
        case = cases.get(case_id)
        if case is None:
            raise CaseNotFound
        return case, cases.summary(case_id)


def _act(cases: CaseStore, agent_id: str, case_id: str, operation: str, payload: dict, key: str | None, change):
    fingerprint = hashlib.sha256(json.dumps({"operation": operation, "case_id": case_id, **payload},
                                            sort_keys=True).encode()).hexdigest()
    with cases.lock:
        if key is not None and (receipt := cases.receipt(agent_id, key)) is not None:
            if receipt.fingerprint != fingerprint:
                raise IdempotencyKeyReused
            return DeskResult.from_json(receipt.response)
        case = cases.get(case_id)
        if case is None:
            raise CaseNotFound
        change(case, cases.clock())
        result = DeskResult(case.case_id, case.status, len(case.messages))
        cases.commit(case, Receipt(agent_id, key, fingerprint, result.to_json(), cases.clock()) if key else None)
        return result


def reply(cases: CaseStore, agent_id: str, case_id: str, text: str, key: str | None = None) -> DeskResult:
    """Send a message to the customer as a support agent. The case stays in its queue."""
    def change(case: Case, now) -> None:
        if case.status is CaseStatus.RESOLVED:
            raise CaseResolved
        case.messages.append(CaseMessage(MessageRole.SUPPORT_AGENT, text, now, author_id=agent_id))
        logger.info("support_reply case=%s agent=%s", case.case_id, agent_id)

    return _act(cases, agent_id, case_id, "reply", {"text": text}, key, change)


def resolve(cases: CaseStore, agent_id: str, case_id: str, note: str, key: str | None = None) -> DeskResult:
    """Close the case. The note is internal; a later customer message reopens the case."""
    def change(case: Case, now) -> None:
        if case.status is CaseStatus.RESOLVED:
            raise CaseResolved
        case.move_to(CaseStatus.RESOLVED, now, "support_agent", agent_id, note)
        logger.info("case_status case=%s status=resolved agent=%s", case.case_id, agent_id)

    return _act(cases, agent_id, case_id, "resolve", {"note": note}, key, change)
