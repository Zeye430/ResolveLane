"""The support agents' side of a case: the review queue, replies to customers, resolution.

Support agents see every case, including internal details customers do not see: review
notes, the conversation summary, who wrote each message and every status change. Their
actions take effect under the case lock and are recorded as status transitions with the
agent's identity.

Replies and resolutions accept an idempotency key, like customer messages, so an agent's
retried request is applied once.

Refunds: an agent proposes one, revises a proposal (a new version; earlier versions are
kept), and approves or rejects it. A decision names the version the agent reviewed and is
refused unless that version is still the current one, so no one approves an amount they
have not seen. Approval prices the version again against the order's other approved
refunds, then tells the customer. A case with an undecided refund cannot be resolved.
"""

import base64
import binascii
import hashlib
import json
import logging
from dataclasses import dataclass

from resolvelane.case_store import CaseStore, QueueEntry, Receipt
from resolvelane.domain.case import Case, CaseMessage, CaseStatus, CaseSummary, MessageRole
from resolvelane.domain.refund import Refund, RefundStatus
from resolvelane.refunds import (RefundInvalid, RefundProposal, Refunded, describe, money, new_refund, new_version,
                                 open_refund, price_refund)
from resolvelane.services.order_lookup import OrderSource

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


class RefundNotFound(Exception):
    pass


class RefundConflict(Exception):
    """The refund's current state does not allow the action; the message says why."""


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


@dataclass(frozen=True)
class RefundResult:
    refund_id: str
    case_id: str
    status: RefundStatus
    version: int
    amount_minor: int
    currency: str
    replayed: bool = False

    @classmethod
    def of(cls, refund: Refund) -> "RefundResult":
        version = refund.decided or refund.current
        return cls(refund.refund_id, refund.case_id, refund.status, version.version, version.amount_minor,
                   refund.currency)

    def to_json(self) -> str:
        return json.dumps({"refund_id": self.refund_id, "case_id": self.case_id, "status": self.status.value,
                           "version": self.version, "amount_minor": self.amount_minor, "currency": self.currency})

    @classmethod
    def from_json(cls, text: str) -> "RefundResult":
        data = json.loads(text)
        return cls(data["refund_id"], data["case_id"], RefundStatus(data["status"]), data["version"],
                   data["amount_minor"], data["currency"], replayed=True)


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


def _act(cases: CaseStore, agent_id: str, case_id: str, operation: str, payload: dict, key: str | None, change,
         result_type=DeskResult):
    """Apply ``change(case, now)``, which returns the result, and commit it with the request's receipt."""
    fingerprint = hashlib.sha256(json.dumps({"operation": operation, "case_id": case_id, **payload},
                                            sort_keys=True).encode()).hexdigest()
    with cases.lock:
        if key is not None and (receipt := cases.receipt(agent_id, key)) is not None:
            if receipt.fingerprint != fingerprint:
                raise IdempotencyKeyReused
            return result_type.from_json(receipt.response)
        case = cases.get(case_id)
        if case is None:
            raise CaseNotFound
        result = change(case, cases.clock())
        cases.commit(case, Receipt(agent_id, key, fingerprint, result.to_json(), cases.clock()) if key else None)
        return result


def _counts(case: Case) -> DeskResult:
    return DeskResult(case.case_id, case.status, len(case.messages))


def reply(cases: CaseStore, agent_id: str, case_id: str, text: str, key: str | None = None) -> DeskResult:
    """Send a message to the customer as a support agent. The case stays in its queue."""
    def change(case: Case, now) -> DeskResult:
        if case.status is CaseStatus.RESOLVED:
            raise CaseResolved
        case.messages.append(CaseMessage(MessageRole.SUPPORT_AGENT, text, now, author_id=agent_id))
        logger.info("support_reply case=%s agent=%s", case.case_id, agent_id)
        return _counts(case)

    return _act(cases, agent_id, case_id, "reply", {"text": text}, key, change)


def resolve(cases: CaseStore, agent_id: str, case_id: str, note: str, key: str | None = None) -> DeskResult:
    """Close the case. The note is internal; a later customer message reopens the case."""
    def change(case: Case, now) -> DeskResult:
        if case.status is CaseStatus.RESOLVED:
            raise CaseResolved
        if open_refund(case.refunds) is not None:
            raise RefundConflict("Approve or reject the refund proposal before resolving the case")
        case.move_to(CaseStatus.RESOLVED, now, "support_agent", agent_id, note)
        logger.info("case_status case=%s status=resolved agent=%s", case.case_id, agent_id)
        return _counts(case)

    return _act(cases, agent_id, case_id, "resolve", {"note": note}, key, change)


# --- refunds ----------------------------------------------------------------------------

def _proposal_payload(proposal: RefundProposal) -> dict:
    return {"items": [list(item) for item in proposal.items], "include_delivery_fee": proposal.include_delivery_fee,
            "reason": proposal.reason}


def _order_of(case: Case, orders: OrderSource):
    order = orders.get(case.order_id) if case.order_id else None
    if order is None:
        raise RefundConflict("This case has no order to refund")
    return order


def _priced(cases: CaseStore, case: Case, orders: OrderSource, proposal: RefundProposal, number: int,
            agent_id: str, now):
    order = _order_of(case, orders)
    version = new_version(order, proposal, Refunded.of(cases.approved_refunds(order.order_id)), number,
                          "support_agent", agent_id, now)
    return order, version


def propose_refund(cases: CaseStore, orders: OrderSource, agent_id: str, case_id: str, proposal: RefundProposal,
                   key: str | None = None) -> RefundResult:
    """Propose a refund on a case without one waiting for a decision."""
    def change(case: Case, now) -> RefundResult:
        if case.status is CaseStatus.RESOLVED:
            raise CaseResolved
        if open_refund(case.refunds) is not None:
            raise RefundConflict("This case already has a refund proposal; revise it instead")
        order, version = _priced(cases, case, orders, proposal, 1, agent_id, now)
        refund = new_refund(case.case_id, order, version, now)
        case.refunds.append(refund)
        logger.info("refund_proposal case=%s refund=%s version=1 amount_minor=%d author=%s", case.case_id,
                    refund.refund_id, version.amount_minor, agent_id)
        return RefundResult.of(refund)

    return _act(cases, agent_id, case_id, "propose_refund", _proposal_payload(proposal), key, change, RefundResult)


def _refund_action(cases: CaseStore, agent_id: str, refund_id: str, operation: str, payload: dict,
                   key: str | None, change) -> RefundResult:
    case_id = cases.refund_case(refund_id)
    if case_id is None:
        raise RefundNotFound

    def on_case(case: Case, now) -> RefundResult:
        refund = next(item for item in case.refunds if item.refund_id == refund_id)
        if refund.status is not RefundStatus.PROPOSED:
            raise RefundConflict(f"This refund was already {refund.status.value}")
        change(case, refund, now)
        return RefundResult.of(refund)

    return _act(cases, agent_id, case_id, operation, {"refund_id": refund_id, **payload}, key, on_case, RefundResult)


def _check_current(refund: Refund, version: int) -> None:
    if version != len(refund.versions):
        raise RefundConflict(f"Version {version} is not the current version of this refund; "
                             f"review version {len(refund.versions)}")


def revise_refund(cases: CaseStore, orders: OrderSource, agent_id: str, refund_id: str, revises: int,
                  proposal: RefundProposal, key: str | None = None) -> RefundResult:
    """Replace the proposal with a new version; ``revises`` is the version the agent started from."""
    def change(case: Case, refund: Refund, now) -> None:
        _check_current(refund, revises)
        _, version = _priced(cases, case, orders, proposal, revises + 1, agent_id, now)
        refund.revise(version)
        logger.info("refund_proposal case=%s refund=%s version=%d amount_minor=%d author=%s", case.case_id,
                    refund_id, version.version, version.amount_minor, agent_id)

    return _refund_action(cases, agent_id, refund_id, "revise_refund",
                          {"revises": revises, **_proposal_payload(proposal)}, key, change)


def approve_refund(cases: CaseStore, orders: OrderSource, agent_id: str, refund_id: str, version: int,
                   note: str = "", key: str | None = None) -> RefundResult:
    """Approve the version the agent reviewed, which must be the current one, and tell the customer."""
    def change(case: Case, refund: Refund, now) -> None:
        _check_current(refund, version)
        approved = refund.current
        # Another refund of the order may have been approved since this version was written.
        again = RefundProposal(tuple((line.item_id, line.quantity) for line in approved.lines),
                               approved.delivery_fee_minor > 0, approved.reason)
        try:
            price_refund(_order_of(case, orders), again, Refunded.of(cases.approved_refunds(refund.order_id)))
        except RefundInvalid as error:
            raise RefundConflict(f"{error}. Revise the proposal") from None
        refund.decide(RefundStatus.APPROVED, version, now, agent_id, note)
        case.messages.append(CaseMessage(
            MessageRole.SUPPORT_AGENT,
            f"I've approved a refund of {money(approved.amount_minor, refund.currency)} for {describe(approved)}. "
            "It goes back to your original payment method.", now, author_id=agent_id))
        logger.info("refund_decision case=%s refund=%s version=%d status=approved amount_minor=%d agent=%s",
                    case.case_id, refund_id, version, approved.amount_minor, agent_id)

    return _refund_action(cases, agent_id, refund_id, "approve_refund", {"version": version, "note": note}, key,
                          change)


def reject_refund(cases: CaseStore, agent_id: str, refund_id: str, version: int, note: str,
                  key: str | None = None) -> RefundResult:
    """Reject the current version. The note is internal; the agent tells the customer in a reply."""
    def change(case: Case, refund: Refund, now) -> None:
        _check_current(refund, version)
        refund.decide(RefundStatus.REJECTED, version, now, agent_id, note)
        logger.info("refund_decision case=%s refund=%s version=%d status=rejected agent=%s", case.case_id,
                    refund_id, version, agent_id)

    return _refund_action(cases, agent_id, refund_id, "reject_refund", {"version": version, "note": note}, key,
                          change)
