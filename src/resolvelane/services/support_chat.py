"""Handle one customer message within a support case."""

import logging
from dataclasses import dataclass, replace

from resolvelane.assistant import AssistantReply, detect_issues, respond
from resolvelane.case_store import InMemoryCaseStore
from resolvelane.domain.case import Case, CaseMessage, CaseStatus, Issue, MessageRole, ordered_issues
from resolvelane.services.order_lookup import OrderSource, lookup_order

logger = logging.getLogger("resolvelane.support")


class CaseNotFound(Exception):
    """The case does not exist or belongs to another customer."""


class OrderMismatch(Exception):
    def __init__(self, case_order_id: str):
        super().__init__(f"This case is about order {case_order_id}. Start a new case for another order.")


@dataclass(frozen=True)
class ChatTurn:
    case: Case
    reply: AssistantReply


def get_case(cases: InMemoryCaseStore, customer_id: str, case_id: str) -> Case:
    with cases.lock:
        case = cases.get(case_id)
        if case is None or case.customer_id != customer_id:
            raise CaseNotFound
        return case.snapshot()


def handle_message(cases: InMemoryCaseStore, orders: OrderSource, customer_id: str, message: str,
                   case_id: str | None = None, order_id: str | None = None) -> ChatTurn:
    with cases.lock:
        if case_id is None:
            case = cases.create(customer_id)
        else:
            case = cases.get(case_id)
            if case is None or case.customer_id != customer_id:
                raise CaseNotFound
        if order_id is not None and case.order_id is not None and order_id != case.order_id:
            raise OrderMismatch(case.order_id)

        # The case remembers its order, so follow-up messages do not need to repeat it.
        effective_order_id = order_id or case.order_id
        order = None
        if effective_order_id is not None:
            lookup = lookup_order(orders, customer_id, effective_order_id)
            order = lookup.order
            logger.info("order_lookup case=%s access=%s subject=%s order_id=%r",
                        case.case_id, lookup.access.value, customer_id, effective_order_id)

        detected = detect_issues(message)
        new_issues = [issue for issue in detected if issue is not Issue.UNKNOWN]
        # Answer what this message raises plus anything still waiting for an order number.
        to_answer = ordered_issues(case.pending_issues + new_issues) or detected
        reply = respond(to_answer, effective_order_id, order)
        if case.status is CaseStatus.NEEDS_REVIEW and not reply.needs_human_review:
            # Follow-ups such as "when will someone get back to me?" are about the open review.
            reply = replace(reply, reply=reply.reply + " Your case is waiting for a support agent to review it.")

        now = cases.clock()
        case.messages.append(CaseMessage(MessageRole.CUSTOMER, message, now))
        case.messages.append(CaseMessage(MessageRole.ASSISTANT, reply.reply, now))
        case.issues = ordered_issues(case.issues + new_issues)
        if order is not None:
            # Bind only an order the customer may access, never one that failed the lookup.
            case.order_id = order.order_id
            case.pending_issues = []
        else:
            case.pending_issues = ordered_issues(case.pending_issues + new_issues)
        if reply.needs_human_review and case.status is not CaseStatus.NEEDS_REVIEW:
            case.status = CaseStatus.NEEDS_REVIEW
            logger.info("case_status case=%s status=%s issues=%s", case.case_id, case.status.value,
                        ",".join(issue.value for issue in case.issues))

        return ChatTurn(case.snapshot(), reply)
