"""Handle one customer message within a support case."""

import logging
from dataclasses import dataclass, replace

from resolvelane.assistant import AssistantReply, respond
from resolvelane.case_store import InMemoryCaseStore
from resolvelane.domain.case import Case, CaseMessage, CaseStatus, Issue, MessageRole, ordered_issues
from resolvelane.recognition import IssueRecognizer
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


def _owned_case(cases: InMemoryCaseStore, customer_id: str, case_id: str) -> Case:
    case = cases.get(case_id)
    if case is None or case.customer_id != customer_id:
        raise CaseNotFound
    return case


def get_case(cases: InMemoryCaseStore, customer_id: str, case_id: str) -> Case:
    with cases.lock:
        return _owned_case(cases, customer_id, case_id).snapshot()


def _check_order(case: Case, order_id: str | None) -> None:
    if order_id is not None and case.order_id is not None and order_id != case.order_id:
        raise OrderMismatch(case.order_id)


def _record(cases: InMemoryCaseStore, case: Case, message: str, reply: AssistantReply) -> None:
    now = cases.clock()
    case.messages.append(CaseMessage(MessageRole.CUSTOMER, message, now))
    case.messages.append(CaseMessage(MessageRole.ASSISTANT, reply.reply, now))


def handle_message(cases: InMemoryCaseStore, orders: OrderSource, recognizer: IssueRecognizer,
                   customer_id: str, message: str,
                   case_id: str | None = None, order_id: str | None = None) -> ChatTurn:
    # Validate the request and copy the history, then release the lock: the model call
    # can take seconds and must not block other customers' messages.
    history: list[CaseMessage] = []
    if case_id is not None:
        with cases.lock:
            case = _owned_case(cases, customer_id, case_id)
            _check_order(case, order_id)
            history = list(case.messages)

    recognition = recognizer.recognize(message, history)

    with cases.lock:
        if case_id is None:
            case = cases.create(customer_id)
        else:
            case = _owned_case(cases, customer_id, case_id)
            _check_order(case, order_id)  # the case may have been bound meanwhile
        logger.info("issue_recognition case=%s source=%s issues=%s", case.case_id, recognition.source,
                    ",".join(issue.value for issue in recognition.issues))

        mentioned = recognition.order_id
        if order_id is None and mentioned is not None and case.order_id not in (None, mentioned):
            reply = AssistantReply(f"This case is about order {case.order_id}. "
                                   f"Please start a new case for order {mentioned}.", [], False)
            _record(cases, case, message, reply)
            return ChatTurn(case.snapshot(), reply)

        # An explicit order ID wins, then the case's order, then one named in the message.
        effective_order_id = order_id or case.order_id or mentioned
        order = None
        if effective_order_id is not None:
            lookup = lookup_order(orders, customer_id, effective_order_id)
            order = lookup.order
            logger.info("order_lookup case=%s access=%s subject=%s order_id=%r",
                        case.case_id, lookup.access.value, customer_id, effective_order_id)

        new_issues = [issue for issue in recognition.issues if issue is not Issue.UNKNOWN]
        # Answer what this message raises plus anything still waiting for an order number.
        to_answer = ordered_issues(case.pending_issues + new_issues) or recognition.issues
        if case.status is CaseStatus.NEEDS_REVIEW and to_answer == [Issue.UNKNOWN]:
            # Nothing new was raised: the customer is asking about the open review.
            reply = AssistantReply(f"Your case about order {case.order_id} is waiting for a support agent "
                                   "to review it. You'll see their reply in this conversation.", to_answer, False)
        else:
            reply = respond(to_answer, effective_order_id, order)
        if case.status is CaseStatus.NEEDS_REVIEW and to_answer != [Issue.UNKNOWN] and not reply.needs_human_review:
            # Follow-ups such as "when will someone get back to me?" are about the open review.
            reply = replace(reply, reply=reply.reply + " Your case is waiting for a support agent to review it.")

        _record(cases, case, message, reply)
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
