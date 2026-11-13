"""Handle one customer message within a support case.

A message is handled in three phases so that slow model calls never hold the case lock:

1. Under the lock: check the case belongs to the customer and copy what the reply needs.
2. Without the lock: recognize issues, look up the order, decide the reply by rules, and let
   the reply agent (when configured) write it.
3. Under the lock again: re-check the case, then record the messages and apply decisions.
   If another message changed the case meanwhile, the reply is decided again from the
   current case and the rule-based reply is sent, because the draft was based on old facts.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from resolvelane.assistant import AssistantReply, respond
from resolvelane.case_store import InMemoryCaseStore
from resolvelane.domain.case import Case, CaseMessage, CaseStatus, Issue, MessageRole, ordered_issues
from resolvelane.domain.delivery import Order
from resolvelane.recognition import IssueRecognizer, Recognition
from resolvelane.reply_agent import AgentDraft, AgentFailed, ReplyAgent, ReplyRequest
from resolvelane.services.order_lookup import OrderLookupResult, OrderSource, lookup_order

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


@dataclass(frozen=True)
class CaseState:
    """The parts of a case that decide the reply; compared to detect changes made meanwhile."""
    order_id: str | None
    status: CaseStatus
    issues: tuple[Issue, ...]
    pending_issues: tuple[Issue, ...]

    @classmethod
    def of(cls, case: Case) -> "CaseState":
        return cls(case.order_id, case.status, tuple(case.issues), tuple(case.pending_issues))


NEW_CASE = CaseState(None, CaseStatus.OPEN, (), ())


@dataclass(frozen=True)
class ReplyPlan:
    reply: AssistantReply                # decided by rules: the facts and whether review is needed
    order: Order | None                  # set only when the customer may access the order
    new_issues: list[Issue]
    lookup: OrderLookupResult | None
    looked_up_order_id: str | None = None
    redirect: bool = False               # the message names a different order than the case's


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


def _plan(orders: OrderSource, customer_id: str, state: CaseState, recognition: Recognition,
          order_id: str | None) -> ReplyPlan:
    mentioned = recognition.order_id
    if order_id is None and mentioned is not None and state.order_id not in (None, mentioned):
        reply = AssistantReply(f"This case is about order {state.order_id}. "
                               f"Please start a new case for order {mentioned}.", [], False)
        return ReplyPlan(reply, None, [], None, redirect=True)

    # An explicit order ID wins, then the case's order, then one named in the message.
    effective_order_id = order_id or state.order_id or mentioned
    lookup = lookup_order(orders, customer_id, effective_order_id) if effective_order_id else None
    order = lookup.order if lookup else None

    new_issues = [issue for issue in recognition.issues if issue is not Issue.UNKNOWN]
    # Answer what this message raises plus anything still waiting for an order number.
    to_answer = ordered_issues(list(state.pending_issues) + new_issues) or recognition.issues
    in_review = state.status is CaseStatus.NEEDS_REVIEW
    if in_review and to_answer == [Issue.UNKNOWN]:
        # Nothing new was raised: the customer is asking about the open review.
        reply = AssistantReply(f"Your case about order {state.order_id} is waiting for a support agent "
                               "to review it. You'll see their reply in this conversation.", to_answer, False)
    else:
        reply = respond(to_answer, effective_order_id, order)
        if in_review and not reply.needs_human_review:
            reply = replace(reply, reply=reply.reply + " Your case is waiting for a support agent to review it.")
    return ReplyPlan(reply, order, new_issues, lookup, effective_order_id)


def _draft(agent: ReplyAgent | None, plan: ReplyPlan, customer_id: str, message: str,
           history: Sequence[CaseMessage], state: CaseState, now: datetime) -> AgentDraft | None:
    if agent is None or plan.redirect:
        return None
    request = ReplyRequest(customer_id, message, history, plan.order, plan.reply.issues,
                           ordered_issues(list(state.issues) + plan.new_issues), state.status,
                           plan.reply.needs_human_review, plan.reply.reply, now)
    try:
        return agent.draft(request)
    except AgentFailed as error:
        logger.warning("reply_agent_failed reason=%s; using the rule-based reply", error)
    except Exception:
        logger.exception("reply_agent_error; using the rule-based reply")
    return None


def _record(cases: InMemoryCaseStore, case: Case, message: str, reply: str) -> None:
    now = cases.clock()
    case.messages.append(CaseMessage(MessageRole.CUSTOMER, message, now))
    case.messages.append(CaseMessage(MessageRole.ASSISTANT, reply, now))


def handle_message(cases: InMemoryCaseStore, orders: OrderSource, recognizer: IssueRecognizer,
                   customer_id: str, message: str, case_id: str | None = None, order_id: str | None = None,
                   reply_agent: ReplyAgent | None = None) -> ChatTurn:
    # Phase 1: validate the request and copy what the reply needs, then release the lock.
    state, history = NEW_CASE, []
    if case_id is not None:
        with cases.lock:
            case = _owned_case(cases, customer_id, case_id)
            _check_order(case, order_id)
            state, history = CaseState.of(case), list(case.messages)

    # Phase 2: slow work without the lock, so other customers' messages are not blocked.
    recognition = recognizer.recognize(message, history)
    plan = _plan(orders, customer_id, state, recognition, order_id)
    draft = _draft(reply_agent, plan, customer_id, message, history, state, cases.clock())

    # Phase 3: re-check and apply.
    with cases.lock:
        if case_id is None:
            case = cases.create(customer_id)
        else:
            case = _owned_case(cases, customer_id, case_id)
            _check_order(case, order_id)  # the case may have been bound meanwhile
        current = CaseState.of(case)
        if current != state:
            logger.info("case_changed_during_reply case=%s; deciding the reply again", case.case_id)
            plan, draft = _plan(orders, customer_id, current, recognition, order_id), None

        logger.info("issue_recognition case=%s source=%s issues=%s", case.case_id, recognition.source,
                    ",".join(issue.value for issue in recognition.issues))
        if plan.lookup is not None:
            logger.info("order_lookup case=%s access=%s subject=%s order_id=%r", case.case_id,
                        plan.lookup.access.value, customer_id, plan.looked_up_order_id)
        if plan.redirect:
            _record(cases, case, message, plan.reply.reply)
            return ChatTurn(case.snapshot(), plan.reply)

        review_request = draft.review_request if draft else None
        reply = AssistantReply(draft.reply if draft else plan.reply.reply, plan.reply.issues,
                               plan.reply.needs_human_review or review_request is not None)
        if draft:
            logger.info("reply case=%s source=agent rounds=%d tools=%s", case.case_id, draft.rounds,
                        ",".join(f"{trace.name}:{'ok' if trace.ok else trace.error}" for trace in draft.tool_traces)
                        or "-")
        else:
            logger.info("reply case=%s source=rules", case.case_id)

        _record(cases, case, message, reply.reply)
        case.issues = ordered_issues(case.issues + plan.new_issues)
        if plan.order is not None:
            # Bind only an order the customer may access, never one that failed the lookup.
            case.order_id = plan.order.order_id
            case.pending_issues = []
        else:
            case.pending_issues = ordered_issues(case.pending_issues + plan.new_issues)
        if review_request is not None:
            case.review_notes.append(review_request.reason)
        if reply.needs_human_review and case.status is not CaseStatus.NEEDS_REVIEW:
            case.status = CaseStatus.NEEDS_REVIEW
            logger.info("case_status case=%s status=%s issues=%s requested_by=%s", case.case_id,
                        case.status.value, ",".join(issue.value for issue in case.issues),
                        "rules" if plan.reply.needs_human_review else "agent")

        return ChatTurn(case.snapshot(), reply)
