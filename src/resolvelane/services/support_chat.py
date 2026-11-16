"""Handle one customer message within a support case.

A message is handled in three phases so that slow model calls never hold the case lock:

1. Under the lock: check the case belongs to the customer and copy what the reply needs.
2. Without the lock: recognize issues, look up the order, decide the reply by rules, and let
   the reply agent (when configured) write it.
3. Under the lock again: re-check the case, then record the messages and apply decisions.
   If another message changed the case meanwhile, the reply is decided again from the
   current case and the rule-based reply is sent, because the draft was based on old facts.

A message sent with an idempotency key leaves a receipt, committed together with the case.
A retry with the same key and body gets the stored response instead of being handled again,
so a client that lost a response can safely send the message again. The receipt is checked
in phase 1 and again in phase 3, in case a duplicate of the request finished meanwhile.

Phase 1 also gathers the agent's memory (see resolvelane.memory): the case summary, the
messages it does not cover, and the customer's other recent cases. Once a conversation is
long enough, phase 3 queues a new summary, which is written in the background.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from resolvelane.assistant import AssistantReply, respond
from resolvelane.case_store import CaseStore, Receipt
from resolvelane.domain.case import (Case, CaseMessage, CaseStatus, CaseSummary, EarlierCase, Issue, MessageRole,
                                     ordered_issues)
from resolvelane.domain.delivery import Order
from resolvelane.domain.knowledge import Citation
from resolvelane.memory import SummaryQueue, earlier_cases, needs_summary, recent_messages
from resolvelane.recognition import IssueRecognizer, Recognition
from resolvelane.reply_agent import AgentDraft, AgentFailed, ReplyAgent, ReplyRequest
from resolvelane.services.order_lookup import OrderLookupResult, OrderSource, lookup_order

logger = logging.getLogger("resolvelane.support")


class CaseNotFound(Exception):
    """The case does not exist or belongs to another customer."""


class OrderMismatch(Exception):
    def __init__(self, case_order_id: str):
        super().__init__(f"This case is about order {case_order_id}. Start a new case for another order.")


class IdempotencyKeyReused(Exception):
    """The idempotency key was already used for a request with a different body."""


@dataclass(frozen=True)
class ChatTurn:
    """The response to one message, as it was when the message was handled."""
    case_id: str
    case_status: CaseStatus
    order_id: str | None
    reply: AssistantReply
    replayed: bool = False  # answered from the receipt of an earlier identical request

    @classmethod
    def of(cls, case: Case, reply: AssistantReply) -> "ChatTurn":
        return cls(case.case_id, case.status, case.order_id, reply)

    def to_json(self) -> str:
        return json.dumps({
            "case_id": self.case_id, "case_status": self.case_status.value, "order_id": self.order_id,
            "reply": self.reply.reply, "issues": [issue.value for issue in self.reply.issues],
            "needs_human_review": self.reply.needs_human_review,
            "citations": [vars(citation) for citation in self.reply.citations]})

    @classmethod
    def from_json(cls, text: str) -> "ChatTurn":
        data = json.loads(text)
        reply = AssistantReply(data["reply"], [Issue(value) for value in data["issues"]], data["needs_human_review"],
                               tuple(Citation(**citation) for citation in data["citations"]))
        return cls(data["case_id"], CaseStatus(data["case_status"]), data["order_id"], reply)


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


def _owned_case(cases: CaseStore, customer_id: str, case_id: str) -> Case:
    case = cases.get(case_id)
    if case is None or case.customer_id != customer_id:
        raise CaseNotFound
    return case


def get_case(cases: CaseStore, customer_id: str, case_id: str) -> Case:
    return _owned_case(cases, customer_id, case_id)


def _fingerprint(message: str, case_id: str | None, order_id: str | None) -> str:
    body = json.dumps({"message": message, "case_id": case_id, "order_id": order_id}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def _replay(cases: CaseStore, customer_id: str, key: str | None, fingerprint: str | None) -> ChatTurn | None:
    """The stored response to an earlier request with this key, if there is one."""
    if key is None:
        return None
    receipt = cases.receipt(customer_id, key)
    if receipt is None:
        return None
    if receipt.fingerprint != fingerprint:
        raise IdempotencyKeyReused
    turn = replace(ChatTurn.from_json(receipt.response), replayed=True)
    logger.info("idempotent_replay case=%s subject=%s", turn.case_id, customer_id)
    return turn


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
           history: Sequence[CaseMessage], summary: CaseSummary | None, earlier: Sequence[EarlierCase],
           state: CaseState, now: datetime) -> AgentDraft | None:
    if agent is None or plan.redirect:
        return None
    request = ReplyRequest(customer_id, message, recent_messages(history, summary), plan.order, plan.reply.issues,
                           ordered_issues(list(state.issues) + plan.new_issues), state.status,
                           plan.reply.needs_human_review, plan.reply.reply, now,
                           summary.text if summary else None, earlier)
    try:
        return agent.draft(request)
    except AgentFailed as error:
        logger.warning("reply_agent_failed reason=%s; using the rule-based reply", error)
    except Exception:
        logger.exception("reply_agent_error; using the rule-based reply")
    return None


def _record(cases: CaseStore, case: Case, message: str, reply: AssistantReply) -> None:
    now = cases.clock()
    case.messages.append(CaseMessage(MessageRole.CUSTOMER, message, now))
    case.messages.append(CaseMessage(MessageRole.ASSISTANT, reply.reply, now, reply.citations))


def _commit(cases: CaseStore, case: Case, reply: AssistantReply, customer_id: str, key: str | None,
            fingerprint: str | None) -> ChatTurn:
    turn = ChatTurn.of(case, reply)
    receipt = (Receipt(customer_id, key, fingerprint, turn.to_json(), cases.clock())
               if key is not None and fingerprint is not None else None)
    cases.commit(case, receipt)
    return turn


def handle_message(cases: CaseStore, orders: OrderSource, recognizer: IssueRecognizer,
                   customer_id: str, message: str, case_id: str | None = None, order_id: str | None = None,
                   reply_agent: ReplyAgent | None = None, idempotency_key: str | None = None,
                   summaries: SummaryQueue | None = None) -> ChatTurn:
    fingerprint = _fingerprint(message, case_id, order_id) if idempotency_key is not None else None

    # Phase 1: validate the request and copy what the reply needs, then release the lock.
    state, history, summary = NEW_CASE, [], None
    with cases.lock:
        if (replayed := _replay(cases, customer_id, idempotency_key, fingerprint)) is not None:
            return replayed
        if case_id is not None:
            case = _owned_case(cases, customer_id, case_id)
            _check_order(case, order_id)
            state, history, summary = CaseState.of(case), list(case.messages), cases.summary(case_id)
        earlier = earlier_cases(cases, customer_id, case_id, cases.clock())

    # Phase 2: slow work without the lock, so other customers' messages are not blocked.
    recognition = recognizer.recognize(message, history)
    plan = _plan(orders, customer_id, state, recognition, order_id)
    draft = _draft(reply_agent, plan, customer_id, message, history, summary, earlier, state, cases.clock())

    # Phase 3: re-check and apply.
    with cases.lock:
        if (replayed := _replay(cases, customer_id, idempotency_key, fingerprint)) is not None:
            return replayed  # a duplicate of this request finished first
        if case_id is None:
            case = cases.new_case(customer_id)
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
            _record(cases, case, message, plan.reply)
            return _commit(cases, case, plan.reply, customer_id, idempotency_key, fingerprint)

        review_request = draft.review_request if draft else None
        reply = AssistantReply(draft.reply if draft else plan.reply.reply, plan.reply.issues,
                               plan.reply.needs_human_review or review_request is not None,
                               draft.citations if draft else ())
        if draft:
            logger.info("reply case=%s source=agent rounds=%d tools=%s citations=%s skills=%s", case.case_id,
                        draft.rounds,
                        ",".join(f"{trace.name}:{'ok' if trace.ok else trace.error}" for trace in draft.tool_traces)
                        or "-", ",".join(citation.source_id for citation in draft.citations) or "-",
                        ",".join(draft.skills) or "-")
        else:
            logger.info("reply case=%s source=rules", case.case_id)

        _record(cases, case, message, reply)
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

        turn = _commit(cases, case, reply, customer_id, idempotency_key, fingerprint)
        if summaries is not None and needs_summary(len(case.messages), cases.summary(case.case_id)):
            summaries.submit(case.case_id)
        return turn
