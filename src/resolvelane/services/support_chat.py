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

A refund the agent drafts is priced again in phase 3 against the refunds approved by then,
and recorded as a proposal for a support agent: a new refund, or a new version of the
case's open one. The case goes to the support agents' queue.
"""

import difflib
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from resolvelane import metrics
from resolvelane.assistant import AssistantReply, respond
from resolvelane.case_store import CaseStore, Receipt
from resolvelane.domain.case import (Case, CaseMessage, CaseStatus, CaseSummary, EarlierCase, Issue, MessageRole,
                                     ModelUsage, ReplyTrace, ToolCall, ordered_issues)
from resolvelane.domain.delivery import Order
from resolvelane.domain.knowledge import Citation
from resolvelane.domain.refund import RefundStatus
from resolvelane.llm import Meter, metered
from resolvelane.memory import SummaryQueue, earlier_cases, needs_summary, recent_messages
from resolvelane.recognition import IssueRecognizer, Recognition
from resolvelane.refunds import (RefundInvalid, RefundProposal, Refunded, money, new_refund, new_version,
                                 open_refund)
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
    # id, status, number of versions, payment status: a refund sent meanwhile changes what the reply may say
    refunds: tuple[tuple[str, RefundStatus, int, str | None], ...] = ()

    @classmethod
    def of(cls, case: Case) -> "CaseState":
        return cls(case.order_id, case.status, tuple(case.issues), tuple(case.pending_issues),
                   tuple((refund.refund_id, refund.status, len(refund.versions),
                          refund.payment.status.value if refund.payment else None) for refund in case.refunds))


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
    metrics.IDEMPOTENT_REPLAYS.inc()
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
        if state.status is CaseStatus.RESOLVED:
            reply = replace(reply, needs_human_review=True,
                            reply=reply.reply + " I've reopened your case so a support agent can look at your message.")
        elif in_review and not reply.needs_human_review:
            reply = replace(reply, reply=reply.reply + " Your case is waiting for a support agent to review it.")
    return ReplyPlan(reply, order, new_issues, lookup, effective_order_id)


def _draft(agent: ReplyAgent | None, cases: CaseStore, plan: ReplyPlan, customer_id: str, message: str,
           history: Sequence[CaseMessage], summary: CaseSummary | None, earlier: Sequence[EarlierCase],
           refunds: Sequence, state: CaseState, now: datetime) -> tuple[AgentDraft | None, str]:
    """The agent's draft, or None and why the rule-based reply is used instead."""
    if agent is None or plan.redirect:
        return None, "none"  # rules by design, not a fallback
    refunded = Refunded.of(cases.approved_refunds(plan.order.order_id)) if plan.order else Refunded()
    request = ReplyRequest(customer_id, message, recent_messages(history, summary), plan.order, plan.reply.issues,
                           ordered_issues(list(state.issues) + plan.new_issues), state.status,
                           plan.reply.needs_human_review, plan.reply.reply, now,
                           summary.text if summary else None, earlier, refunds, refunded)
    try:
        return agent.draft(request), "none"
    except AgentFailed as error:
        logger.warning("reply_agent_failed reason=%s; using the rule-based reply", error)
        return None, str(error)
    except Exception:
        logger.exception("reply_agent_error; using the rule-based reply")
        return None, "error"


def _apply_refund_proposal(cases: CaseStore, case: Case, order: Order, proposal: RefundProposal) -> str:
    """Record the agent's draft refund and say what happened, for the support agent's notes."""
    current = open_refund(case.refunds)
    if current is not None and current.current.author_role == "support_agent":
        metrics.REFUND_EVENTS.labels("draft_dropped", "assistant").inc()
        return "The assistant's refund draft was not recorded: a support agent is handling the refund"
    try:
        # Priced again: another refund of the order may have been approved since the draft.
        version = new_version(order, proposal, Refunded.of(cases.approved_refunds(order.order_id)),
                              len(current.versions) + 1 if current else 1, "assistant", None, cases.clock())
    except RefundInvalid as error:
        logger.info("refund_proposal case=%s recorded=false reason=%s", case.case_id, error)
        metrics.REFUND_EVENTS.labels("draft_dropped", "assistant").inc()
        return f"The assistant's refund draft was not recorded: {error}"
    if current is not None and current.current.same_content(version):
        return f"The assistant's refund draft is unchanged: {proposal.reason}"
    if current is not None:
        current.revise(version)
        refund = current
    else:
        refund = new_refund(case.case_id, order, version, cases.clock())
        case.refunds.append(refund)
    metrics.REFUND_EVENTS.labels("revised" if current else "proposed", "assistant").inc()
    logger.info("refund_proposal case=%s refund=%s version=%d amount_minor=%d author=assistant", case.case_id,
                refund.refund_id, version.version, version.amount_minor)
    return f"Refund draft of {money(version.amount_minor, refund.currency)} (version {version.version}): {proposal.reason}"


REPEAT_SIMILARITY = 0.9


def _repeats(case: Case, reply: str) -> bool:
    """Whether the reply is nearly the same as the assistant's previous reply in the case: a
    sign the conversation is not moving on, seen in live runs."""
    previous = next((message.text for message in reversed(case.messages)
                     if message.role is MessageRole.ASSISTANT), None)
    return previous is not None and difflib.SequenceMatcher(None, previous, reply).ratio() >= REPEAT_SIMILARITY


def _record(cases: CaseStore, case: Case, message: str, reply: AssistantReply, trace: ReplyTrace) -> None:
    now = cases.clock()
    case.messages.append(CaseMessage(MessageRole.CUSTOMER, message, now))
    case.messages.append(CaseMessage(MessageRole.ASSISTANT, reply.reply, now, reply.citations, trace=trace))


def _usage(meter: Meter) -> ModelUsage:
    usage, cost = meter.usage(), meter.cost()
    return ModelUsage(len(meter.calls), usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
                      usage.cache_write_tokens, None if cost is None else round(cost, 6))


def _trace(recognition: Recognition, draft: AgentDraft | None, fallback: str, agent: ReplyAgent | None,
           seconds: float, repeat: bool, meter: Meter) -> ReplyTrace:
    llm = getattr(agent, "llm", None)
    return ReplyTrace(
        written_by="agent" if draft else "rules", fallback=fallback, recognition=recognition.source,
        issues=tuple(recognition.issues), seconds=round(seconds, 2),
        model=getattr(llm, "model", None) if fallback != "none" or draft else None,
        rounds=draft.rounds if draft else 0,
        tools=tuple(ToolCall(trace.name, trace.ok, round(trace.latency_ms, 1), trace.error)
                    for trace in draft.tool_traces) if draft else (),
        skills=tuple(draft.skills) if draft else (), repeat=repeat, usage=_usage(meter))


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
    state, history, summary, refunds = NEW_CASE, [], None, []
    with cases.lock:
        if (replayed := _replay(cases, customer_id, idempotency_key, fingerprint)) is not None:
            return replayed
        if case_id is not None:
            case = _owned_case(cases, customer_id, case_id)
            _check_order(case, order_id)
            state, history, summary = CaseState.of(case), list(case.messages), cases.summary(case_id)
            refunds = case.refunds
        earlier = earlier_cases(cases, customer_id, case_id, cases.clock())

    # Phase 2: slow work without the lock, so other customers' messages are not blocked.
    # Every model call made for this message is metered, including answers that are not used.
    started = time.perf_counter()
    with metered() as meter:
        recognition = recognizer.recognize(message, history)
        plan = _plan(orders, customer_id, state, recognition, order_id)
        draft, fallback = _draft(reply_agent, cases, plan, customer_id, message, history, summary, earlier,
                                 refunds, state, cases.clock())
    seconds = time.perf_counter() - started
    metrics.RECOGNITIONS.labels(recognition.source).inc()

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
            if draft is not None:
                fallback = "case_changed"
            plan, draft = _plan(orders, customer_id, current, recognition, order_id), None
        if case.status is CaseStatus.RESOLVED:
            # A support agent made the last decision, so they see the follow-up.
            case.move_to(CaseStatus.NEEDS_REVIEW, cases.clock(), "customer", customer_id,
                         "The customer wrote again after the case was resolved")
            logger.info("case_status case=%s status=needs_review reopened_by=customer", case.case_id)

        logger.info("issue_recognition case=%s source=%s issues=%s", case.case_id, recognition.source,
                    ",".join(issue.value for issue in recognition.issues))
        if plan.lookup is not None:
            logger.info("order_lookup case=%s access=%s subject=%s order_id=%r", case.case_id,
                        plan.lookup.access.value, customer_id, plan.looked_up_order_id)
        if plan.redirect:
            metrics.REPLIES.labels("rules", "none").inc()
            _record(cases, case, message, plan.reply, _trace(recognition, None, "none", None, seconds, False, meter))
            return _commit(cases, case, plan.reply, customer_id, idempotency_key, fingerprint)

        review_request = draft.review_request if draft else None
        proposal = draft.refund_proposal if draft and plan.order is not None else None
        reply = AssistantReply(draft.reply if draft else plan.reply.reply, plan.reply.issues,
                               plan.reply.needs_human_review or review_request is not None or proposal is not None,
                               draft.citations if draft else ())
        if draft:
            logger.info("reply case=%s source=agent rounds=%d tools=%s citations=%s skills=%s", case.case_id,
                        draft.rounds,
                        ",".join(f"{trace.name}:{'ok' if trace.ok else trace.error}" for trace in draft.tool_traces)
                        or "-", ",".join(citation.source_id for citation in draft.citations) or "-",
                        ",".join(draft.skills) or "-")
        else:
            logger.info("reply case=%s source=rules fallback=%s", case.case_id, fallback)
        source = "agent" if draft else "rules"
        metrics.REPLIES.labels(source, fallback).inc()
        repeat = _repeats(case, reply.reply)
        if repeat:
            metrics.REPLY_REPEATS.labels(source).inc()
            logger.info("reply_repeat case=%s source=%s", case.case_id, source)

        _record(cases, case, message, reply, _trace(recognition, draft, fallback, reply_agent, seconds, repeat, meter))
        case.issues = ordered_issues(case.issues + plan.new_issues)
        if plan.order is not None:
            # Bind only an order the customer may access, never one that failed the lookup.
            case.order_id = plan.order.order_id
            case.pending_issues = []
        else:
            case.pending_issues = ordered_issues(case.pending_issues + plan.new_issues)
        if review_request is not None:
            case.review_notes.append(review_request.reason)
        if proposal is not None:
            case.review_notes.append(_apply_refund_proposal(cases, case, plan.order, proposal))
        if reply.needs_human_review and case.status is not CaseStatus.NEEDS_REVIEW:
            if plan.reply.needs_human_review:
                reason = f"Rules: {', '.join(issue.value for issue in plan.reply.issues)} needs a support agent"
            elif review_request is not None:
                reason = f"Assistant: {review_request.reason}"
            else:
                reason = "Assistant: drafted a refund for a support agent to decide"
            case.move_to(CaseStatus.NEEDS_REVIEW, cases.clock(), "assistant", None, reason)
            logger.info("case_status case=%s status=%s issues=%s requested_by=%s", case.case_id,
                        case.status.value, ",".join(issue.value for issue in case.issues),
                        "rules" if plan.reply.needs_human_review else "agent")

        turn = _commit(cases, case, reply, customer_id, idempotency_key, fingerprint)
        if summaries is not None and needs_summary(len(case.messages), cases.summary(case.case_id)):
            summaries.submit(case.case_id)
        return turn
