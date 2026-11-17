import asyncio
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from resolvelane import payments
from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import CaseStore
from resolvelane.domain.case import CaseStatus, MessageRole
from resolvelane.domain.refund import PaymentStatus
from resolvelane.payments import MAX_ATTEMPTS, PaymentDispatcher, load_payment_dispatcher, send_due
from resolvelane.refunds import RefundProposal
from resolvelane.services import support_desk
from resolvelane.simulator.fixtures import ORDERS
from resolvelane.simulator.order_source import InMemoryOrderSource
from resolvelane.simulator.payment_provider import (DECLINE, TIMEOUT_AFTER, TIMEOUT_BEFORE, UNAVAILABLE,
                                                    SimulatedPaymentProvider)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ORDERS_SOURCE = InMemoryOrderSource(ORDERS)
AGENT = "demo-agent-301"


class ManualClock:
    """Moves a millisecond per reading, so events are ordered; tests move it further explicitly."""

    def __init__(self):
        self.now = NOW

    def __call__(self):
        self.now += timedelta(milliseconds=1)
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def store(clock):
    return CaseStore(clock=clock)


def approved_case(store, items=(("item-1", 1),), resolve=False):
    case = store.new_case("demo-customer-101")
    case.order_id = "demo-order-1001"
    store.commit(case)
    refund = support_desk.propose_refund(store, ORDERS_SOURCE, AGENT, case.case_id,
                                         RefundProposal(tuple(items), False, "Checked with the restaurant"))
    support_desk.approve_refund(store, ORDERS_SOURCE, AGENT, refund.refund_id, 1)
    if resolve:
        support_desk.resolve(store, AGENT, case.case_id, "Refunded")
    return case.case_id


def payment_of(store, case_id):
    return store.get(case_id).refunds[0].payment


# --- the outbox ------------------------------------------------------------------------

def test_approval_writes_the_payment_with_the_decision(store):
    case_id = approved_case(store)
    payment = payment_of(store, case_id)
    assert (payment.status, payment.amount_minor, payment.currency, payment.attempts) == \
        (PaymentStatus.PENDING, 950, "USD", 0)
    assert store.due_payments(store.clock(), 10) == [(case_id, payment.payment_id)]


def test_rejected_refund_gets_no_payment(store):
    case = store.new_case("demo-customer-101")
    case.order_id = "demo-order-1001"
    store.commit(case)
    refund = support_desk.propose_refund(store, ORDERS_SOURCE, AGENT, case.case_id,
                                         RefundProposal((("item-1", 1),), False, "Report"))
    support_desk.reject_refund(store, AGENT, refund.refund_id, 1, "Packed per restaurant")
    assert store.get(case.case_id).refunds[0].payment is None
    assert store.due_payments(store.clock(), 10) == []


def test_retried_approval_creates_no_second_payment(store):
    case = store.new_case("demo-customer-101")
    case.order_id = "demo-order-1001"
    store.commit(case)
    refund = support_desk.propose_refund(store, ORDERS_SOURCE, AGENT, case.case_id,
                                         RefundProposal((("item-1", 1),), False, "Report"))
    for _ in range(2):
        support_desk.approve_refund(store, ORDERS_SOURCE, AGENT, refund.refund_id, 1, key="approve-1")
    assert len(store.due_payments(store.clock(), 10)) == 1


# --- sending ------------------------------------------------------------------------------

def test_sent_payment_tells_the_customer(store):
    provider = SimulatedPaymentProvider()
    case_id = approved_case(store)
    assert send_due(store, provider) == 1
    payment = payment_of(store, case_id)
    assert payment.status is PaymentStatus.SUCCEEDED and payment.provider_refund_id == "sim-refund-0001"
    [made] = provider.refunds()
    assert (made.idempotency_key, made.amount_minor, made.reference) == \
        (payment.payment_id, 950, store.get(case_id).refunds[0].refund_id)
    notice = store.get(case_id).messages[-1]
    assert notice.role is MessageRole.SYSTEM
    assert notice.text == ("Your refund of 9.50 USD for 1 x Vegetable wrap has been sent to your original "
                           "payment method.")
    assert send_due(store, provider) == 0  # nothing left to send


def test_notice_does_not_reopen_a_resolved_case(store):
    case_id = approved_case(store, resolve=True)
    send_due(store, SimulatedPaymentProvider())
    assert store.get(case_id).status is CaseStatus.RESOLVED


@pytest.mark.parametrize("fault", [TIMEOUT_BEFORE, UNAVAILABLE])
def test_no_answer_is_retried_later(store, clock, fault):
    provider = SimulatedPaymentProvider([fault])
    case_id = approved_case(store)
    send_due(store, provider)
    payment = payment_of(store, case_id)
    assert (payment.status, payment.attempts) == (PaymentStatus.PENDING, 1)
    assert payment.last_error.startswith(("ProviderTimeout", "ProviderUnavailable"))
    assert send_due(store, provider) == 0  # not due yet
    clock.advance(seconds=2)
    send_due(store, provider)
    assert payment_of(store, case_id).status is PaymentStatus.SUCCEEDED
    assert len(provider.refunds()) == 1


def test_timeout_after_the_refund_was_made_is_reconciled_not_sent_again(store, clock):
    provider = SimulatedPaymentProvider([TIMEOUT_AFTER])
    case_id = approved_case(store)
    send_due(store, provider)
    assert payment_of(store, case_id).status is PaymentStatus.PENDING  # we could not tell
    clock.advance(seconds=2)
    send_due(store, provider)
    assert payment_of(store, case_id).status is PaymentStatus.SUCCEEDED
    assert provider.create_calls == 1  # the second attempt only looked it up


def test_crash_between_sending_and_recording(store, clock):
    provider = SimulatedPaymentProvider()
    case_id = approved_case(store)
    payment_id = payment_of(store, case_id).payment_id
    claim = payments._claim(store, case_id, payment_id)
    payments._ask(provider, claim)  # the provider refunds; the process stops before recording it
    assert send_due(store, provider) == 0  # the claim's lease holds
    clock.advance(seconds=61)
    send_due(store, provider)
    assert payment_of(store, case_id).status is PaymentStatus.SUCCEEDED
    assert provider.create_calls == 1


def test_declined_payment_goes_back_to_a_support_agent(store):
    provider = SimulatedPaymentProvider([DECLINE])
    case_id = approved_case(store, resolve=True)
    send_due(store, provider)
    case = store.get(case_id)
    payment = case.refunds[0].payment
    assert payment.status is PaymentStatus.FAILED
    assert payment.last_error == "The original payment method can no longer receive refunds"
    assert case.status is CaseStatus.NEEDS_REVIEW
    assert (case.transitions[-1].actor_role, case.transitions[-1].note) == \
        ("system", "Refund payment declined by the provider: The original payment method can no longer receive refunds")
    assert case.messages[-1].role is MessageRole.SUPPORT_AGENT  # the customer is told by an agent, not a notice


def test_payment_without_an_answer_becomes_unconfirmed_and_is_only_checked(store, clock):
    provider = SimulatedPaymentProvider([TIMEOUT_BEFORE] * MAX_ATTEMPTS)
    case_id = approved_case(store)
    for _ in range(MAX_ATTEMPTS):
        clock.advance(minutes=5)
        send_due(store, provider)
    case = store.get(case_id)
    payment = case.refunds[0].payment
    assert (payment.status, payment.attempts) == (PaymentStatus.UNCONFIRMED, MAX_ATTEMPTS)
    assert case.status is CaseStatus.NEEDS_REVIEW
    assert case.review_notes[-1].startswith(f"Refund payment unconfirmed after {MAX_ATTEMPTS} attempts")

    clock.advance(minutes=10)
    send_due(store, provider)
    assert provider.create_calls == MAX_ATTEMPTS  # checked, not sent again
    # The provider's record turns up later (e.g. it processed a request it had not answered).
    provider.create_refund(payment.payment_id, 950, "USD", "late")
    clock.advance(minutes=10)
    send_due(store, provider)
    assert payment_of(store, case_id).status is PaymentStatus.SUCCEEDED


def test_payments_survive_a_restart(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    store = CaseStore(path)
    case_id = approved_case(store)
    store.close()
    reopened = CaseStore(path)
    provider = SimulatedPaymentProvider()
    send_due(reopened, provider)
    assert payment_of(reopened, case_id).status is PaymentStatus.SUCCEEDED
    payment_id = payment_of(reopened, case_id).payment_id
    reopened.close()

    db = sqlite3.connect(path)
    for statement in ("UPDATE payments SET status = 'pending' WHERE payment_id = ?",
                      "UPDATE payments SET amount_minor = 1 WHERE payment_id = ?",
                      "DELETE FROM payments WHERE payment_id = ?"):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(statement, (payment_id,))
    db.close()


def test_provider_refuses_a_reused_key_with_another_amount():
    provider = SimulatedPaymentProvider()
    first = provider.create_refund("key-1", 950, "USD", "ref")
    assert provider.create_refund("key-1", 950, "USD", "ref") == first
    with pytest.raises(Exception, match="key-1"):
        provider.create_refund("key-1", 1200, "USD", "ref")
    with pytest.raises(ValueError, match="Unknown payment faults"):
        SimulatedPaymentProvider(["explode"])


# --- the dispatcher and the API ------------------------------------------------------------

def test_dispatcher_sends_in_the_background(store):
    dispatcher = PaymentDispatcher(store, SimulatedPaymentProvider(), poll_seconds=0.05)
    try:
        case_id = approved_case(store)
        dispatcher.poke()
        deadline = time.monotonic() + 5
        while payment_of(store, case_id).status is PaymentStatus.PENDING and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        dispatcher.close()
    assert payment_of(store, case_id).status is PaymentStatus.SUCCEEDED


def test_payments_can_be_paused(store):
    assert load_payment_dispatcher(store, SimulatedPaymentProvider(), {"RESOLVELANE_PAYMENTS": "paused"}) is None


TOKENS = {"test-token-demo-customer-101-xxxxxxxx": {"subject_id": "demo-customer-101", "role": "customer"},
          "test-token-demo-agent-301-xxxxxxxxxxx": {"subject_id": "demo-agent-301", "role": "support_agent"}}


def get(app, path, token):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.get(path, headers={"Authorization": f"Bearer {token}"})
    return asyncio.run(go()).json()


def test_customer_sees_when_the_refund_was_sent_and_agents_see_the_payment(store):
    provider = SimulatedPaymentProvider()
    app = create_app(TokenRegistry(TOKENS), case_store=store, payment_provider=provider)
    case_id = approved_case(store)
    customer_token, agent_token = list(TOKENS)
    assert get(app, f"/api/v1/cases/{case_id}", customer_token)["refunds"][0]["payment_status"] == "processing"
    send_due(store, provider)
    customer = get(app, f"/api/v1/cases/{case_id}", customer_token)
    refund = customer["refunds"][0]
    assert refund["payment_status"] == "sent" and refund["sent_at"] is not None
    assert "payment" not in refund and "provider_refund_id" not in str(customer)
    assert customer["messages"][-1]["role"] == "system"
    payment = get(app, f"/api/v1/support/cases/{case_id}", agent_token)["refunds"][0]["payment"]
    assert (payment["status"], payment["attempts"], payment["provider_refund_id"]) == ("succeeded", 1,
                                                                                       "sim-refund-0001")


def test_saying_a_refund_was_sent_needs_a_sent_payment(store):
    import json
    from types import SimpleNamespace

    from resolvelane.llm import LLMProvider
    from resolvelane.reply_agent import AgentFailed, ReplyAgent, ReplyRequest
    from resolvelane.resilience import CircuitBreaker
    from resolvelane.simulator.fixtures import POLICIES

    def agent(text):
        answer = SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="text", text=json.dumps({"reply": text, "sources": []}))])
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: answer))
        return ReplyAgent(LLMProvider(client, "test-model", CircuitBreaker("llm", 3, 30)), POLICIES)

    def ask(refunds):
        order = next(order for order in ORDERS if order.order_id == "demo-order-1001")
        return ReplyRequest("demo-customer-101", "Was it sent?", [], order, [], [], CaseStatus.NEEDS_REVIEW,
                            False, "Reference reply.", NOW, refunds=refunds)

    case_id = approved_case(store)
    with pytest.raises(AgentFailed, match="commits_to_an_outcome"):
        agent("Your refund has been sent.").draft(ask(store.get(case_id).refunds))
    send_due(store, SimulatedPaymentProvider())
    sent = store.get(case_id).refunds
    assert agent("Your refund has been sent.").draft(ask(sent)).reply == "Your refund has been sent."
    with pytest.raises(AgentFailed, match="commits_to_an_outcome"):
        agent("We will refund you again.").draft(ask(sent))


def post(app, path, token, body):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post(path, headers={"Authorization": f"Bearer {token}"}, json=body)
    return asyncio.run(go()).json()


def test_reply_written_while_the_refund_was_sent_is_decided_again(store):
    # Found in a live run: the draft said "not sent yet" although the notice had just been recorded.
    from resolvelane.reply_agent import AgentDraft

    provider = SimulatedPaymentProvider()

    class SendsWhileDrafting:
        def draft(self, request):
            send_due(store, provider)  # the dispatcher finishes while the model is writing
            return AgentDraft("Your refund is approved and hasn't been sent yet.", None, [], 1)

    app = create_app(TokenRegistry(TOKENS), case_store=store, reply_agent=SendsWhileDrafting(),
                     payment_provider=provider)
    case_id = approved_case(store)
    customer_token, _ = list(TOKENS)
    turn = post(app, "/api/v1/chat", customer_token, {"message": "Has it been sent?", "case_id": case_id})
    assert "hasn't been sent" not in turn["reply"]
    messages = store.get(case_id).messages
    assert [message.role for message in messages[-3:]] == [MessageRole.SYSTEM, MessageRole.CUSTOMER,
                                                           MessageRole.ASSISTANT]


def test_declined_refund_is_delayed_for_the_customer_and_the_assistant(store):
    from resolvelane.reply_agent import _refund_fact

    provider = SimulatedPaymentProvider([DECLINE])
    app = create_app(TokenRegistry(TOKENS), case_store=store, payment_provider=provider)
    case_id = approved_case(store)
    send_due(store, provider)
    customer_token, _ = list(TOKENS)
    assert get(app, f"/api/v1/cases/{case_id}", customer_token)["refunds"][0]["payment_status"] == "delayed"
    assert _refund_fact(store.get(case_id).refunds[0])["payment"] == "delayed"
