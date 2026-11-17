import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import CaseStore
from resolvelane.domain.case import CaseStatus
from resolvelane.llm import LLMProvider
from resolvelane.refunds import RefundInvalid, RefundProposal, Refunded, describe, price_refund
from resolvelane.reply_agent import AgentDraft, ReplyAgent, ReplyRequest, case_facts
from resolvelane.resilience import CircuitBreaker
from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.tools import ToolContext, ToolGateway

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ORDER = next(order for order in ORDERS if order.order_id == "demo-order-1001")  # 2 wraps 9.50, water 2.50, fee 3.99
IN_PROGRESS = next(order for order in ORDERS if order.order_id == "demo-order-1002")
# Invented test credentials for fictional principals.
TOKENS = {"test-token-demo-customer-101-xxxxxxxx": {"subject_id": "demo-customer-101", "role": "customer"},
          "test-token-demo-agent-301-xxxxxxxxxxx": {"subject_id": "demo-agent-301", "role": "support_agent"},
          "test-token-demo-agent-302-xxxxxxxxxxx": {"subject_id": "demo-agent-302", "role": "support_agent"}}


def ask(items=(("item-1", 1),), fee=False, reason="Customer reports a missing wrap"):
    return RefundProposal(tuple(items), fee, reason)


# --- pricing from the order record ------------------------------------------------------

def test_amount_comes_from_the_order_record():
    lines, fee = price_refund(ORDER, ask((("item-1", 1), ("item-2", 1)), fee=True), Refunded())
    assert [(line.name, line.quantity, line.amount_minor) for line in lines] == \
        [("Vegetable wrap", 1, 950), ("Sparkling water", 1, 250)]
    assert fee == 399


@pytest.mark.parametrize("proposal,problem", [
    (ask((("item-9", 1),)), "no item item-9"),
    (ask((("item-1", 3),)), "2 x Vegetable wrap"),
    (ask((("item-1", 1), ("item-1", 1))), "once"),
    (ask(()), "at least one"),
])
def test_proposal_the_order_does_not_allow_is_rejected(proposal, problem):
    with pytest.raises(RefundInvalid, match=problem):
        price_refund(ORDER, proposal, Refunded())


def test_only_delivered_orders_are_refunded():
    with pytest.raises(RefundInvalid, match="not been delivered"):
        price_refund(IN_PROGRESS, ask(), Refunded())


def test_approved_refunds_limit_what_remains():
    refunded = Refunded({"item-1": 2}, delivery_fee=True)
    with pytest.raises(RefundInvalid, match="0 x Vegetable wrap"):
        price_refund(ORDER, ask(), refunded)
    with pytest.raises(RefundInvalid, match="delivery fee was already refunded"):
        price_refund(ORDER, ask((("item-2", 1),), fee=True), refunded)


# --- the assistant's tool ---------------------------------------------------------------

def run_tool(args, order=ORDER, refunds=()):
    context = ToolContext("demo-customer-101", order, NOW, POLICIES, refunds=refunds)
    result = ToolGateway().execute("propose_refund", args, context)
    return json.loads(result.content), result, context


def test_tool_records_a_draft_priced_by_code():
    payload, result, context = run_tool({"items": [{"item_id": "item-1", "quantity": 1}],
                                         "reason": "One wrap missing"})
    assert not result.is_error
    assert payload["amount"] == "9.50 USD" and payload["items"] == ["1 x Vegetable wrap"]
    assert "Do not state the amount" in payload["note"]
    assert context.refund_proposal == ask(reason="One wrap missing")


def test_tool_refuses_what_the_order_does_not_allow():
    payload, result, context = run_tool({"items": [{"item_id": "item-1", "quantity": 5}], "reason": "Five wraps"})
    assert result.trace.error == "refused" and "2 x Vegetable wrap" in payload["error"]
    assert context.refund_proposal is None
    payload, result, _ = run_tool({"items": [{"item_id": "item-1", "quantity": 1}], "reason": "x"}, order=None)
    assert result.trace.error == "refused"


def test_tool_cannot_name_an_amount():
    _, result, _ = run_tool({"items": [{"item_id": "item-1", "quantity": 1}], "reason": "x", "amount": 5000})
    assert result.trace.error == "invalid_input"


# --- through the API ----------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        self.now += timedelta(seconds=1)
        return self.now


class DraftingAgent:
    """Stands in for the reply agent: drafts the refunds it is given, one per message."""

    def __init__(self):
        self.proposals = []
        self.requests = []

    def draft(self, request):
        self.requests.append(request)
        proposal = self.proposals.pop(0) if self.proposals else None
        return AgentDraft("A support agent will review your refund request.", None, [], 1,
                          refund_proposal=proposal)


class Desk:
    def __init__(self, path=None):
        self.agent = DraftingAgent()
        self.store = CaseStore(path, clock=Clock()) if path else CaseStore(clock=Clock())
        self.app = create_app(TokenRegistry(TOKENS), case_store=self.store, reply_agent=self.agent)

    def send(self, method, path, who, expect=200, key=None, **kwargs):
        token = next(token for token, account in TOKENS.items() if account["subject_id"] == who)
        headers = {"Authorization": f"Bearer {token}", **({"Idempotency-Key": key} if key else {})}

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://testserver") as client:
                return await client.request(method, path, headers=headers, **kwargs)
        response = asyncio.run(go())
        assert response.status_code == expect, response.text
        return response.json()

    def chat(self, message, case_id=None, proposal=None):
        if proposal is not None:
            self.agent.proposals.append(proposal)
        body = {"message": message, **({"case_id": case_id} if case_id else {"order_id": "demo-order-1001"})}
        return self.send("POST", "/api/v1/chat", "demo-customer-101", json=body)

    def case(self, case_id):
        return self.send("GET", f"/api/v1/support/cases/{case_id}", "demo-agent-301")

    def customer_case(self, case_id):
        return self.send("GET", f"/api/v1/cases/{case_id}", "demo-customer-101")

    def refund(self, case_id):
        return self.case(case_id)["refunds"][-1]

    def propose(self, case_id, items=(("item-1", 1),), fee=False, agent="demo-agent-301", expect=200):
        return self.send("POST", f"/api/v1/support/cases/{case_id}/refunds", agent, expect=expect,
                         json={"items": [{"item_id": i, "quantity": q} for i, q in items],
                               "include_delivery_fee": fee, "reason": "Checked with the restaurant"})

    def revise(self, refund_id, revises, items, agent="demo-agent-301", expect=200):
        return self.send("POST", f"/api/v1/support/refunds/{refund_id}/versions", agent, expect=expect,
                         json={"revises": revises, "items": [{"item_id": i, "quantity": q} for i, q in items],
                               "reason": "Restaurant confirmed"})

    def approve(self, refund_id, version, agent="demo-agent-301", expect=200, key=None):
        return self.send("POST", f"/api/v1/support/refunds/{refund_id}/approval", agent, expect=expect, key=key,
                         json={"version": version})

    def reject(self, refund_id, version, agent="demo-agent-301", expect=200):
        return self.send("POST", f"/api/v1/support/refunds/{refund_id}/rejection", agent, expect=expect,
                         json={"version": version, "note": "The restaurant shows both wraps packed"})


@pytest.fixture
def desk():
    return Desk()


def test_assistant_draft_becomes_a_proposal_for_a_support_agent(desk):
    turn = desk.chat("One of my wraps is missing.", proposal=ask())
    assert turn["case_status"] == "needs_review" and turn["needs_human_review"]
    case = desk.case(turn["case_id"])
    refund = case["refunds"][0]
    assert refund["status"] == "proposed" and refund["currency"] == "USD"
    assert [(v["version"], v["amount_minor"], v["author_role"], v["author_id"]) for v in refund["versions"]] == \
        [(1, 950, "assistant", None)]
    assert "Refund draft of 9.50 USD (version 1): Customer reports a missing wrap" in case["review_notes"]
    assert desk.customer_case(turn["case_id"])["refunds"] == []  # a proposal is not something to rely on


def test_assistant_draft_alone_sends_the_case_for_review(desk):
    turn = desk.chat("Where is my order?", proposal=ask())  # rules alone would not ask for review
    assert turn["case_status"] == "needs_review"
    transition = desk.case(turn["case_id"])["transitions"][-1]
    assert transition["note"] == "Assistant: drafted a refund for a support agent to decide"


def test_follow_up_revises_the_open_proposal_and_keeps_earlier_versions(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    desk.chat("And the water too.", case_id, proposal=ask((("item-1", 1), ("item-2", 1))))
    desk.chat("Any news?", case_id, proposal=ask((("item-1", 1), ("item-2", 1))))  # same draft: no new version
    refund = desk.refund(case_id)
    assert len(desk.case(case_id)["refunds"]) == 1
    assert [(v["version"], v["amount_minor"]) for v in refund["versions"]] == [(1, 950), (2, 1200)]


def test_approval_must_name_the_current_version(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    # The agent reviewed version 1; meanwhile the customer added the water and the assistant revised it.
    desk.chat("And the water too.", case_id, proposal=ask((("item-1", 1), ("item-2", 1))))
    assert desk.approve(refund_id, 1, expect=409) == {
        "detail": "Version 1 is not the current version of this refund; review version 2"}
    result = desk.approve(refund_id, 2)
    assert result == {"refund_id": refund_id, "case_id": case_id, "status": "approved", "version": 2,
                      "amount_minor": 1200, "currency": "USD"}


def test_approval_tells_the_customer_and_is_final(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    desk.approve(refund_id, 1)
    customer = desk.customer_case(case_id)
    assert customer["messages"][-1]["role"] == "support_agent"
    assert customer["messages"][-1]["text"] == ("I've approved a refund of 9.50 USD for 1 x Vegetable wrap. "
                                                "You'll get a message here once it has been sent to your "
                                                "original payment method.")
    assert [(item["status"], item["amount_minor"], item["payment_status"]) for item in customer["refunds"]] == \
        [("approved", 950, "processing")]
    refund = desk.refund(case_id)
    assert (refund["decided_version"], refund["decided_by"]) == (1, "demo-agent-301")
    assert desk.approve(refund_id, 1, agent="demo-agent-302", expect=409) == {
        "detail": "This refund was already approved"}
    desk.reject(refund_id, 1, expect=409)


def test_rejection_is_internal(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    assert desk.reject(refund_id, 1)["status"] == "rejected"
    assert desk.refund(case_id)["decision_note"] == "The restaurant shows both wraps packed"
    customer = desk.customer_case(case_id)
    assert customer["refunds"] == [] and customer["messages"][-1]["role"] == "assistant"


def test_support_agent_revision_is_not_overwritten_by_the_assistant(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    desk.revise(refund_id, 1, (("item-1", 2),))
    desk.chat("Actually it was the water.", case_id, proposal=ask((("item-2", 1),)))
    versions = desk.refund(case_id)["versions"]
    assert [(v["version"], v["author_role"], v["author_id"]) for v in versions] == \
        [(1, "assistant", None), (2, "support_agent", "demo-agent-301")]
    assert "a support agent is handling the refund" in desk.case(case_id)["review_notes"][-1]
    # The assistant is told so too, through the refund state it is given.
    assert desk.agent.requests[-1].refunds[0].current.author_role == "support_agent"


def test_agent_proposals_follow_the_order_record(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    assert "2 x Vegetable wrap" in desk.propose(case_id, (("item-1", 3),), expect=422)["detail"]
    result = desk.propose(case_id, (("item-1", 2),), fee=True)
    assert result["amount_minor"] == 2 * 950 + 399 and result["version"] == 1
    assert desk.propose(case_id, expect=409) == {
        "detail": "This case already has a refund proposal; revise it instead"}
    assert desk.revise(result["refund_id"], 3, (("item-1", 1),), expect=409)["detail"].startswith("Version 3")


def test_case_without_an_order_has_nothing_to_refund(desk):
    case_id = desk.send("POST", "/api/v1/chat", "demo-customer-101", json={"message": "Hello"})["case_id"]
    assert desk.propose(case_id, expect=409) == {"detail": "This case has no order to refund"}


def test_case_with_an_undecided_refund_cannot_be_resolved(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    resolve = {"note": "Done"}
    assert desk.send("POST", f"/api/v1/support/cases/{case_id}/resolution", "demo-agent-301", expect=409,
                     json=resolve) == {"detail": "Approve or reject the refund proposal before resolving the case"}
    desk.approve(desk.refund(case_id)["refund_id"], 1)
    desk.send("POST", f"/api/v1/support/cases/{case_id}/resolution", "demo-agent-301", json=resolve)


def test_order_limits_hold_across_cases(desk):
    first = desk.chat("Both wraps are missing.", proposal=ask((("item-1", 2),)))["case_id"]
    second = desk.chat("My wraps never came.", proposal=ask((("item-1", 2),)))["case_id"]
    desk.approve(desk.refund(first)["refund_id"], 1)
    # The second proposal was valid when written; approving it now would refund the wraps twice.
    detail = desk.approve(desk.refund(second)["refund_id"], 1, expect=409)["detail"]
    assert detail == "0 x Vegetable wrap (item-1) can still be refunded, not 2. Revise the proposal"
    # And the assistant can no longer draft one.
    desk.chat("Still waiting on the wraps.", second, proposal=ask((("item-1", 1),)))
    assert desk.case(second)["review_notes"][-1].startswith("The assistant's refund draft was not recorded")
    assert desk.agent.requests[-1].refunded.quantities == {"item-1": 2}


def test_refund_actions_are_idempotent(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    first = desk.approve(refund_id, 1, key="approve-1")
    assert desk.approve(refund_id, 1, key="approve-1") == first  # replayed, not "already approved"
    assert len(desk.customer_case(case_id)["messages"]) == 3
    desk.approve(refund_id, 2, key="approve-1", expect=422)


def test_refund_desk_is_for_support_agents_only(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    desk.send("POST", f"/api/v1/support/refunds/{refund_id}/approval", "demo-customer-101", expect=403,
              json={"version": 1})
    desk.send("POST", f"/api/v1/support/cases/{case_id}/refunds", "demo-customer-101", expect=403,
              json={"items": [{"item_id": "item-1", "quantity": 1}], "reason": "Mine"})
    desk.approve("refund-does-not-exist", 1, expect=404)


def test_assistant_states_an_approved_refund_from_the_facts(desk):
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    assert case_facts(desk.agent.requests[-1], POLICIES)["refunds"] == []
    desk.approve(desk.refund(case_id)["refund_id"], 1)
    desk.chat("Did my refund go through?", case_id)
    facts = case_facts(desk.agent.requests[-1], POLICIES)
    assert facts["refunds"] == [{"status": "approved", "amount": "9.50 USD", "items": "1 x Vegetable wrap",
                                 "payment": "processing"}]


# --- the record -----------------------------------------------------------------------------

def test_refunds_survive_a_restart_and_cannot_be_rewritten(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    desk = Desk(path)
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    refund_id = desk.refund(case_id)["refund_id"]
    desk.revise(refund_id, 1, (("item-1", 2),))
    desk.approve(refund_id, 2)
    desk.store.close()

    reopened = CaseStore(path)
    refund = reopened.get(case_id).refunds[0]
    assert (refund.status.value, refund.decided_version, refund.decided.amount_minor) == ("approved", 2, 1900)
    assert describe(refund.decided) == "2 x Vegetable wrap"
    assert [version.amount_minor for version in reopened.approved_refunds("demo-order-1001")] == [1900]
    reopened.close()

    db = sqlite3.connect(path)
    for statement in ("UPDATE refund_versions SET amount_minor = 1 WHERE refund_id = ?",
                      "DELETE FROM refund_versions WHERE refund_id = ?",
                      "UPDATE refunds SET status = 'rejected' WHERE refund_id = ?",
                      "DELETE FROM refunds WHERE refund_id = ?"):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(statement, (refund_id,))
    db.close()


# --- the reply check ----------------------------------------------------------------------

def scripted(*outcomes):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return outcomes[len(calls) - 1]
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return LLMProvider(client, "test-model", CircuitBreaker("llm", 3, 30)), calls


def answer(text):
    return SimpleNamespace(stop_reason="end_turn",
                           content=[SimpleNamespace(type="text", text=json.dumps({"reply": text, "sources": []}))])


def reply_request(message, refunds=()):
    return ReplyRequest("demo-customer-101", message, [], ORDER, [], [], CaseStatus.NEEDS_REVIEW, False,
                        "Reference reply.", NOW, refunds=refunds)


def test_agent_drafts_through_the_tool_and_the_reply_names_the_review():
    tool_call = SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(
        type="tool_use", id="call-1", name="propose_refund",
        input={"items": [{"item_id": "item-1", "quantity": 1}], "reason": "One wrap missing"})])
    llm, _ = scripted(tool_call, answer("Thanks, I've passed your refund request on."))
    result = ReplyAgent(llm, POLICIES).draft(reply_request("One wrap is missing."))
    assert result.refund_proposal == ask(reason="One wrap missing")
    assert result.reply.endswith("A support agent will review your case.")


def test_saying_a_refund_is_approved_needs_an_approved_refund(tmp_path):
    llm, _ = scripted(answer("Your refund has been approved."))
    with pytest.raises(Exception, match="commits_to_an_outcome"):
        ReplyAgent(llm, POLICIES).draft(reply_request("Is my refund approved?"))

    desk = Desk(str(tmp_path / "cases.sqlite3"))
    case_id = desk.chat("One of my wraps is missing.", proposal=ask())["case_id"]
    desk.approve(desk.refund(case_id)["refund_id"], 1)
    approved = desk.store.get(case_id).refunds
    llm, _ = scripted(answer("Your refund has been approved."), answer("We will refund you again."))
    agent = ReplyAgent(llm, POLICIES)
    assert agent.draft(reply_request("Is my refund approved?", approved)).reply == "Your refund has been approved."
    with pytest.raises(Exception, match="commits_to_an_outcome"):
        agent.draft(reply_request("Can I get more?", approved))


def test_policy_in_effect_says_when_the_delivery_fee_is_refunded():
    # The rule is a policy term, so it lives in the policy the agent may state, not in a handling guide.
    from resolvelane.tools import policy_in_effect
    policy = policy_in_effect(POLICIES, "missing_item", datetime(2026, 11, 1, tzinfo=timezone.utc))["policy"]
    assert policy["version"] == "1.1" and "delivery fee is refunded only when nothing" in policy["guidance"]
