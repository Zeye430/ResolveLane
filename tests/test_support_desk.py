import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import CaseStore
from resolvelane.domain.case import Issue
from resolvelane.reply_agent import AgentDraft

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
# Invented test credentials for fictional principals.
TOKENS = {"test-token-demo-customer-101-xxxxxxxx": {"subject_id": "demo-customer-101", "role": "customer"},
          "test-token-demo-customer-102-xxxxxxxx": {"subject_id": "demo-customer-102", "role": "customer"},
          "test-token-demo-agent-301-xxxxxxxxxxx": {"subject_id": "demo-agent-301", "role": "support_agent"},
          "test-token-demo-agent-302-xxxxxxxxxxx": {"subject_id": "demo-agent-302", "role": "support_agent"}}


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        self.now += timedelta(seconds=1)  # every event happens a moment after the previous one
        return self.now


class Desk:
    def __init__(self, reply_agent=None):
        self.store = CaseStore(clock=Clock())
        self.app = create_app(TokenRegistry(TOKENS), case_store=self.store, reply_agent=reply_agent)

    def send(self, method, path, who, expect=200, key=None, **kwargs):
        token = next(token for token, account in TOKENS.items() if account["subject_id"] == who)
        headers = {"Authorization": f"Bearer {token}", **({"Idempotency-Key": key} if key else {})}

        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://testserver") as client:
                return await client.request(method, path, headers=headers, **kwargs)
        response = asyncio.run(go())
        assert response.status_code == expect, response.text
        return response

    def chat(self, message, case_id=None, customer="demo-customer-101", order_id="demo-order-1001"):
        body = {"message": message, **({"case_id": case_id} if case_id else {"order_id": order_id})}
        return self.send("POST", "/api/v1/chat", customer, json=body).json()

    def queue(self, agent="demo-agent-301", **params):
        return self.send("GET", "/api/v1/support/cases", agent, params=params).json()

    def case(self, case_id, agent="demo-agent-301", expect=200):
        return self.send("GET", f"/api/v1/support/cases/{case_id}", agent, expect=expect).json()

    def reply(self, case_id, text, agent="demo-agent-301", expect=200, key=None):
        return self.send("POST", f"/api/v1/support/cases/{case_id}/messages", agent, expect=expect, key=key,
                         json={"text": text})

    def resolve(self, case_id, note="Checked with the restaurant.", agent="demo-agent-301", expect=200, key=None):
        return self.send("POST", f"/api/v1/support/cases/{case_id}/resolution", agent, expect=expect, key=key,
                         json={"note": note})


@pytest.fixture
def desk():
    return Desk()


# --- the queue ------------------------------------------------------------------------

def test_cases_needing_review_are_queued_longest_waiting_first(desk):
    first = desk.chat("One of my wraps is missing.")["case_id"]
    desk.chat("Where is my order?")  # answered by the assistant; not queued
    second = desk.chat("My order arrived late.")["case_id"]
    page = desk.queue()
    assert [entry["case_id"] for entry in page["cases"]] == [first, second]
    assert page["cases"][0]["issues"] == ["missing_item"] and page["cases"][0]["message_count"] == 2
    assert page["next_cursor"] is None


def test_queue_is_read_a_page_at_a_time(desk):
    ids = [desk.chat("One of my wraps is missing.")["case_id"] for _ in range(5)]
    seen, cursor = [], None
    while True:
        page = desk.queue(limit=2, **({"cursor": cursor} if cursor else {}))
        seen += [entry["case_id"] for entry in page["cases"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == ids


def test_invalid_cursor_is_rejected(desk):
    assert desk.send("GET", "/api/v1/support/cases", "demo-agent-301", expect=400,
                     params={"cursor": "not-a-cursor"}).json() == {"detail": "Invalid cursor"}


def test_desk_is_for_support_agents_only(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    desk.send("GET", "/api/v1/support/cases", "demo-customer-101", expect=403)
    desk.send("GET", f"/api/v1/support/cases/{case_id}", "demo-customer-101", expect=403)
    desk.send("POST", f"/api/v1/support/cases/{case_id}/messages", "demo-customer-101", expect=403,
              json={"text": "I approve my own refund."})
    desk.send("POST", "/api/v1/chat", "demo-agent-301", expect=403, json={"message": "Hello"})


# --- handling a case ----------------------------------------------------------------------

def test_agent_sees_internal_details(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    case = desk.case(case_id)
    assert case["customer_id"] == "demo-customer-101" and case["status"] == "needs_review"
    assert case["transitions"] == [{"at": case["waiting_since"], "from_status": "open", "to_status": "needs_review",
                                    "actor_role": "assistant", "actor_id": None,
                                    "note": "Rules: missing_item needs a support agent"}]
    assert case["summary"] is None
    assert case["order"]["order_id"] == "demo-order-1001" and case["order"]["items"][0]["item_id"] == "item-1"
    desk.case("case-does-not-exist", expect=404)


def test_case_without_a_known_order_has_no_order_record(desk):
    case_id = desk.chat("My order arrived late.", order_id="demo-order-9999")["case_id"]
    assert desk.case(case_id)["order"] is None


def test_customer_lists_only_their_own_cases_newest_first(desk):
    first = desk.chat("My order arrived late.")["case_id"]
    second = desk.chat("One of my wraps is missing.")["case_id"]
    desk.chat("Where is my order?", customer="demo-customer-102", order_id="demo-order-1002")
    cases = desk.send("GET", "/api/v1/cases", "demo-customer-101").json()["cases"]
    assert [item["case_id"] for item in cases] == [second, first]
    assert cases[0]["order_id"] == "demo-order-1001" and cases[0]["issues"] == ["missing_item"]
    assert "customer_id" not in cases[0] and "review_notes" not in cases[0]
    assert len(desk.send("GET", "/api/v1/cases?limit=1", "demo-customer-101").json()["cases"]) == 1
    desk.send("GET", "/api/v1/cases", "demo-agent-301", expect=403)


def test_agent_reply_reaches_the_customer_without_internal_details(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    result = desk.reply(case_id, "Sorry about the wrap. I've checked the order and refunded it.").json()
    assert result == {"case_id": case_id, "status": "needs_review", "message_count": 3}

    customer_view = desk.send("GET", f"/api/v1/cases/{case_id}", "demo-customer-101").json()
    last = customer_view["messages"][-1]
    assert last["role"] == "support_agent" and "author_id" not in last
    assert "transitions" not in customer_view and "review_notes" not in customer_view
    assert desk.case(case_id)["messages"][-1]["author_id"] == "demo-agent-301"


def test_resolving_records_who_and_why_and_leaves_the_queue(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    assert desk.resolve(case_id, "Refunded the wrap after checking.").json()["status"] == "resolved"
    assert desk.queue()["cases"] == []
    assert [entry["case_id"] for entry in desk.queue(status="resolved")["cases"]] == [case_id]
    transition = desk.case(case_id)["transitions"][-1]
    assert (transition["to_status"], transition["actor_role"], transition["actor_id"], transition["note"]) == \
        ("resolved", "support_agent", "demo-agent-301", "Refunded the wrap after checking.")
    assert desk.send("GET", f"/api/v1/cases/{case_id}", "demo-customer-101").json()["status"] == "resolved"


def test_resolved_case_takes_no_agent_replies_or_second_resolution(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    desk.resolve(case_id)
    assert desk.reply(case_id, "One more thing.", expect=409).json() == {"detail": "This case is resolved"}
    desk.resolve(case_id, agent="demo-agent-302", expect=409)
    assert len(desk.case(case_id)["transitions"]) == 2


def test_customer_message_puts_a_resolved_case_back_in_the_queue(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    desk.resolve(case_id)
    turn = desk.chat("Actually the drink was missing too.", case_id=case_id)
    assert turn["case_status"] == "needs_review"
    statuses = [(item["from_status"], item["to_status"], item["actor_role"]) for item in desk.case(case_id)["transitions"]]
    assert statuses == [("open", "needs_review", "assistant"), ("needs_review", "resolved", "support_agent"),
                        ("resolved", "needs_review", "customer")]
    assert turn["needs_human_review"] is True
    assert [entry["case_id"] for entry in desk.queue()["cases"]] == [case_id]


def test_follow_up_without_an_issue_also_reopens_the_case(desk):
    case_id = desk.chat("Where is my order?")["case_id"]
    desk.resolve(case_id)
    turn = desk.chat("Thanks!", case_id=case_id)
    assert turn["case_status"] == "needs_review"
    assert turn["reply"].endswith("I've reopened your case so a support agent can look at your message.")


def test_reopened_case_joins_the_end_of_the_queue(desk):
    first = desk.chat("One of my wraps is missing.")["case_id"]
    second = desk.chat("My order arrived late.")["case_id"]
    desk.resolve(first)
    desk.chat("The drink was missing too.", case_id=first)
    assert [entry["case_id"] for entry in desk.queue()["cases"]] == [second, first]


def test_agent_actions_are_idempotent(desk):
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    first = desk.reply(case_id, "Checking now.", key="reply-1")
    retry = desk.reply(case_id, "Checking now.", key="reply-1")
    assert retry.json() == first.json() and retry.headers["Idempotent-Replayed"] == "true"
    assert len(desk.case(case_id)["messages"]) == 3
    desk.reply(case_id, "Something else.", key="reply-1", expect=422)

    desk.resolve(case_id, key="resolve-1")
    assert desk.resolve(case_id, key="resolve-1").headers["Idempotent-Replayed"] == "true"


def test_transitions_survive_a_restart(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    store = CaseStore(path, clock=Clock())
    app = create_app(TokenRegistry(TOKENS), case_store=store)
    desk = Desk.__new__(Desk)
    desk.store, desk.app = store, app
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    desk.reply(case_id, "Checking now.")
    desk.resolve(case_id)
    store.close()

    reopened = CaseStore(path).get(case_id)
    assert [item.to_status.value for item in reopened.transitions] == ["needs_review", "resolved"]
    assert reopened.messages[-1].author_id == "demo-agent-301"


# --- the assistant after a support agent replied -----------------------------------------------------

class CapturingAgent:
    def __init__(self):
        self.requests = []

    def draft(self, request):
        self.requests.append(request)
        return AgentDraft("Agent reply.", None, [], 1)


def test_assistant_sees_the_support_agents_message():
    agent = CapturingAgent()
    desk = Desk(reply_agent=agent)
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    desk.reply(case_id, "I've passed this to the restaurant.")
    desk.chat("Thanks, any news?", case_id=case_id)
    history = agent.requests[-1].history
    assert (history[-1].role.value, history[-1].text) == ("support_agent", "I've passed this to the restaurant.")
    assert json.dumps([message.role.value for message in history]) == \
        '["customer", "assistant", "support_agent"]'


# --- reply traces --------------------------------------------------------------------------------

class TracedAgent:
    """Answers with the given drafts in order; an exception is raised instead of answering."""

    def __init__(self, *drafts):
        self.drafts = list(drafts)
        self.llm = type("LLM", (), {"model": "test-model"})()

    def draft(self, request):
        item = self.drafts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_agent_sees_how_each_reply_came_about_and_customers_do_not():
    from resolvelane.reply_agent import AgentFailed
    from resolvelane.tools import ToolTrace

    draft = AgentDraft("Sorry about the wrap; a support agent will review it.", None,
                       [ToolTrace("get_support_policy", True, 12.34), ToolTrace("search_help_center", False, 3.0,
                                                                                 "timeout")],
                       2, skills=("missing-item@3",))
    desk = Desk(reply_agent=TracedAgent(draft, AgentFailed("ungrounded_number")))
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    desk.chat("It was the vegetable wrap.", case_id=case_id)

    replies = [message for message in desk.case(case_id)["messages"] if message["role"] == "assistant"]
    first, second = (message["trace"] for message in replies)
    assert first["written_by"] == "agent" and first["fallback"] == "none" and first["model"] == "test-model"
    assert first["recognition"] == "keywords" and first["issues"] == ["missing_item"]
    assert first["rounds"] == 2 and first["skills"] == ["missing-item@3"] and first["repeat"] is False
    assert first["tools"] == [{"name": "get_support_policy", "ok": True, "latency_ms": 12.3, "error": ""},
                              {"name": "search_help_center", "ok": False, "latency_ms": 3.0, "error": "timeout"}]
    assert first["seconds"] >= 0
    assert second["written_by"] == "rules" and second["fallback"] == "ungrounded_number"
    assert second["model"] == "test-model" and second["tools"] == []
    customer_messages = desk.send("GET", f"/api/v1/cases/{case_id}", "demo-customer-101").json()["messages"]
    assert all("trace" not in message for message in customer_messages)
    assert all(message["trace"] is None for message in desk.case(case_id)["messages"]
               if message["role"] != "assistant")


def test_rule_based_reply_without_an_agent_is_not_a_fallback(desk):
    case_id = desk.chat("My order arrived late.")["case_id"]
    trace = desk.case(case_id)["messages"][1]["trace"]
    assert (trace["written_by"], trace["fallback"], trace["model"]) == ("rules", "none", None)


def test_traces_survive_a_restart(tmp_path):
    path = str(tmp_path / "cases.sqlite3")
    store = CaseStore(path, clock=Clock())
    desk = Desk.__new__(Desk)
    desk.store, desk.app = store, create_app(TokenRegistry(TOKENS), case_store=store,
                                             reply_agent=TracedAgent(AgentDraft("Noted.", None, [], 1)))
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    store.close()
    trace = CaseStore(path).get(case_id).messages[1].trace
    assert trace.written_by == "agent" and trace.issues == (Issue.MISSING_ITEM,) and trace.model == "test-model"


class MeteredAgent(TracedAgent):
    """A traced agent that spends tokens the way a real one does: through create_message."""

    def draft(self, request):
        from types import SimpleNamespace

        from resolvelane.llm import create_message
        answer = SimpleNamespace(usage=SimpleNamespace(input_tokens=1500, output_tokens=120), content=[])
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **request: answer))
        create_message(client, "reply", model="test-model", max_tokens=100, messages=[])
        return super().draft(request)


def test_each_reply_records_the_model_usage_it_cost_even_when_its_draft_is_not_used(monkeypatch):
    from resolvelane.llm import load_prices
    from resolvelane.reply_agent import AgentFailed

    monkeypatch.setenv("RESOLVELANE_MODEL_PRICES", '{"test-model": {"input": 2.0, "output": 10.0}}')
    try:
        desk = Desk(reply_agent=MeteredAgent(AgentDraft("Noted; a support agent will review it.", None, [], 1),
                                             AgentFailed("ungrounded_number")))
        case_id = desk.chat("One of my wraps is missing.")["case_id"]
        desk.chat("It was the vegetable wrap.", case_id=case_id)
    finally:
        load_prices({})
    first, second = (message["trace"] for message in desk.case(case_id)["messages"] if message["role"] == "assistant")
    assert first["usage"] == {"calls": 1, "input_tokens": 1500, "output_tokens": 120, "cache_read_tokens": 0,
                              "cache_write_tokens": 0, "cost_usd": 0.0042}
    # The rule-based reply was sent, but the agent's discarded draft was paid for.
    assert second["written_by"] == "rules" and second["usage"]["calls"] == 1 and second["usage"]["cost_usd"] == 0.0042


def test_usage_without_prices_is_counted_in_tokens_only_and_old_traces_have_none(tmp_path):
    import sqlite3

    path = str(tmp_path / "cases.sqlite3")
    store = CaseStore(path, clock=Clock())
    desk = Desk.__new__(Desk)
    desk.store, desk.app = store, create_app(TokenRegistry(TOKENS), case_store=store,
                                             reply_agent=MeteredAgent(AgentDraft("Noted.", None, [], 1)))
    case_id = desk.chat("One of my wraps is missing.")["case_id"]
    trace = desk.case(case_id)["messages"][1]["trace"]
    assert trace["usage"]["input_tokens"] == 1500 and trace["usage"]["cost_usd"] is None
    store.close()

    # A trace stored before usage was recorded.
    with sqlite3.connect(path) as db:
        stored = json.loads(db.execute("SELECT trace FROM case_messages WHERE trace IS NOT NULL").fetchone()[0])
        del stored["usage"]
        db.execute("UPDATE case_messages SET trace = ? WHERE trace IS NOT NULL", (json.dumps(stored),))
    assert CaseStore(path).get(case_id).messages[1].trace.usage is None
