import asyncio
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app
from resolvelane.case_store import CaseStore
from resolvelane.domain.case import CaseStatus, Issue
from resolvelane.llm import LLMProvider
from resolvelane.recognition import KeywordRecognizer, load_recognizer
from resolvelane.reply_agent import (REVIEW_SENTENCE, SYSTEM_PROMPT, AgentDraft, AgentFailed, ReplyAgent,
                                     ReplyRequest)
from resolvelane.resilience import BreakerState, CircuitBreaker
from resolvelane.simulator.fixtures import ORDERS, POLICIES

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
ORDER = next(order for order in ORDERS if order.order_id == "demo-order-1001")


def text(value, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="thinking", thinking=""),
                                                      SimpleNamespace(type="text", text=value)])


def tool_use(name, args, call_id="call-1"):
    return SimpleNamespace(stop_reason="tool_use",
                           content=[SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)])


class ScriptedClient:
    """Stands in for anthropic.Anthropic: returns (or raises) the scripted outcomes in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs, default=lambda value: vars(value))))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def provider(client):
    return LLMProvider(client, "test-model", CircuitBreaker("llm", failure_threshold=3, cooldown_s=30))


def request(**overrides):
    values = dict(customer_id="demo-customer-101", message="Was my order late?", history=[], order=ORDER,
                  issues=[Issue.LATE_DELIVERY], case_issues=[Issue.LATE_DELIVERY, Issue.DAMAGED_ITEM],
                  case_status=CaseStatus.OPEN, review_required=True,
                  reference_reply="Order demo-order-1001 was delivered 15 minutes after the estimated time. "
                                  "I've passed this to a support agent for review.", now=NOW)
    return ReplyRequest(**{**values, **overrides})


def draft(client, **overrides):
    return ReplyAgent(provider(client), POLICIES).draft(request(**overrides))


def _request():
    return httpx2.Request("POST", "https://api.example.test/v1/messages")


# --- the agent on its own ----------------------------------------------------

def test_reply_is_grounded_in_case_facts():
    client = ScriptedClient(text("Yes, it arrived 15 minutes late. A support agent will review your case."))
    result = draft(client)
    assert result.reply == "Yes, it arrived 15 minutes late. A support agent will review your case."
    assert result.rounds == 1 and result.review_request is None
    call = client.calls[0]
    assert call["system"] == SYSTEM_PROMPT and "temperature" not in call
    assert {tool["name"] for tool in call["tools"]} == {"search_help_center", "get_support_policy",
                                                        "request_human_review", "propose_refund"}
    prompt = call["messages"][0]["content"]
    facts = json.loads(prompt.split("<case_facts>\n")[1].split("\n</case_facts>")[0])
    assert facts["delivered_minutes_after_estimate"] == 15
    assert facts["review_required"] is True
    # Policies for the case's issues come with the facts, so no tool call is needed for them.
    assert facts["policies"]["late_delivery"]["policy"]["title"] == "Late delivery review"
    assert facts["policies"]["damaged_item"]["policy"] is None  # expired
    assert "<customer_message>\nWas my order late?\n</customer_message>" in prompt
    assert "demo-customer-101" not in prompt  # internal IDs stay out of the model's view


def test_policy_lookup_round_trip():
    client = ScriptedClient(tool_use("get_support_policy", {"issue": "late_delivery"}),
                            text("Late deliveries are reviewed by a support agent, who decides on compensation."))
    result = draft(client)
    assert result.rounds == 2
    assert [(trace.name, trace.ok) for trace in result.tool_traces] == [("get_support_policy", True)]
    tool_result = client.calls[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "call-1" and tool_result["is_error"] is False
    assert "Late delivery review" in tool_result["content"]
    assert client.calls[1]["messages"][1]["role"] == "assistant"


def test_bad_tool_call_is_reported_to_the_model_and_the_reply_continues():
    client = ScriptedClient(tool_use("issue_refund", {"amount": 500}), text("A support agent will review it."))
    result = draft(client)
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert result.tool_traces[0].error == "unknown_tool"


def test_review_requested_by_the_agent_is_returned_not_applied():
    client = ScriptedClient(tool_use("request_human_review", {"reason": "Customer asked for a person"}),
                            text("I've asked a colleague to look at this."))
    result = draft(client, review_required=False, issues=[Issue.UNKNOWN])
    assert result.review_request.reason == "Customer asked for a person"
    assert result.reply == f"I've asked a colleague to look at this. {REVIEW_SENTENCE}"


def test_review_notice_is_added_when_the_model_leaves_it_out():
    result = draft(ScriptedClient(text("It arrived 15 minutes late.")))
    assert result.reply == f"It arrived 15 minutes late. {REVIEW_SENTENCE}"


@pytest.mark.parametrize("reply", [
    "We will refund you right away.",
    "I'll issue a full refund now.",
    "Your refund has been approved.",
    "You'll receive a credit for the delay.",
    "We have sent a replacement.",
])
def test_replies_that_commit_to_an_outcome_are_rejected(reply):
    with pytest.raises(AgentFailed, match="commits_to_an_outcome"):
        draft(ScriptedClient(text(reply)))


@pytest.mark.parametrize("reply", [
    "A support agent will review your refund request.",
    "I can't promise a refund; a support agent decides that.",
    "We will review your case and reply here.",
    "Refunds are not issued automatically.",
])
def test_replies_that_leave_the_outcome_to_an_agent_are_kept(reply):
    assert draft(ScriptedClient(text(reply))).reply.startswith(reply)


@pytest.mark.parametrize("outcome,reason", [
    (text(""), "empty_reply"),
    (text("x" * 1300), "reply_too_long"),
    (text("Partial answer", stop="max_tokens"), "stop_max_tokens"),
    (text("No", stop="refusal"), "stop_refusal"),
    (tool_use("get_support_policy", {"issue": "late_delivery"}), "too_many_rounds"),
    (anthropic.APIConnectionError(request=_request()), "APIConnectionError"),
    (anthropic.APIStatusError("overloaded", response=httpx2.Response(529, request=_request()), body=None),
     "status_529"),
])
def test_unusable_outcomes_fail(outcome, reason):
    with pytest.raises(AgentFailed, match=reason):
        draft(ScriptedClient(outcome))


def test_tool_call_limit():
    calls = SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="tool_use", id=f"call-{n}", name="get_support_policy", input={"issue": "late_delivery"})
        for n in range(6)])
    client = ScriptedClient(calls, text("A support agent will review it."))
    result = draft(client)
    assert len(result.tool_traces) == ReplyAgent.MAX_TOOL_CALLS
    results = client.calls[1]["messages"][2]["content"]
    assert [item["is_error"] for item in results] == [False] * 4 + [True] * 2


def test_provider_errors_count_against_the_breaker_but_bad_requests_do_not():
    llm = provider(ScriptedClient(anthropic.APIStatusError("bad", response=httpx2.Response(400, request=_request()),
                                                           body=None)))
    agent = ReplyAgent(llm, POLICIES)
    for _ in range(3):
        with pytest.raises(AgentFailed):
            agent.draft(request())
    assert llm.breaker.state is BreakerState.CLOSED


def test_open_breaker_skips_the_model():
    client = ScriptedClient(text("unused"))
    llm = provider(client)
    for _ in range(3):
        llm.breaker.failed(llm.breaker.acquire())
    with pytest.raises(AgentFailed, match="circuit_open"):
        ReplyAgent(llm, POLICIES).draft(request())
    assert client.calls == []


# --- help center answers --------------------------------------------------------

from resolvelane.knowledge import HelpCenter  # noqa: E402
from resolvelane.reply_agent import ungrounded_numbers  # noqa: E402
from resolvelane.simulator.help_center import ARTICLES  # noqa: E402


def answer(reply, *sources):
    return text(json.dumps({"reply": reply, "sources": list(sources)}))


def help_draft(*outcomes, **overrides):
    agent = ReplyAgent(provider(ScriptedClient(*outcomes)), POLICIES, HelpCenter(ARTICLES))
    return agent.draft(request(**{"review_required": False, "issues": [Issue.UNKNOWN], **overrides}))


SEARCH = tool_use("search_help_center", {"query": "refund arrival time"})


def test_answer_from_the_help_center_carries_its_citation():
    result = help_draft(SEARCH, answer("Once approved, a refund reaches your card within 3 to 5 business days.",
                                       "refund-timing@2#arrival"))
    assert result.reply.startswith("Once approved")
    assert [citation.source_id for citation in result.citations] == ["refund-timing@2#arrival"]
    assert result.citations[0].version == 2


def test_citing_a_passage_that_was_not_retrieved_is_rejected():
    with pytest.raises(AgentFailed, match="cites_unretrieved_source"):
        help_draft(answer("Refunds take 3 to 5 business days.", "refund-timing@2#arrival"))


def test_citing_an_outdated_version_is_rejected():
    with pytest.raises(AgentFailed, match="cites_unretrieved_source"):
        help_draft(SEARCH, answer("Refunds take 7 to 10 business days.", "refund-timing@1#arrival"))


def test_number_from_an_uncited_passage_is_rejected():
    with pytest.raises(AgentFailed, match="ungrounded_number"):
        help_draft(SEARCH, answer("A refund reaches your card within 3 to 5 business days."))


def test_invented_number_is_rejected_even_with_a_citation():
    with pytest.raises(AgentFailed, match="ungrounded_number"):
        help_draft(SEARCH, answer("A refund reaches your card within 9 business days.", "refund-timing@2#arrival"))


@pytest.mark.parametrize("bad", ['{"reply": 5}', '{"reply": "Hi", "sources": "refund-timing@2#arrival"}',
                                 '{"reply": "Hi", "sources": [3]}'])
def test_malformed_answers_are_rejected(bad):
    with pytest.raises(AgentFailed, match="invalid_"):
        help_draft(text(bad))


@pytest.mark.parametrize("reply,ok", [
    ("Delivered at 18:55 UTC on Jan 15, 15 minutes late.", True),   # from 2026-01-15T18:55:00 and the facts
    ("Your total was 24.50 USD.", None),                            # depends on the order's amounts
    ("It will arrive at 19:30.", False),
    ("Order demo-order-1001 is on its way.", True),
    ("It is 07:50 now.", True),                                     # from a timestamp with microseconds
])
def test_numbers_must_appear_in_the_sources(reply, ok):
    sources = '{"now": "2026-11-13T07:50:50.285317+00:00", ' \
              '"delivered_at": "2026-01-15T18:55:00+00:00", "delivered_minutes_after_estimate": 15, ' \
              '"order_id": "demo-order-1001", "total": "24.50"}'
    missing = ungrounded_numbers(reply, sources)
    assert (missing == []) is (ok is not False)


def test_order_amounts_are_given_as_written_to_customers():
    client = ScriptedClient(text("Your order total is shown on the receipt."))
    ReplyAgent(provider(client), POLICIES).draft(request())
    facts = json.loads(client.calls[0]["messages"][0]["content"].split("<case_facts>\n")[1].split("\n</case_facts>")[0])
    total = ORDER.total_minor / 100
    assert facts["amounts"]["total"] == f"{total:.2f}"


# --- the agent in a conversation -----------------------------------------------

TOKEN = "test-token-demo-customer-101-xxxxxxxx"


class Api:
    def __init__(self, reply_agent=None, recognizer=None):
        self.app = create_app(token_registry=TokenRegistry({TOKEN: {"subject_id": "demo-customer-101",
                                                                     "role": "customer"}}),
                              case_store=CaseStore(clock=lambda: NOW),
                              recognizer=recognizer or KeywordRecognizer(), reply_agent=reply_agent)
        self.store = self.app.state.case_store

    def chat(self, message, order_id=None, case_id=None):
        body = {"message": message, **({"order_id": order_id} if order_id else {}),
                **({"case_id": case_id} if case_id else {})}

        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://testserver") as client:
                return await client.post("/api/v1/chat", json=body, headers={"Authorization": f"Bearer {TOKEN}"})

        response = asyncio.run(send())
        assert response.status_code == 200, response.text
        return response.json()


def test_cited_answer_is_returned_and_kept_with_the_message():
    client = ScriptedClient(SEARCH, answer("Once approved, a refund reaches your card within 3 to 5 business days.",
                                           "refund-timing@2#arrival"))
    api = Api(ReplyAgent(provider(client), POLICIES, HelpCenter(ARTICLES)))
    turn = api.chat("How long do refunds take?")
    assert turn["citations"] == [{"article_id": "refund-timing", "version": 2, "title": "How refunds work",
                                  "section": "arrival"}]
    message = api.store.get(turn["case_id"]).messages[-1]
    assert [citation.source_id for citation in message.citations] == ["refund-timing@2#arrival"]


def test_agent_reply_is_sent_and_recorded():
    api = Api(ReplyAgent(provider(ScriptedClient(text("Sorry, it was 15 minutes late. A support agent will "
                                                      "review your case."))), POLICIES))
    turn = api.chat("My order arrived late.", "demo-order-1001")
    assert turn["reply"] == "Sorry, it was 15 minutes late. A support agent will review your case."
    assert turn["case_status"] == "needs_review" and turn["needs_human_review"] is True
    assert api.store.get(turn["case_id"]).messages[-1].text == turn["reply"]


def test_failed_agent_falls_back_to_the_rule_based_reply():
    rules = Api().chat("My order arrived late.", "demo-order-1001")
    failing = Api(ReplyAgent(provider(ScriptedClient(anthropic.APIConnectionError(request=_request()))), POLICIES))
    turn = failing.chat("My order arrived late.", "demo-order-1001")
    assert turn["reply"] == rules["reply"]
    assert turn["case_status"] == "needs_review"


def test_agent_review_request_moves_the_case_to_review():
    client = ScriptedClient(tool_use("request_human_review", {"reason": "Customer asked for a person"}),
                            text("I've asked a support agent to review your case."))
    api = Api(ReplyAgent(provider(client), POLICIES))
    turn = api.chat("Can I talk to a real person about demo-order-1001?")
    assert turn["case_status"] == "needs_review" and turn["needs_human_review"] is True
    case = api.store.get(turn["case_id"])
    assert case.review_notes == ["Customer asked for a person"]


def test_without_an_order_the_agent_sees_no_order_facts():
    client = ScriptedClient(tool_use("request_human_review", {"reason": "Customer wants a person"}),
                            text("Please share your order number. I've also asked a support agent to help."))
    turn = Api(ReplyAgent(provider(client), POLICIES)).chat("I want to talk to a human.")
    facts = json.loads(client.calls[0]["messages"][0]["content"].split("<case_facts>\n")[1].split("\n</case_facts>")[0])
    assert facts["order"] is None and "delivered_minutes_after_estimate" not in facts
    assert turn["case_status"] == "needs_review" and turn["order_id"] is None


def test_redirect_to_another_order_does_not_use_the_agent():
    client = ScriptedClient(text("Sure, it arrived late. A support agent will review your case."))
    api = Api(ReplyAgent(provider(client), POLICIES))
    first = api.chat("Where is it?", "demo-order-1001")
    calls = len(client.calls)
    turn = api.chat("And demo-order-1003?", case_id=first["case_id"])
    assert turn["reply"].startswith("This case is about order demo-order-1001.")
    assert len(client.calls) == calls


def test_recognition_and_replies_share_one_breaker():
    client = ScriptedClient(anthropic.APIConnectionError(request=_request()))
    llm = provider(client)
    recognizer = load_recognizer({"RESOLVELANE_LOCAL_EMBEDDING_MODEL": "off"}, llm=llm)
    api = Api(ReplyAgent(llm, POLICIES), recognizer)
    api.chat("My order arrived late.", "demo-order-1001")  # recognition fails, then the reply fails
    api.chat("My order arrived late.", "demo-order-1001")  # recognition fails a third time: breaker opens
    assert len(client.calls) == 3 and llm.breaker.state is BreakerState.OPEN
    turn = api.chat("My order arrived late.", "demo-order-1001")
    assert len(client.calls) == 3  # neither recognition nor the reply waits for the provider now
    assert "15 minutes after" in turn["reply"]


class StubAgent:
    def __init__(self, during=None):
        self.during = during

    def draft(self, request):
        if self.during:
            self.during()
        return AgentDraft("Agent reply.", None, [], 1)


def test_agent_call_does_not_hold_the_case_lock():
    api = Api()
    acquired = []

    def other_thread_takes_the_lock():
        def take():
            got = api.store.lock.acquire(timeout=1)
            acquired.append(got)
            if got:
                api.store.lock.release()
        worker = threading.Thread(target=take)
        worker.start()
        worker.join()

    api.app.state.reply_agent = StubAgent(other_thread_takes_the_lock)
    assert api.chat("Where is my order?", "demo-order-1001")["reply"] == "Agent reply."
    assert acquired == [True]


def test_case_changed_while_the_agent_wrote_gets_the_rule_based_reply():
    api = Api()
    first = api.chat("Where is my order?", "demo-order-1001")

    def another_message_moved_the_case_to_review():
        case = api.store.get(first["case_id"])
        case.status = CaseStatus.NEEDS_REVIEW
        api.store.commit(case)

    api.app.state.reply_agent = StubAgent(another_message_moved_the_case_to_review)
    turn = api.chat("Where is my order?", case_id=first["case_id"])
    assert turn["reply"] != "Agent reply."
    assert turn["reply"].endswith("Your case is waiting for a support agent to review it.")
