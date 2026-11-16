import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import anthropic
import httpx2

from resolvelane.case_store import CaseStore
from resolvelane.domain.case import CaseMessage, CaseStatus, CaseSummary, EarlierCase, Issue, MessageRole
from resolvelane.llm import LLMProvider
from resolvelane.memory import (KEEP_RECENT, RECENT_WINDOW, SUMMARIZE_AT, CaseSummarizer, SummaryQueue,
                                earlier_cases, needs_summary, recent_messages)
from resolvelane.recognition import KeywordRecognizer
from resolvelane.reply_agent import AgentDraft, ReplyAgent, ReplyRequest, case_facts
from resolvelane.resilience import CircuitBreaker
from resolvelane.services.support_chat import handle_message
from resolvelane.simulator.fixtures import ORDERS, POLICIES
from resolvelane.simulator.order_source import InMemoryOrderSource

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
CUSTOMER = "demo-customer-101"


def messages(count, start=0):
    return [CaseMessage(MessageRole.CUSTOMER if n % 2 == 0 else MessageRole.ASSISTANT, f"message {n}", NOW)
            for n in range(start, start + count)]


def summary(covers, version=1, text="The customer says a wrap was missing."):
    return CaseSummary(text, covers, version, NOW)


def stored_case(cases, count, customer=CUSTOMER, opened=NOW, **fields):
    case = cases.new_case(customer)
    case.created_at = opened
    case.messages = messages(count)
    for name, value in fields.items():
        setattr(case, name, value)
    cases.commit(case)
    return case


# --- which messages the agent sees -------------------------------------------------

def test_short_conversation_is_sent_whole():
    assert recent_messages(messages(5), None) == messages(5)


def test_messages_covered_by_the_summary_are_left_out():
    assert recent_messages(messages(10), summary(6)) == messages(4, start=6)


def test_summarizer_falling_behind_keeps_only_the_newest_messages(caplog):
    with caplog.at_level(logging.INFO, logger="resolvelane.memory"):
        recent = recent_messages(messages(RECENT_WINDOW + 3), None)
    assert recent == messages(RECENT_WINDOW, start=3)
    assert any("memory_gap" in record.getMessage() for record in caplog.records)


def test_summary_is_needed_once_enough_messages_are_uncovered():
    assert not needs_summary(SUMMARIZE_AT - 1, None)
    assert needs_summary(SUMMARIZE_AT, None)
    assert not needs_summary(SUMMARIZE_AT + 5, summary(6))
    assert needs_summary(SUMMARIZE_AT + 6, summary(6))


# --- writing summaries ---------------------------------------------------------------

def text(value, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=value)])


class ScriptedClient:
    def __init__(self, *outcomes, during=None):
        self.outcomes = list(outcomes)
        self.calls = []
        self.messages = self
        self.during = during

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.during:
            self.during()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def summarizer(cases, *outcomes, during=None):
    client = ScriptedClient(*outcomes, during=during)
    llm = LLMProvider(client, "test-model", CircuitBreaker("llm", failure_threshold=1, cooldown_s=30))
    return CaseSummarizer(llm, cases), client


def request():
    return httpx2.Request("POST", "https://api.example.test/v1/messages")


def test_older_messages_are_folded_into_a_summary():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, SUMMARIZE_AT)
    writer, client = summarizer(cases, text("The customer says wrap 1 was missing."))
    written = writer.summarize(case.case_id)
    assert written == CaseSummary("The customer says wrap 1 was missing.", SUMMARIZE_AT - KEEP_RECENT, 1, NOW)
    assert cases.summary(case.case_id) == written
    prompt = client.calls[0]["messages"][0]["content"]
    assert "message 0" in prompt and f"message {SUMMARIZE_AT - KEEP_RECENT - 1}" in prompt
    assert f"message {SUMMARIZE_AT - KEEP_RECENT}" not in prompt  # recent messages stay word for word
    assert len(cases.get(case.case_id).messages) == SUMMARIZE_AT  # messages are never removed


def test_next_summary_merges_the_previous_one():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, 20)
    cases.save_summary(case.case_id, summary(6, text="Earlier: wrap missing."))
    writer, client = summarizer(cases, text("Wrap missing; customer asked about refund timing."))
    written = writer.summarize(case.case_id)
    assert (written.version, written.covers) == (2, 20 - KEEP_RECENT)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Earlier: wrap missing." in prompt
    assert "message 6" in prompt and "message 5\n" not in prompt


def test_short_conversation_is_not_summarized():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, SUMMARIZE_AT - 1)
    writer, client = summarizer(cases, text("unused"))
    assert writer.summarize(case.case_id) is None and client.calls == []


def test_failed_summary_writes_nothing_and_opens_the_breaker():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, SUMMARIZE_AT)
    writer, client = summarizer(cases, anthropic.APIConnectionError(request=request()))
    assert writer.summarize(case.case_id) is None
    assert cases.summary(case.case_id) is None
    assert writer.summarize(case.case_id) is None and len(client.calls) == 1  # breaker open: no second call


def test_unusable_summary_is_not_stored():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, SUMMARIZE_AT)
    for outcome in (text("Cut off", stop="max_tokens"), text("   "), text("x" * 1001)):
        writer, _ = summarizer(cases, outcome)
        assert writer.summarize(case.case_id) is None
    assert cases.summary(case.case_id) is None


def test_summary_written_meanwhile_wins():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, SUMMARIZE_AT)

    def another_summary_meanwhile():
        cases.save_summary(case.case_id, summary(6, text="Written by another worker."))

    writer, _ = summarizer(cases, text("Late summary."), during=another_summary_meanwhile)
    assert writer.summarize(case.case_id) is None
    assert cases.summary(case.case_id).text == "Written by another worker."


# --- the background queue ------------------------------------------------------------

class RecordingSummarizer:
    def __init__(self, fail_on=()):
        self.calls = []
        self.release = threading.Event()
        self.fail_on = fail_on

    def summarize(self, case_id):
        self.release.wait(5)
        self.calls.append(case_id)
        if case_id in self.fail_on:
            raise RuntimeError("boom")


def test_queue_runs_each_waiting_case_once():
    recording = RecordingSummarizer()
    summaries = SummaryQueue(recording)
    assert summaries.submit("case-a")
    while "case-a" in summaries._waiting:  # the worker has taken case-a and is blocked in it
        threading.Event().wait(0.01)
    assert summaries.submit("case-b")
    assert not summaries.submit("case-b")
    recording.release.set()
    summaries.join()
    assert recording.calls == ["case-a", "case-b"]
    summaries.close()


def test_queue_keeps_working_after_an_error():
    recording = RecordingSummarizer(fail_on={"case-a"})
    recording.release.set()
    summaries = SummaryQueue(recording)
    summaries.submit("case-a")
    summaries.submit("case-b")
    summaries.join()
    assert recording.calls == ["case-a", "case-b"]
    summaries.close()


def test_full_queue_drops_new_requests():
    recording = RecordingSummarizer()
    summaries = SummaryQueue(recording, max_pending=1)
    summaries.submit("case-a")
    while "case-a" in summaries._waiting:
        threading.Event().wait(0.01)
    assert summaries.submit("case-b")
    assert not summaries.submit("case-c")
    recording.release.set()
    summaries.join()
    summaries.close()


# --- earlier cases -------------------------------------------------------------------

def test_earlier_cases_are_the_customers_own_recent_cases_newest_first():
    cases = CaseStore(clock=lambda: NOW)
    old = stored_case(cases, 2, opened=NOW - timedelta(days=91))
    first = stored_case(cases, 2, opened=NOW - timedelta(days=10), order_id="demo-order-1001",
                        issues=[Issue.MISSING_ITEM], status=CaseStatus.NEEDS_REVIEW)
    second = stored_case(cases, 2, opened=NOW - timedelta(days=2))
    current = stored_case(cases, 2, opened=NOW)
    stored_case(cases, 2, customer="demo-customer-102", opened=NOW - timedelta(days=1))

    found = earlier_cases(cases, CUSTOMER, current.case_id, NOW)
    assert [case.case_id for case in found] == [second.case_id, first.case_id]
    assert found[1] == EarlierCase(first.case_id, "demo-order-1001", (Issue.MISSING_ITEM,),
                                   CaseStatus.NEEDS_REVIEW, NOW - timedelta(days=10))
    assert old.case_id not in [case.case_id for case in found]


def test_at_most_five_earlier_cases_are_used():
    cases = CaseStore(clock=lambda: NOW)
    for days in range(8):
        stored_case(cases, 0, opened=NOW - timedelta(days=days + 1))
    assert len(earlier_cases(cases, CUSTOMER, None, NOW)) == 5


def test_earlier_cases_are_case_facts():
    earlier = EarlierCase("case-earlier", "demo-order-1001", (Issue.MISSING_ITEM,), CaseStatus.NEEDS_REVIEW,
                          datetime(2026, 2, 20, 9, 30, tzinfo=timezone.utc))
    facts = case_facts(ReplyRequest(CUSTOMER, "Any news?", [], None, [Issue.UNKNOWN], [], CaseStatus.OPEN, False,
                                    "reference", NOW, earlier_cases=[earlier]), POLICIES)
    assert facts["earlier_cases"] == [{"case_id": "case-earlier", "order_id": "demo-order-1001",
                                       "issues": ["missing_item"], "status": "needs_review",
                                       "opened_on": "2026-02-20"}]


# --- in a conversation -----------------------------------------------------------------

class CapturingAgent:
    def __init__(self):
        self.requests = []

    def draft(self, request):
        self.requests.append(request)
        return AgentDraft("Agent reply.", None, [], 1)


class RecordingQueue:
    def __init__(self):
        self.submitted = []

    def submit(self, case_id):
        self.submitted.append(case_id)
        return True


def send(cases, message, case_id=None, agent=None, summaries=None, order_id=None):
    return handle_message(cases, InMemoryOrderSource(ORDERS), KeywordRecognizer(), CUSTOMER, message, case_id,
                          order_id, agent, summaries=summaries)


def test_agent_gets_the_summary_and_the_messages_it_does_not_cover():
    cases = CaseStore(clock=lambda: NOW)
    case = stored_case(cases, 10, order_id="demo-order-1001")
    cases.save_summary(case.case_id, summary(6))
    agent = CapturingAgent()
    send(cases, "Any news?", case.case_id, agent)
    request = agent.requests[0]
    assert request.summary == "The customer says a wrap was missing."
    assert [message.text for message in request.history] == ["message 6", "message 7", "message 8", "message 9"]


def test_new_case_gets_the_customers_earlier_cases():
    cases = CaseStore(clock=lambda: NOW)
    earlier = stored_case(cases, 2, opened=NOW - timedelta(days=1), order_id="demo-order-1001",
                          issues=[Issue.MISSING_ITEM], status=CaseStatus.NEEDS_REVIEW)
    agent = CapturingAgent()
    send(cases, "I reported a missing wrap yesterday. Any news?", agent=agent)
    assert [case.case_id for case in agent.requests[0].earlier_cases] == [earlier.case_id]
    assert agent.requests[0].summary is None


def test_long_conversation_queues_a_summary():
    cases = CaseStore(clock=lambda: NOW)
    summaries = RecordingQueue()
    case_id = send(cases, "Where is my order?", order_id="demo-order-1001", summaries=summaries).case_id
    for n in range(SUMMARIZE_AT // 2 - 1):
        assert summaries.submitted == []
        send(cases, f"Follow-up {n}", case_id, summaries=summaries)
    assert summaries.submitted == [case_id]  # the turn that reached SUMMARIZE_AT messages


def test_prompt_includes_the_summary_only_when_there_is_one():
    class Client:
        messages = None

    agent = ReplyAgent(LLMProvider(Client(), "test-model", CircuitBreaker("llm", 3, 30)), POLICIES)
    base = ReplyRequest(CUSTOMER, "Any news?", messages(2), None, [Issue.UNKNOWN], [], CaseStatus.OPEN, False,
                        "reference", NOW)
    assert "<earlier_summary>" not in agent._prompt(base)
    with_summary = agent._prompt(ReplyRequest(**{**vars(base), "summary": "The customer says it was cold."}))
    assert "<earlier_summary>\nThe customer says it was cold.\n</earlier_summary>" in with_summary
    assert json.loads(with_summary.split("<case_facts>\n")[1].split("\n</case_facts>")[0])["earlier_cases"] == []
